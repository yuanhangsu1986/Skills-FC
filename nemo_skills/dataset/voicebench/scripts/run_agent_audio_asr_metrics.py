#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage 2 for VoiceBench S2S runs: agent-audio ASR + WER/CER.

Reads nemo-skills generation output.jsonl, which may include:
  - "generation": model-generated text
  - "audio": {"path": "..."} pointing to agent audio (wav) produced by server

Writes:
  - output_asr.jsonl: same items, but with "generation" replaced by the agent-ASR transcript
    and original generation saved as "generation_text".
  - agent_audio_metrics.json: aggregated agent WER/CER in nemo-skills metric format.

Segmentation + ASR logic is borrowed (lightly adapted) from:
  nemo_skills/dataset/s2s_demo/scripts/eval_conversation_behavior_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text_for_wer(text: str) -> str:
    """A light normalization similar to s2s_demo evaluation code."""
    if not text:
        return ""
    text = text.lower()
    # remove special timing tokens commonly produced by S2S models
    text = re.sub(r"<\$\s*[\d.]+\s*\$>", " ", text)
    text = re.sub(r"<\|\s*[\d.]+\s*\|>", " ", text)
    # normalize punctuation -> spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int
    ref_len: int


def _edit_distance_counts(ref: List[str], hyp: List[str]) -> EditCounts:
    """Compute S/I/D counts for ref->hyp using DP."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = min edits for ref[:i] -> hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]  # type: ignore[var-annotated]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                if best == sub:
                    back[i][j] = "S"
                elif best == ins:
                    back[i][j] = "I"
                else:
                    back[i][j] = "D"

    i, j = n, m
    s = ins = d = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            if op == "S":
                s += 1
            i -= 1
            j -= 1
        elif op == "I":
            ins += 1
            j -= 1
        elif op == "D":
            d += 1
            i -= 1
        else:
            # Should not happen, but keep it safe.
            if i > 0:
                d += 1
                i -= 1
            elif j > 0:
                ins += 1
                j -= 1

    return EditCounts(substitutions=s, insertions=ins, deletions=d, ref_len=n)


def compute_wer(ref_text: str, hyp_text: str) -> Tuple[Optional[float], EditCounts]:
    ref_norm = normalize_text_for_wer(ref_text)
    hyp_norm = normalize_text_for_wer(hyp_text)
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []
    counts = _edit_distance_counts(ref_words, hyp_words)
    if counts.ref_len == 0:
        return (0.0 if not hyp_words else 1.0), counts
    wer = (counts.substitutions + counts.insertions + counts.deletions) / counts.ref_len
    return wer, counts


def compute_cer(ref_text: str, hyp_text: str) -> Tuple[Optional[float], EditCounts]:
    ref_norm = normalize_text_for_wer(ref_text)
    hyp_norm = normalize_text_for_wer(hyp_text)
    ref_chars = list(ref_norm) if ref_norm else []
    hyp_chars = list(hyp_norm) if hyp_norm else []
    counts = _edit_distance_counts(ref_chars, hyp_chars)
    if counts.ref_len == 0:
        return (0.0 if not hyp_chars else 1.0), counts
    cer = (counts.substitutions + counts.insertions + counts.deletions) / counts.ref_len
    return cer, counts


def _load_audio_mono(path: str) -> Tuple["Any", int]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32", copy=False)
    return audio, int(sr)


def _resample_to_16k(audio, sr: int):
    if sr == 16000:
        return audio, 16000
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr, target_sr=16000), 16000
    except Exception:
        import torch
        import torchaudio

        x = torch.from_numpy(audio).float().unsqueeze(0)
        y = torchaudio.functional.resample(x, sr, 16000)
        return y.squeeze(0).cpu().numpy(), 16000


class LazySileroVAD:
    def __init__(self):
        self._vad_model = None
        self._get_speech_timestamps = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def _load(self):
        if self._vad_model is not None:
            return
        import torch

        # This is the same entrypoint used in s2s_demo eval code.
        vad_model, utils = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False)
        vad_model = vad_model.to("cuda" if torch.cuda.is_available() else "cpu")
        get_speech_timestamps, _, _, _, _ = utils
        self._vad_model = vad_model
        self._get_speech_timestamps = get_speech_timestamps

    def speech_segments(self, audio_16k, sr: int) -> List[Tuple[float, float]]:
        """Return list of (start_sec, end_sec). Falls back to full segment on failure."""
        if sr != 16000:
            raise ValueError("VAD expects 16k audio")
        try:
            self._load()
            import torch

            x = torch.from_numpy(audio_16k).float()
            device = next(self._vad_model.parameters()).device  # type: ignore[union-attr]
            x = x.to(device)
            segs = self._get_speech_timestamps(x, self._vad_model, sampling_rate=16000)  # type: ignore[misc]
            out = []
            for s in segs or []:
                start = float(s.get("start", 0)) / 16000.0
                end = float(s.get("end", 0)) / 16000.0
                if end > start:
                    out.append((start, end))
            return out
        except Exception:
            # no VAD available/cached; treat whole audio as one segment
            dur = float(len(audio_16k)) / 16000.0 if audio_16k is not None else 0.0
            return [(0.0, dur)] if dur > 0 else []


class LazyNeMoASR:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr

        m = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        m = m.cuda() if hasattr(m, "cuda") else m
        self._model = m

    def transcribe_segment(self, audio, sr: int, start_sec: float, end_sec: float) -> str:
        """Borrowed approach from s2s_demo: write temp wav then ASRModel.transcribe."""
        self._load()
        import numpy as np
        import soundfile as sf

        start = max(0, int(start_sec * sr))
        end = min(len(audio), int(end_sec * sr))
        if end <= start:
            return ""
        seg = np.asarray(audio[start:end], dtype="float32")
        if seg.size < int(0.02 * sr):  # too short
            return ""
        if sr != 16000:
            seg, _ = _resample_to_16k(seg, sr)
            sr = 16000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, seg, sr, format="WAV")
            outs = self._model.transcribe([tmp_path])  # type: ignore[union-attr]
            if not outs:
                return ""
            first = outs[0]
            # NeMo returns either string or object with .text
            if hasattr(first, "text"):
                return (first.text or "").strip()
            if isinstance(first, str):
                return first.strip()
            return str(first).strip()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Agent-audio ASR + WER/CER stage for VoiceBench runs")
    parser.add_argument("--eval_results_dir", required=True, help="Path to eval-results/voicebench.<subtest>/")
    parser.add_argument("--subtest", required=True, help="Subtest name (for metrics key voicebench.<subtest>)")
    parser.add_argument("--input_jsonl", default="output.jsonl", help="Input jsonl filename inside eval_results_dir")
    parser.add_argument("--output_jsonl", default="output_asr.jsonl", help="Output jsonl filename inside eval_results_dir")
    parser.add_argument("--asr_model", default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo ASR model name")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs if they exist")
    args = parser.parse_args()

    eval_dir = Path(args.eval_results_dir)
    in_path = eval_dir / args.input_jsonl
    out_path = eval_dir / args.output_jsonl
    metrics_path = eval_dir / "agent_audio_metrics.json"

    if (out_path.exists() or metrics_path.exists()) and not args.force:
        print("Agent-audio ASR stage already done (output_asr/agent_audio_metrics exists). Skipping.")
        return

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input jsonl: {in_path}")

    # Aggregate counts
    total_word_sub = total_word_ins = total_word_del = total_ref_words = 0
    total_char_sub = total_char_ins = total_char_del = total_ref_chars = 0

    out_rows: List[Dict[str, Any]] = []

    rows_in = list(_read_jsonl(in_path))
    has_any_audio = False
    for row in rows_in:
        if isinstance(row.get("audio"), dict) and row["audio"].get("path"):
            has_any_audio = True
            break

    # Fast path: if no audio paths are present at all, emit output_asr.jsonl identical-in-text to output.jsonl
    # so downstream "ASR scoring" is still meaningful (it will match generated scoring).
    if not has_any_audio:
        for row in rows_in:
            generation_text = row.get("generation", "") or ""
            debug_info = row.get("debug_info")
            if not isinstance(debug_info, dict):
                debug_info = {}
            debug_info["agent_audio_missing"] = True
            debug_info["agent_audio_asr"] = None
            debug_info["agent_audio_wer"] = None
            debug_info["agent_audio_cer"] = None

            out_row = dict(row)
            out_row["generation_text"] = generation_text
            out_row["generation"] = generation_text
            out_row["debug_info"] = debug_info
            out_rows.append(out_row)

        _write_jsonl(out_path, out_rows)
        metrics = {
            f"voicebench.{args.subtest}": {
                "agent_wer": None,
                "agent_cer": None,
                "agent_ref_words": 0,
                "agent_ref_chars": 0,
                "agent_word_substitutions": 0,
                "agent_word_insertions": 0,
                "agent_word_deletions": 0,
                "agent_char_substitutions": 0,
                "agent_char_insertions": 0,
                "agent_char_deletions": 0,
            }
        }
        with metrics_path.open("wt", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"No audio paths found in {in_path}; wrote passthrough {out_path} and {metrics_path}")
        return

    vad = LazySileroVAD()
    asr = LazyNeMoASR(args.asr_model)

    for row in rows_in:
        generation_text = row.get("generation", "") or ""
        audio_path = None
        if isinstance(row.get("audio"), dict):
            audio_path = row["audio"].get("path")

        debug_info = row.get("debug_info")
        if not isinstance(debug_info, dict):
            debug_info = {}

        agent_asr_text = ""
        agent_wer = None
        agent_cer = None
        segments: List[Tuple[float, float]] = []

        if audio_path and Path(audio_path).exists():
            audio, sr = _load_audio_mono(audio_path)
            audio16, _ = _resample_to_16k(audio, sr)

            # VAD segmentation (fallback to full audio if VAD fails)
            segments = vad.speech_segments(audio16, 16000) if vad.available() else [(0.0, len(audio16) / 16000.0)]
            if not segments:
                segments = [(0.0, float(len(audio)) / float(sr))] if sr > 0 and len(audio) > 0 else []

            parts: List[str] = []
            for (s, e) in segments:
                t = asr.transcribe_segment(audio, sr, s, e)
                if t:
                    parts.append(t)
            agent_asr_text = " ".join(parts).strip()

            agent_wer, w_counts = compute_wer(generation_text, agent_asr_text)
            agent_cer, c_counts = compute_cer(generation_text, agent_asr_text)

            total_word_sub += w_counts.substitutions
            total_word_ins += w_counts.insertions
            total_word_del += w_counts.deletions
            total_ref_words += w_counts.ref_len

            total_char_sub += c_counts.substitutions
            total_char_ins += c_counts.insertions
            total_char_del += c_counts.deletions
            total_ref_chars += c_counts.ref_len
        else:
            debug_info["agent_audio_missing"] = True
            # Keep ASR output as generated text so ASR-scoring remains usable even when audio is missing.
            agent_asr_text = generation_text

        debug_info["agent_audio_asr"] = agent_asr_text
        debug_info["agent_audio_wer"] = agent_wer
        debug_info["agent_audio_cer"] = agent_cer
        if segments:
            debug_info["agent_audio_segments_sec"] = [{"start": s, "end": e} for (s, e) in segments]

        out_row = dict(row)
        out_row["generation_text"] = generation_text
        out_row["generation"] = agent_asr_text
        out_row["debug_info"] = debug_info
        out_rows.append(out_row)

    _write_jsonl(out_path, out_rows)

    agent_wer_total = None
    if total_ref_words > 0:
        agent_wer_total = (total_word_sub + total_word_ins + total_word_del) / total_ref_words
    agent_cer_total = None
    if total_ref_chars > 0:
        agent_cer_total = (total_char_sub + total_char_ins + total_char_del) / total_ref_chars

    metrics = {
        f"voicebench.{args.subtest}": {
            "agent_wer": agent_wer_total,
            "agent_cer": agent_cer_total,
            "agent_ref_words": total_ref_words,
            "agent_ref_chars": total_ref_chars,
            "agent_word_substitutions": total_word_sub,
            "agent_word_insertions": total_word_ins,
            "agent_word_deletions": total_word_del,
            "agent_char_substitutions": total_char_sub,
            "agent_char_insertions": total_char_ins,
            "agent_char_deletions": total_char_del,
        }
    }
    with metrics_path.open("wt", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path} and {metrics_path}")


if __name__ == "__main__":
    main()

