# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""FD3 (FDB v3) scoring: reconstruct FD3-shaped per-sample result files from a
nemo-skills `output.jsonl`, then invoke the vendored FDBV3 evaluators
(evaluate_tool_calls.py / evaluate_pass_rate.py / analyze_tool_latency.py) and
merge the three reports into a single `metrics.json` keyed by `fdb_v3.tool_call`.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _alias_judge_api_key():
    """Cluster env sets NV_INFERENCE_KEY, but FD3 evaluators read NVIDIA_API_KEY /
    OPENAI_API_KEY. Mirror the value over so the LLM judge client can authenticate."""
    nv_key = os.environ.get("NV_INFERENCE_KEY")
    if not nv_key:
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = nv_key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = nv_key


_alias_judge_api_key()

# Match the cache locations the original FD3 pipeline sets, so HF/NeMo model
# downloads land on a writable path inside the scoring container.
for _var in ("HF_HOME", "TORCH_HOME", "NEMO_CACHE_DIR", "TRITON_CACHE_DIR"):
    os.environ.setdefault(_var, "/tmp/cache")

DEFAULT_FDB_REPO = Path("/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/FDBV3_CHENCHEN")
TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)
# Parakeet ASR used by the original FD3 pipeline (run_tool_benchmark.py).
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_tool_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("function")
    args = item.get("arguments", item.get("args", {}))
    fn_obj = item.get("function")
    if isinstance(fn_obj, dict):
        name = fn_obj.get("name") or name
        args = fn_obj.get("arguments", args)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw_arguments": args}
    if args is None:
        args = {}
    if not name:
        return None
    return {"function": str(name), "args": args}


def _parse_toolcall_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    seen: set[str] = set()
    calls: list[dict[str, Any]] = []
    payloads = [m.group(1).strip() for m in TOOLCALL_RE.finditer(text)]
    payloads.extend(part.split("</TOOLCALL>", 1)[0].strip() for part in text.split("<TOOLCALL>")[1:])
    for payload in payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            call = _normalize_tool_call(item)
            if call is None:
                continue
            key = json.dumps(call, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            calls.append(call)
    return calls


def _clean_s2s_text(text: str) -> str:
    """Strip the S2S text-channel control tokens (`<$0.72$>`, `<|...|>`, `<SPECIAL_N>`, `^`)
    so the LLM judge sees plain text. Mirrors `_clean_s2s_text` in
    FD3/release_code/run_s2s_offline_benchmark.py."""
    if not text:
        return ""
    cleaned = re.sub(r"<\$\d+(?:\.\d+)?\$>", " ", text)
    cleaned = re.sub(r"<\|\d+(?:\.\d+)?\|>", " ", cleaned)
    cleaned = re.sub(r"<SPECIAL_\d+>", " ", cleaned)
    cleaned = cleaned.replace("^", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _load_asr_model():
    """Load NeMo Parakeet for output-audio ASR. Returns None when NeMo is
    unavailable so the caller can fall back to silence-only detection."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except Exception as e:
        print(f"  warning: NeMo ASR unavailable ({e}); falling back to silence-only detection")
        return None
    try:
        print(f"  loading ASR model {ASR_MODEL_NAME}...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
        if torch.cuda.is_available() and hasattr(model, "cuda"):
            model = model.cuda()
            print("  ASR model on CUDA")
        else:
            print("  ASR model on CPU (slow — scoring_gpus=0)")
        return model
    except Exception as e:
        print(f"  warning: failed to load Parakeet ASR ({e}); continuing without ASR")
        return None


def _unwrap_transcribe_output(outputs):
    """NeMo's transcribe() return shape varies by model. Reduce it to a single
    result object. Mirrors run_tool_benchmark.py."""
    value = outputs
    while isinstance(value, tuple) and value:
        value = value[0]
    if isinstance(value, dict):
        for key in ("pred_text", "text", "hypotheses", "results"):
            if key in value and value[key]:
                value = value[key]
                break
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return value


def _parse_transcribe_result(result) -> dict:
    """Extract `{text, chunks}` from one NeMo transcription result. Chunks carry
    `timestamp: [start_s, end_s]` per word."""
    if result is None:
        return {"text": "", "chunks": []}
    text = ""
    chunks: list[dict[str, Any]] = []
    word_items: list = []
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = str(result.get("text") or result.get("pred_text") or result.get("transcript") or "")
        timestamp = result.get("timestamp") or {}
        if isinstance(timestamp, dict):
            word_items = timestamp.get("word") or []
    else:
        text = str(getattr(result, "text", "") or getattr(result, "pred_text", "") or "")
        timestamp = getattr(result, "timestamp", {}) or {}
        if isinstance(timestamp, dict):
            word_items = timestamp.get("word") or []
    if isinstance(word_items, list):
        for item in word_items:
            if not isinstance(item, dict):
                continue
            token_text = str(item.get("word") or item.get("text") or item.get("token") or "").strip()
            start, end = item.get("start"), item.get("end")
            if token_text and start is not None and end is not None:
                chunks.append({"text": token_text, "timestamp": [float(start), float(end)]})
    if not text and chunks:
        text = " ".join(c["text"] for c in chunks).strip()
    return {"text": text.strip(), "chunks": chunks}


def _run_asr(asr_model, audio_path: Path) -> dict:
    """Transcribe `audio_path` with NeMo Parakeet. Returns `{text, chunks}` or
    `{text: '', chunks: []}` on any failure."""
    if asr_model is None or not audio_path.exists():
        return {"text": "", "chunks": []}
    for label, kwargs in (("timestamps", {"timestamps": True}), ("plain", {"timestamps": False})):
        try:
            outputs = asr_model.transcribe([str(audio_path)], **kwargs)
            return _parse_transcribe_result(_unwrap_transcribe_output(outputs))
        except Exception as e:
            print(f"  ASR retry ({label}) on {audio_path.name} failed: {e}")
    return {"text": "", "chunks": []}


def _to_mono_wav(audio_path: Path, dst: Path) -> Path:
    """Convert audio to mono so older Parakeet checkpoints don't trip on stereo
    input. Falls back to the source path if pydub is unavailable."""
    try:
        from pydub import AudioSegment

        AudioSegment.from_file(str(audio_path)).set_channels(1).export(str(dst), format="wav")
        return dst
    except Exception:
        return audio_path


def _to_model_channel_mono_wav(audio_path: Path, dst: Path) -> Path:
    """Extract the model (agent) channel from a duplex output WAV.

    Convention in this codebase (see run_fdb_scoring._convert_stereo_to_mono):
    stereo output.wav has ch0=user, ch1=model. Averaging the two channels
    mixes the user's mic back into the agent ASR, so we pick ch1 explicitly.
    Mono inputs are passed through unchanged."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(audio_path))
        if data.ndim == 2 and data.shape[1] >= 2:
            sf.write(str(dst), data[:, 1], sr)
            return dst
        return audio_path
    except Exception:
        return _to_mono_wav(audio_path, dst)


def _user_speech_end_from_chunks(chunks: list[dict], bounds: dict) -> float | None:
    """Locate when the user finished speaking. Prefer ASR chunks with a 2s-gap
    heuristic (matches run_s2s_offline_benchmark.py); fall back to silence-based
    bounds otherwise."""
    if chunks:
        for idx in range(len(chunks) - 1):
            curr_end = chunks[idx]["timestamp"][1]
            next_start = chunks[idx + 1]["timestamp"][0]
            if next_start - curr_end > 2.0:
                return float(curr_end)
        return float(chunks[-1]["timestamp"][1])
    if bounds.get("last_speech_s") is not None:
        return float(bounds["last_speech_s"])
    if bounds.get("duration_s") is not None:
        return float(bounds["duration_s"])
    return None


def _measure_latency(input_wav: Path, output_wav: Path, silence_threshold_db: int = -40) -> dict:
    try:
        from pydub import AudioSegment

        in_audio = AudioSegment.from_file(str(input_wav))
        out_audio = AudioSegment.from_file(str(output_wav))
        first_speech_ms = None
        chunk_ms = 50
        for i in range(0, len(out_audio), chunk_ms):
            if out_audio[i : i + chunk_ms].dBFS > silence_threshold_db:
                first_speech_ms = i
                break
        first_speech_s = first_speech_ms / 1000.0 if first_speech_ms is not None else len(out_audio) / 1000.0
        return {
            "input_duration_s": round(len(in_audio) / 1000.0, 3),
            "output_duration_s": round(len(out_audio) / 1000.0, 3),
            "first_speech_s": round(first_speech_s, 3),
        }
    except Exception as e:
        return {"error": str(e)}


def _detect_speech_bounds(audio_path: Path, silence_threshold_db: int = -40) -> dict:
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(str(audio_path))
        regions = detect_nonsilent(audio, min_silence_len=120, silence_thresh=silence_threshold_db, seek_step=10)
        duration_s = round(len(audio) / 1000.0, 3)
        if not regions:
            return {"duration_s": duration_s, "first_speech_s": None, "last_speech_s": None}
        return {
            "duration_s": duration_s,
            "first_speech_s": round(regions[0][0] / 1000.0, 3),
            "last_speech_s": round(regions[-1][1] / 1000.0, 3),
        }
    except Exception as e:
        return {"duration_s": None, "first_speech_s": None, "last_speech_s": None, "error": str(e)}


def _extract_text_from_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Return (function_channel_text, actual_tool_calls, s2s_text_channel) from an
    output.jsonl row. `s2s_text_channel` is the model's raw text-channel output
    (often just `<$0.72$>` for tool-call-only responses) — it is NOT an ASR
    transcript and should only be used as a fallback after _clean_s2s_text."""
    generation = str(row.get("generation") or "")
    s2s_text = ""
    serialized = row.get("serialized_output") or []
    if isinstance(serialized, list):
        for msg in serialized:
            if not isinstance(msg, dict):
                continue
            audio = msg.get("audio") or {}
            if isinstance(audio, dict) and audio.get("transcript"):
                s2s_text = audio["transcript"]
                break
            if msg.get("content"):
                s2s_text = s2s_text or msg["content"]
    if not s2s_text:
        audio = row.get("audio") or {}
        if isinstance(audio, dict):
            s2s_text = audio.get("transcript", "")
    if not s2s_text:
        s2s_text = generation

    function_text = generation
    actual_calls = _parse_toolcall_text(generation)
    if not actual_calls and isinstance(serialized, list):
        for msg in serialized:
            if not isinstance(msg, dict):
                continue
            tc = msg.get("tool_calls") or []
            for entry in tc or []:
                fn = entry.get("function") if isinstance(entry, dict) else None
                if isinstance(fn, dict):
                    actual_calls.append(_normalize_tool_call(fn))
        actual_calls = [c for c in actual_calls if c]
    return function_text, actual_calls, s2s_text


def _reconstruct_fd3_layout(
    output_jsonl: Path,
    fd3_data_root: Path,
    layout_root: Path,
    provider: str,
    skip_latency: bool,
    skip_asr: bool = False,
) -> int:
    """Materialize layout_root/{example_id}_{speaker_id}/result_<provider>.json
    plus an output_<provider>.wav copy when the generated audio path is reachable.
    Returns the number of result files written.

    When ASR is enabled (default), this transcribes the generated output audio
    with Parakeet and uses the ASR result for `transcript` + first-speech timing.
    The S2S model's text-channel output is preserved separately in
    `s2s_pred_text` (cleaned of control tokens) and only used as a transcript
    fallback when ASR returns no text.
    """
    layout_root.mkdir(parents=True, exist_ok=True)
    asr_model = None if skip_asr else _load_asr_model()
    written = 0
    with output_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            example_id = row.get("example_id")
            speaker_id = row.get("speaker_id")
            if not (sample_id and example_id and speaker_id):
                continue

            sample_dir = layout_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            fd3_input_dir = fd3_data_root / sample_id
            input_wav = fd3_input_dir / "input.wav"

            function_text, actual_calls, raw_s2s_text = _extract_text_from_row(row)
            s2s_pred_text = _clean_s2s_text(raw_s2s_text)

            audio_meta = row.get("audio") or {}
            output_wav_src = Path(audio_meta["path"]) if isinstance(audio_meta, dict) and audio_meta.get("path") else None
            output_wav_dst = sample_dir / f"output_{provider}.wav"
            if output_wav_src and output_wav_src.exists():
                if not output_wav_dst.exists() or output_wav_dst.stat().st_size != output_wav_src.stat().st_size:
                    shutil.copy2(output_wav_src, output_wav_dst)

            # Pre-extract the mono input + model-channel-only output WAVs ONCE.
            # Both ASR and silence-based latency need them; in the stereo
            # output, ch0 is the user (loud, starting at t≈0) and ch1 is the
            # model. Using the raw stereo WAV for silence-detection causes
            # `first_speech_s = 0` for every sample (because ch0 fires
            # immediately), which collapses every sample into "interruption"
            # and breaks avg latency.
            asr_output_path = output_wav_dst
            asr_input_path = input_wav
            out_mono_path = sample_dir / "_output_mono.wav"
            mono_path = sample_dir / "_input_mono.wav"
            try:
                if output_wav_dst.exists():
                    asr_output_path = _to_model_channel_mono_wav(output_wav_dst, out_mono_path)
                if input_wav.exists():
                    asr_input_path = _to_mono_wav(input_wav, mono_path)

                output_asr = {"text": "", "chunks": []}
                input_asr = {"text": "", "chunks": []}
                if asr_model is not None:
                    if output_wav_dst.exists():
                        output_asr = _run_asr(asr_model, asr_output_path)
                    if input_wav.exists():
                        input_asr = _run_asr(asr_model, asr_input_path)

                if output_asr["text"]:
                    transcript = output_asr["text"]
                    transcript_source = "asr"
                else:
                    transcript = s2s_pred_text
                    transcript_source = "s2s_pred_text_fallback"

                latency = {}
                user_speech_end_rel = None
                agent_speech_start_rel = None
                perceived_total_latency = None
                if not skip_latency and input_wav.exists() and output_wav_dst.exists():
                    # Use mono channels here too — otherwise silence detection
                    # on the stereo output hits ch0 (user) at t=0.
                    latency = _measure_latency(asr_input_path, asr_output_path)
                    bounds_in = _detect_speech_bounds(asr_input_path)
                    bounds_out = _detect_speech_bounds(asr_output_path)
                    user_speech_end_rel = _user_speech_end_from_chunks(input_asr["chunks"], bounds_in)
                    if output_asr["chunks"]:
                        # Skip greeting/prelude audio emitted before the user finished
                        # speaking — otherwise latency goes negative when the model
                        # opens with e.g. "Hi! How can I help you today?".
                        first_chunk_ts = None
                        if user_speech_end_rel is not None:
                            for ch in output_asr["chunks"]:
                                ts = float(ch["timestamp"][0])
                                if ts >= user_speech_end_rel:
                                    first_chunk_ts = ts
                                    break
                        if first_chunk_ts is None:
                            first_chunk_ts = float(output_asr["chunks"][0]["timestamp"][0])
                        agent_speech_start_rel = first_chunk_ts
                    elif bounds_out.get("first_speech_s") is not None:
                        agent_speech_start_rel = float(bounds_out["first_speech_s"])
                    if user_speech_end_rel is not None and agent_speech_start_rel is not None:
                        perceived_total_latency = round(agent_speech_start_rel - user_speech_end_rel, 3)
            finally:
                for tmp in (out_mono_path, mono_path):
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass

            result = {
                "pid": speaker_id,
                "example_id": example_id,
                "category": row.get("domain", "unknown"),
                "title": row.get("title", example_id),
                "provider": provider,
                "evaluated_at": datetime.datetime.now().isoformat(),
                "tool_call_source": "s2s_function_head",
                "backend_used": False,
                "function_channel_text": function_text,
                "function_channel_text_raw": function_text,
                "actual_tool_calls": actual_calls,
                "function_call_count": len(actual_calls),
                "s2s_pred_text": s2s_pred_text,
                "transcript": transcript,
                "asr_chunks": output_asr["chunks"],
                "input_transcript": input_asr["text"],
                "input_asr_chunks": input_asr["chunks"],
                "output_transcript_source": transcript_source,
                "agent_audio_detected": agent_speech_start_rel is not None,
                "user_speech_end_rel": round(user_speech_end_rel, 3) if user_speech_end_rel is not None else None,
                "audio_agent_speech_start": round(agent_speech_start_rel, 3) if agent_speech_start_rel is not None else None,
                "perceived_total_latency": perceived_total_latency,
                "absolute_total_latency": perceived_total_latency,
                "latency": latency,
                "status": "completed",
            }
            (sample_dir / f"result_{provider}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            written += 1
    return written


def _run_evaluator(release_code: Path, script_name: str, args_list: list[str], log_prefix: str) -> dict | None:
    """Run a vendored FD3 evaluator and return its parsed JSON report.

    The upstream `evaluate_tool_calls.py` writes its JSON report before
    printing a summary that can crash on edge cases (e.g. N=0 turn-taken
    samples). We always try to recover the report from disk regardless of
    the script's exit code.
    """
    script = release_code / script_name
    if not script.exists():
        print(f"  warning: {script_name} not found at {script}; skipping {log_prefix}")
        return None
    cmd = [sys.executable, str(script), *args_list]
    print(f"  running {log_prefix}: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(release_code), capture_output=True, text=True)
    print(res.stdout)

    output_idx = args_list.index("--output") if "--output" in args_list else -1
    report_path = Path(args_list[output_idx + 1]) if output_idx >= 0 and output_idx + 1 < len(args_list) else None
    report_written = report_path.exists() if report_path else False

    if res.returncode != 0 and not report_written:
        print(res.stderr, file=sys.stderr)
        print(f"  warning: {log_prefix} exited with code {res.returncode} and produced no report")
    elif res.returncode != 0:
        print(f"  note: {log_prefix} exited with code {res.returncode} but the report was written; continuing")

    if report_written:
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  warning: could not parse {report_path}: {e}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FD3 (FDB v3) scoring entry point")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument(
        "--fdb_repo",
        type=Path,
        default=DEFAULT_FDB_REPO,
        help="Vendored FDBV3_CHENCHEN root (containing FD3/release_code/ and FD3/fdb_v3_data_released/)",
    )
    parser.add_argument("--provider", type=str, default="fdb_v3")
    parser.add_argument("--use_llm_judge", action="store_true", help="Pass --use-llm to the FD3 evaluators")
    parser.add_argument("--skip_latency", action="store_true", help="Skip audio-derived latency computation")
    parser.add_argument(
        "--skip_asr",
        action="store_true",
        help=(
            "Skip Parakeet ASR on the output audio. Without ASR, response_qual is "
            "judged against the S2S text channel (often just `<$0.72$>` for tool-call-only "
            "replies) which can collapse the metric — only use this for debugging."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json already populated")
    parser.add_argument(
        "--release_code_dir",
        type=Path,
        default=None,
        help=(
            "Override the directory holding the FD3 evaluator scripts + "
            "benchmark_data_v2.json. Defaults to <fdb_repo>/FD3/release_code "
            "(the CHENCHEN copy). The fdb_v3_official variant points this at the "
            "vendored upstream scripts under scripts/fdb_v3_official/release_code/."
        ),
    )
    parser.add_argument(
        "--benchmark_key",
        type=str,
        default="fdb_v3.tool_call",
        help="Top-level key under which metrics are written in metrics.json.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["asr", "judge", "both"],
        default="both",
        help=(
            "Which scoring stage(s) to run. 'asr' only reconstructs FD3 layout "
            "+ runs Parakeet ASR on output WAVs (needs GPU; fast, ~2 min). "
            "'judge' only runs the FD3 evaluators (LLM judge + latency analysis; "
            "no GPU, can run for tens of minutes via gpt-5.2 API). Default 'both' "
            "preserves the original single-job behavior."
        ),
    )
    args = parser.parse_args()

    eval_results_dir = args.eval_results_dir.resolve()
    metrics_file = eval_results_dir / "metrics.json"
    benchmark_key = args.benchmark_key

    if metrics_file.exists() and not args.force:
        try:
            existing = json.loads(metrics_file.read_text(encoding="utf-8"))
            if existing.get(benchmark_key):
                print(f"Scoring already done for {benchmark_key}. Skipping (use --force to re-run).")
                return
        except Exception:
            pass

    output_jsonl = eval_results_dir / "output.jsonl"
    if not output_jsonl.exists():
        sys.exit(f"output.jsonl not found at {output_jsonl}")

    release_code = args.release_code_dir.resolve() if args.release_code_dir else (args.fdb_repo / "FD3" / "release_code")
    fd3_data_root = args.fdb_repo / "FD3" / "fdb_v3_data_released"
    benchmark_json = release_code / "benchmark_data_v2.json"
    for path, label in [(release_code, "release_code"), (fd3_data_root, "fdb_v3_data_released"), (benchmark_json, "benchmark_data_v2.json")]:
        if not path.exists():
            sys.exit(f"Required FD3 path missing ({label}): {path}")

    layout_root = eval_results_dir / "fdb_v3_layout"

    # Stage 1 (ASR): reconstruct FD3 per-sample layout + run Parakeet ASR on
    # output WAVs. Needs GPU. Idempotent — _reconstruct_fd3_layout writes
    # per-sample result_{provider}.json files that the judge stage reads.
    if args.stage in ("asr", "both"):
        print(f"[stage=asr] Reconstructing FD3 layout under {layout_root}")
        n_results = _reconstruct_fd3_layout(
            output_jsonl, fd3_data_root, layout_root, args.provider, args.skip_latency, args.skip_asr
        )
        print(f"[stage=asr] Wrote {n_results} result_{args.provider}.json files")

    if args.stage == "asr":
        # ASR-only mode: write a small marker so the downstream judge job knows
        # this layout is ready. Skip evaluators entirely.
        marker = eval_results_dir / ".asr_done"
        marker.write_text("ok\n")
        print(f"[stage=asr] ASR stage complete; wrote marker {marker}")
        return

    # Stage 2 (Judge): FD3 evaluators (LLM judge for tool_calls/pass_rate +
    # latency analysis). No GPU; calls gpt-5.2 over HTTP API. Reads from the
    # layout written by stage 1.
    if not layout_root.exists():
        sys.exit(
            f"[stage=judge] Expected per-sample layout at {layout_root}, but it "
            f"is missing. Run with --stage=asr first, or use --stage=both."
        )

    report_dir = eval_results_dir / "summarized-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_out = report_dir / f"{args.provider}_eval.json"
    pass_out = report_dir / f"{args.provider}_pass_rate.json"
    latency_out = report_dir / f"{args.provider}_latency.json"

    llm_flag = ["--use-llm"] if args.use_llm_judge else []

    eval_report = _run_evaluator(
        release_code,
        "evaluate_tool_calls.py",
        [
            "--benchmark", str(benchmark_json),
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(eval_out),
            *llm_flag,
        ],
        "evaluate_tool_calls",
    )
    pass_report = _run_evaluator(
        release_code,
        "evaluate_pass_rate.py",
        [
            "--benchmark", str(benchmark_json),
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(pass_out),
            *llm_flag,
        ],
        "evaluate_pass_rate",
    )
    latency_report = _run_evaluator(
        release_code,
        "analyze_tool_latency.py",
        [
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(latency_out),
        ],
        "analyze_tool_latency",
    )

    if eval_report is None and pass_report is None and latency_report is None:
        raise RuntimeError(
            f"FDB v3 scoring produced no reports — all of evaluate_tool_calls, evaluate_pass_rate, "
            f"and analyze_tool_latency failed. Refusing to write a metrics.json containing only "
            f"metadata. Inspect the evaluator stderr above and the layout under {layout_root}."
        )

    metrics: dict[str, Any] = {benchmark_key: {}}
    if eval_report is not None:
        metrics[benchmark_key]["eval"] = eval_report
    if pass_report is not None:
        metrics[benchmark_key]["pass_rate"] = pass_report
    if latency_report is not None:
        metrics[benchmark_key]["latency"] = latency_report

    if eval_report and "by_metric" in eval_report:
        bm = eval_report["by_metric"]
        metrics[benchmark_key]["headline"] = {
            "tool_selection_acc": bm.get("tool_selection_acc"),
            "argument_acc": bm.get("argument_acc"),
            "response_qual": bm.get("response_qual"),
            "turn_take_rate": eval_report.get("turn_taking", {}).get("turn_take_rate"),
            "total_scenarios": eval_report.get("total_scenarios"),
        }
    # Count result files from the on-disk layout instead of the ASR-stage
    # return value: this code path also runs in --stage=judge (after a
    # separate ASR job populated layout_root), where n_results is unbound.
    metrics[benchmark_key]["num_result_files"] = sum(
        1 for _ in layout_root.glob(f"*/result_{args.provider}.json")
    )
    metrics[benchmark_key]["provider"] = args.provider
    metrics[benchmark_key]["fdb_repo"] = str(args.fdb_repo)

    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_file}")


if __name__ == "__main__":
    main()
