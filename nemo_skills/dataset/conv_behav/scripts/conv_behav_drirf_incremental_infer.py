#!/usr/bin/env python3
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

"""
Run conv_behav inference with DRIRF's s2s_incremental_v2 backend while writing
the validation_logs artifacts consumed by the original conv_behav scorer.

This intentionally delegates model construction, checkpoint loading, vLLM setup,
and frame-by-frame inference to the existing DRIRF backend. The only logic here
is SHAR iteration plus output adaptation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return str(value)


def _safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "_")


def _write_cut_audio(cut, output_path: Path) -> None:
    audio = np.asarray(cut.load_audio())
    if audio.ndim == 2 and audio.shape[0] <= 16:
        audio = audio.T
    elif audio.ndim > 2:
        audio = np.squeeze(audio)
        if audio.ndim == 2 and audio.shape[0] <= 16:
            audio = audio.T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, int(getattr(cut, "sampling_rate", 16000) or 16000))


def _supervision_text(cut, role: str | None = None) -> str:
    texts = []
    for supervision in sorted(cut.supervisions, key=lambda s: s.start):
        if role is not None and getattr(supervision, "speaker", None) != role:
            continue
        text = (getattr(supervision, "text", None) or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def _gt_turns(cut):
    turns = []
    for supervision in sorted(cut.supervisions, key=lambda s: s.start):
        turns.append(
            {
                "role": getattr(supervision, "speaker", "") or "",
                "start_time": float(supervision.start),
                "duration": float(supervision.duration),
                "text": (getattr(supervision, "text", None) or "").strip(),
            }
        )
    return turns


def _clean_timestamped_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|[\d.]+\|>", " ", text)
    text = re.sub(r"<\$[\d.]+\$>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("^", " ")
    return re.sub(r"\s+", " ", text).strip()


def _pred_turns(pred_text: str | None):
    cleaned = _clean_timestamped_text(pred_text)
    if not cleaned:
        return []
    return [{"role": "agent", "start_time": None, "duration": None, "text": cleaned}]


def _build_backend(args):
    from recipes.multimodal.server.backends.s2s_incremental_backend_v2 import (
        S2SIncrementalBackendV2,
        S2SIncrementalV2Config,
    )

    vllm_llm_config = None
    vllm_tts_config = None
    if "vllm" in args.engine_type:
        llm_path = args.llm_checkpoint_path or args.model_path
        vllm_llm_config = {
            "model_path": args.model_path,
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "dtype": "bfloat16",
            "engine_path": None,
            "pretrained_llm": llm_path,
        }
        vllm_tts_config = {
            "model_path": args.model_path,
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "dtype": "float32",
            "engine_path": None,
            "pretrained_llm": None,
            "skip_tokenizer_init": True,
        }

    cfg = S2SIncrementalV2Config(
        model_path=args.model_path,
        llm_checkpoint_path=args.llm_checkpoint_path or args.model_path,
        tts_checkpoint_path=args.tts_checkpoint_path,
        speaker_reference=args.speaker_reference,
        num_frames_per_inference=args.num_frames_per_inference,
        buffer_size_frames=args.buffer_size_frames,
        codec_token_history_size=args.codec_token_history_size,
        silence_padding_sec=args.silence_padding_sec,
        pad_to_duration_secs=args.pad_to_duration_secs,
        force_turn_taking=args.force_turn_taking,
        force_turn_taking_threshold=args.force_turn_taking_threshold,
        force_turn_taking_pad_window=args.force_turn_taking_pad_window,
        decode_audio=True,
        merge_user_channel=True,
        save_session_artifacts=not args.no_save_session_artifacts,
        session_artifacts_dir=str(Path(args.output_dir) / "validation_logs" / "session_artifacts"),
        output_frame_alignment=args.output_frame_alignment,
        engine_type=args.engine_type,
        use_perception_cache=args.use_perception_cache,
        use_perception_cudagraph=args.use_perception_cudagraph,
        use_codec_cache=args.use_codec_cache,
        disable_rnnt_decoder_cuda_graphs=args.disable_rnnt_decoder_cuda_graphs,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        temperature=args.temperature,
        inference_pad_boost=args.inference_pad_boost,
        inference_bos_boost=args.inference_bos_boost,
        inference_eos_boost=args.inference_eos_boost,
        inference_user_pad_boost=args.inference_user_pad_boost,
        inference_user_bos_boost=args.inference_user_bos_boost,
        inference_user_eos_boost=args.inference_user_eos_boost,
        system_prompt=args.system_prompt,
        tts_system_prompt=args.tts_system_prompt,
        vllm_llm_config=vllm_llm_config,
        vllm_tts_config=vllm_tts_config,
        matmul_precision=args.matmul_precision,
        max_new_tokens=args.max_new_tokens,
    )
    backend = S2SIncrementalBackendV2(cfg)
    backend.load_model()
    return backend


def run(args) -> None:
    from lhotse import CutSet
    from recipes.multimodal.server.backends.base import GenerationRequest

    if args.hf_home:
        os.environ.setdefault("HF_HOME", args.hf_home)
    if args.torch_home:
        os.environ.setdefault("TORCH_HOME", args.torch_home)

    output_dir = Path(args.output_dir)
    validation_dir = output_dir / "validation_logs"
    pred_audio_dir = validation_dir / "pred_wavs"
    metadata_dir = validation_dir / "metadatas"
    input_audio_dir = validation_dir / "input_wavs"
    pred_audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    input_audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / f"{args.dataset_name}.json"
    rank_metadata_path = metadata_dir / f"{args.dataset_name}_rank0.json"
    done_marker = metadata_path.with_suffix(metadata_path.suffix + ".done")
    done_marker.unlink(missing_ok=True)

    backend = _build_backend(args)
    try:
        cuts = CutSet.from_shar(in_dir=args.shar_input_dir, shuffle_shards=False)
    except TypeError:
        cuts = CutSet.from_shar(in_dir=args.shar_input_dir)

    processed = 0
    with open(metadata_path, "w", encoding="utf-8") as meta_fout, open(
        rank_metadata_path, "w", encoding="utf-8"
    ) as rank_fout:
        for idx, cut in enumerate(cuts):
            if args.max_samples is not None and idx >= args.max_samples:
                break

            sample_id = str(cut.id)
            safe_id = _safe_name(sample_id)
            input_audio_path = input_audio_dir / f"{safe_id}.wav"
            pred_audio_name = f"{args.dataset_name}_{safe_id}_rank0.wav"
            pred_audio_path = pred_audio_dir / pred_audio_name

            _write_cut_audio(cut, input_audio_path)
            result = backend.generate(
                [
                    GenerationRequest(
                        audio_path=str(input_audio_path),
                        request_id=f"conv_behav_{idx}_{safe_id}",
                        system_prompt=args.system_prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                ]
            )[0]
            if result.error:
                raise RuntimeError(f"DRIRF inference failed for {sample_id}: {result.error}")
            if not result.audio_bytes:
                raise RuntimeError(f"DRIRF inference returned no audio bytes for {sample_id}")

            pred_audio_path.write_bytes(result.audio_bytes)
            pred_text = result.text or ""
            pred_src_text = result.asr_text or ""
            entry = {
                "id": sample_id,
                "target_text": _supervision_text(cut, role="agent"),
                "pred_text": pred_text,
                "pred_audio": None,
                "src_text": _supervision_text(cut, role="user"),
                "pred_src_text": pred_src_text,
                "asr_pred_text": pred_src_text,
                "audio_path": f"pred_wavs/{pred_audio_name}",
                "generation_time_ms": result.generation_time_ms,
                "debug_info": _json_safe(result.debug_info or {}),
                "conversation_turns": {
                    "gt_conversation": _gt_turns(cut),
                    "pred_conversation": _pred_turns(pred_text),
                },
            }
            line = json.dumps(entry, ensure_ascii=False)
            meta_fout.write(line + "\n")
            rank_fout.write(line + "\n")
            meta_fout.flush()
            rank_fout.flush()
            processed += 1
            print(f"[conv_behav_drirf_incremental] {processed}: wrote {pred_audio_path}")

    print(f"[conv_behav_drirf_incremental] Wrote {processed} samples")
    print(f"[conv_behav_drirf_incremental] Metadata: {metadata_path}")
    if processed == 0:
        raise RuntimeError(f"No conv_behav samples were processed from {args.shar_input_dir}")
    done_marker.write_text("done\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run conv_behav incremental-format inference via DRIRF backend")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--llm_checkpoint_path")
    parser.add_argument("--tts_checkpoint_path")
    parser.add_argument("--speaker_reference")
    parser.add_argument("--shar_input_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True, help="Path to eval-results directory")
    parser.add_argument("--num_frames_per_inference", type=int, default=3)
    parser.add_argument("--buffer_size_frames", type=int, default=21)
    parser.add_argument("--codec_token_history_size", type=int, default=60)
    parser.add_argument("--engine_type", default="vllm_llm_vllm_eartts")
    parser.add_argument("--use_perception_cache", action="store_true")
    parser.add_argument("--use_perception_cudagraph", action="store_true")
    parser.add_argument("--use_codec_cache", dest="use_codec_cache", action="store_true", default=True)
    parser.add_argument("--no_use_codec_cache", dest="use_codec_cache", action="store_false")
    parser.add_argument("--disable_rnnt_decoder_cuda_graphs", action="store_true")
    parser.add_argument("--force_turn_taking", action="store_true")
    parser.add_argument("--force_turn_taking_threshold", type=int, default=40)
    parser.add_argument("--force_turn_taking_pad_window", type=int, default=25)
    parser.add_argument("--matmul_precision", default="medium")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--silence_padding_sec", type=float, default=0.0)
    parser.add_argument("--pad_to_duration_secs", type=float)
    parser.add_argument("--max_new_tokens", type=int, default=4500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--system_prompt")
    parser.add_argument("--tts_system_prompt")
    parser.add_argument("--inference_pad_boost", type=float)
    parser.add_argument("--inference_bos_boost", type=float)
    parser.add_argument("--inference_eos_boost", type=float)
    parser.add_argument("--inference_user_pad_boost", type=float)
    parser.add_argument("--inference_user_bos_boost", type=float)
    parser.add_argument("--inference_user_eos_boost", type=float)
    parser.add_argument("--output_frame_alignment", action="store_true")
    parser.add_argument("--no_save_session_artifacts", action="store_true")
    parser.add_argument("--hf_home")
    parser.add_argument("--torch_home")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
