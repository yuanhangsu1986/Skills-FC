#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
CLI wrapper for the Unified NeMo Inference Server.

This module provides a command-line interface compatible with nemo-skills
server deployment patterns. It translates standard vllm-style CLI arguments
to the unified server configuration.

Usage via NeMo-Skills:

    # SALM backend (speech-augmented language model)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend salm"

    # TTS backend (text-to-speech)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/tts_model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend tts --codec_model /path/to/codec"

    # S2S backend (speech-to-speech)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/s2s_model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend s2s"

Environment Variables:
    UNIFIED_SERVER_HOST: Server host (default: 0.0.0.0)
    UNIFIED_SERVER_PORT: Server port (default: 8000)
    UNIFIED_SERVER_BACKEND: Backend type (default: salm)
    UNIFIED_SERVER_MODEL_PATH: Path to model
    UNIFIED_SERVER_CODEC_MODEL_PATH: Path to codec model
    UNIFIED_SERVER_BATCH_SIZE: Batch size (default: 8)
    UNIFIED_SERVER_BATCH_TIMEOUT: Batch timeout (default: 0.1)
    DEBUG: Enable debug mode
"""

import argparse
import inspect
import os
import shutil
import sys
from typing import Optional


def setup_pythonpath(code_path: Optional[str] = None):
    """Set up PYTHONPATH for NeMo and the unified server.

    Args:
        code_path: Single path or colon-separated paths to add to PYTHONPATH
    """
    paths_to_add = []

    # Add explicit code path(s) if provided (supports colon-separated paths)
    if code_path:
        for path in code_path.split(":"):
            if path and path not in paths_to_add:
                paths_to_add.append(path)

    # Add recipes path for unified server imports
    # Look for the recipes directory relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find ns_eval root (go up from nemo_skills/inference/server/)
    ns_eval_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    if os.path.exists(os.path.join(ns_eval_root, "recipes")):
        paths_to_add.append(ns_eval_root)

    # Also check /nemo_run/code pattern used in containers
    if os.path.exists("/nemo_run/code"):
        paths_to_add.append("/nemo_run/code")

    # Update PYTHONPATH
    current_path = os.environ.get("PYTHONPATH", "")
    for path in paths_to_add:
        if path not in current_path.split(":"):
            current_path = f"{path}:{current_path}" if current_path else path

    os.environ["PYTHONPATH"] = current_path

    # Also add to sys.path for immediate imports
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def apply_safetensors_patch(hack_path: Optional[str]):
    """Apply safetensors patch if provided (for some NeMo models)."""
    if not hack_path or not os.path.exists(hack_path):
        return

    try:
        import safetensors.torch as st_torch

        dest_path = inspect.getfile(st_torch)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(hack_path, dest_path)
        print(f"[serve_unified] Applied safetensors patch: {hack_path} -> {dest_path}")
    except Exception as e:
        print(f"[serve_unified] Warning: Failed to apply safetensors patch: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified NeMo Inference Server CLI wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Standard vllm-style arguments (for nemo-skills compatibility)
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Backend selection
    parser.add_argument(
        "--backend",
        default="salm",
        choices=["salm", "tts", "s2s", "s2s_voicechat", "s2s_incremental", "s2s_incremental_v2", "s2s_session"],
        help="Backend type: salm (speech-augmented LM), tts (text-to-speech), s2s (speech-to-speech offline), s2s_voicechat (NemotronVoiceChat offline, YAML-driven), s2s_incremental (frame-by-frame processing), s2s_incremental_v2 (V2 frame-by-frame with NeMo wrapper, vLLM, caches), s2s_session (session-aware multi-turn)",
    )

    # Backend-specific model paths
    parser.add_argument("--codec_model", default=None, help="Path to codec model (required for TTS, optional for S2S)")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument(
        "--batch_timeout", type=float, default=0.1, help="Batch timeout in seconds (0 for no batching delay)"
    )

    # Generation defaults
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")

    # Model configuration
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")

    # Backend-specific options
    parser.add_argument("--prompt_format", default=None, help="Prompt format (SALM backend)")
    parser.add_argument(
        "--phoneme_input_type", default="predicted", help="Phoneme input type: predicted or gt (TTS backend)"
    )
    parser.add_argument(
        "--decoder_only_model", action="store_true", help="Use decoder-only model architecture (TTS backend)"
    )
    parser.add_argument("--use_local_transformer", action="store_true", help="Use local transformer (TTS backend)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (TTS backend)")

    # Environment setup
    parser.add_argument("--code_path", default=None, help="Path to NeMo source code to add to PYTHONPATH")
    parser.add_argument("--hack_path", default=None, help="Path to safetensors/torch.py patch file")

    # S2S backend options
    parser.add_argument(
        "--ignore_system_prompt",
        action="store_true",
        help="Ignore system prompts from requests (for models that don't support them)",
    )
    parser.add_argument(
        "--silence_padding_sec",
        type=float,
        default=5.0,
        help="Seconds of silence to append after audio (legacy, prefer --extra_decoding_seconds)",
    )
    parser.add_argument(
        "--extra_decoding_seconds",
        type=float,
        default=0.0,
        help="Extra decoding time in seconds (0 for FDB, 20 for Voicebench)",
    )
    parser.add_argument(
        "--tts_ckpt_path",
        default=None,
        help="Path to TTS checkpoint (s2s offline backend)",
    )
    parser.add_argument(
        "--inference_pad_boost",
        type=float,
        default=None,
        help="Boost for PAD token logits during inference",
    )
    parser.add_argument(
        "--inference_bos_boost",
        type=float,
        default=None,
        help="Boost for BOS token logits during inference",
    )
    parser.add_argument(
        "--inference_eos_boost",
        type=float,
        default=None,
        help="Boost for EOS token logits during inference",
    )
    parser.add_argument(
        "--inference_user_pad_boost",
        type=float,
        default=None,
        help="Boost for ASR PAD token logits during inference (requires nemotron_h.py patch for vLLM)",
    )
    parser.add_argument(
        "--inference_user_bos_boost",
        type=float,
        default=None,
        help="Boost for ASR BOS token logits during inference (requires nemotron_h.py patch for vLLM)",
    )
    parser.add_argument(
        "--inference_user_eos_boost",
        type=float,
        default=None,
        help="Boost for ASR EOS token logits during inference (requires nemotron_h.py patch for vLLM)",
    )

    # s2s_voicechat (nemotron_voicechat_infer-like) options
    parser.add_argument(
        "--decode_audio",
        action="store_true",
        help="Enable audio decoding/output (s2s_voicechat backend; default: text-only)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output directory for artifacts (s2s_voicechat backend)",
    )
    parser.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Save per-request artifacts under output_dir (s2s_voicechat backend)",
    )
    parser.add_argument(
        "--trim_leading_silence",
        action="store_true",
        help="Trim leading silence from response audio to reduce FDB-reported latency (s2s_voicechat)",
    )
    parser.add_argument(
        "--trim_leading_silence_padding_sec",
        type=float,
        default=0.01,
        help="Seconds to keep before first speech when trimming (default: 0.01)",
    )
    parser.add_argument(
        "--merge_user_channel",
        action="store_true",
        help="Merge user (input) + model (pred) into two-channel WAV like NeMo ResultsLogger (for FDB)",
    )

    # S2S Incremental backend options
    parser.add_argument(
        "--config_path",
        default=None,
        help="Path to YAML config file (s2s_incremental backend)",
    )
    parser.add_argument(
        "--llm_checkpoint_path",
        default=None,
        help="Path to LLM checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--tts_checkpoint_path",
        default=None,
        help="Path to TTS checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--speaker_reference",
        default=None,
        help="Path to speaker reference audio for TTS (s2s_incremental backend)",
    )
    parser.add_argument(
        "--num_frames_per_inference",
        type=int,
        default=1,
        help="Frames per inference step (s2s_incremental backend)",
    )
    parser.add_argument(
        "--no_decode_audio",
        action="store_true",
        help="Disable audio output (s2s_incremental backend)",
    )
    parser.add_argument(
        "--response_end_detection_mode",
        type=str,
        default="audio_energy",
        choices=["audio_energy", "eos"],
        help="Response end detection mode: audio_energy (TTS silence) or eos (consecutive PAD tokens)",
    )
    parser.add_argument(
        "--eos_detection_window",
        type=int,
        default=10,
        help="Number of consecutive PAD tokens to detect end of response (used when mode=eos)",
    )

    # S2S Incremental V2 backend options
    parser.add_argument(
        "--engine_type",
        type=str,
        default="native",
        choices=["native", "vllm_llm", "vllm_eartts", "vllm_llm_vllm_eartts"],
        help="Inference engine type (s2s_incremental_v2 backend)",
    )
    parser.add_argument(
        "--use_perception_cache",
        action="store_true",
        help="Enable cache-aware streaming for perception encoder (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--use_perception_cudagraph",
        action="store_true",
        help="Enable CUDA graph-accelerated perception encoder (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--use_codec_cache",
        action="store_true",
        help="Incremental codec decode to remove clicking (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--buffer_size_frames",
        type=int,
        default=None,
        help="Number of frames in audio buffer (s2s_incremental_v2, default: 20 w/ perception cache, 71 without)",
    )
    parser.add_argument(
        "--codec_token_history_size",
        type=int,
        default=60,
        help="Sliding-window buffer size; ignored when use_codec_cache is on (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--pad_audio_to_sec",
        "--pad_to_duration_secs",
        type=float,
        default=None,
        dest="pad_to_duration_secs",
        help="Pad input audio to this duration in seconds (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt for the model (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--tts_system_prompt",
        type=str,
        default=None,
        help="TTS system prompt to condition generation style (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--force_turn_taking",
        action="store_true",
        help="Enable forced turn-taking (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--force_turn_taking_threshold",
        type=int,
        default=40,
        help="Threshold for forced turn-taking (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--force_turn_taking_pad_window",
        type=int,
        default=25,
        help="Pad window for forced turn-taking (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        help="torch float32 matmul precision (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization for vLLM engines (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=8192,
        help="Max model sequence length for vLLM engines (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--merge_user_channel_v2",
        action="store_true",
        help="Return dual-channel (user+agent) WAV in response (s2s_incremental_v2, for FDB)",
    )
    parser.add_argument(
        "--use_asr_as_response",
        action="store_true",
        help="Use ASR channel (user transcription) as primary response text instead of agent text (for ASR evaluation)",
    )
    parser.add_argument(
        "--no_inference_guidance_enabled",
        action="store_true",
        help="Disable TTS classifier-free guidance (s2s_incremental_v2, default: enabled)",
    )
    parser.add_argument(
        "--inference_guidance_scale",
        type=float,
        default=None,
        help="TTS guidance scale; overrides checkpoint default when set (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--inference_top_p_or_k",
        type=float,
        default=None,
        help="TTS top-p/top-k; overrides checkpoint default when set (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--inference_noise_scale",
        type=float,
        default=None,
        help="TTS noise scale; overrides checkpoint default when set (s2s_incremental_v2)",
    )
    parser.add_argument(
        "--tts_sliding_window",
        type=int,
        default=None,
        help="Override sliding_window in TTS vLLM config.json before engine creation (s2s_incremental_v2)",
    )

    # Session management options (s2s_session backend)
    parser.add_argument(
        "--session_ttl",
        type=float,
        default=300.0,
        help="Session time-to-live in seconds (s2s_session backend)",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=100,
        help="Maximum number of concurrent sessions (s2s_session backend)",
    )
    parser.add_argument(
        "--session_artifacts_dir",
        type=str,
        default=None,
        help="Directory to save session artifacts (input/output audio, JSON). Default: /tmp/s2s_sessions",
    )
    parser.add_argument(
        "--no_save_session_artifacts",
        action="store_true",
        help="Disable saving session artifacts to disk",
    )
    parser.add_argument(
        "--output_frame_alignment",
        action="store_true",
        help="Include per-frame alignment data in debug output (user/agent/ASR per frame)",
    )

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Pre-server pip install (runs inside the server process before model loading)
    parser.add_argument(
        "--pip_install",
        type=str,
        default=None,
        help="Space-separated pip packages to install before starting the server "
        "(e.g. 'lhotse==1.32.2 transformers==4.56.0')",
    )

    # Parse known args, allowing extra args to be passed through
    args, extra_args = parser.parse_known_args()

    # Run pre-server pip install if requested
    if args.pip_install:
        import subprocess

        pip_cmd = f"pip install {args.pip_install}"
        print(f"[serve_unified] Running: {pip_cmd}")
        result = subprocess.run(pip_cmd, shell=True)
        if result.returncode != 0:
            print("[serve_unified] pip install failed, exiting.")
            sys.exit(1)
        print("[serve_unified] pip install completed.")

    # Setup environment
    setup_pythonpath(args.code_path)
    apply_safetensors_patch(args.hack_path)

    # Set environment variables
    os.environ["UNIFIED_SERVER_HOST"] = args.host
    os.environ["UNIFIED_SERVER_PORT"] = str(args.port)
    os.environ["UNIFIED_SERVER_BACKEND"] = args.backend
    os.environ["UNIFIED_SERVER_MODEL_PATH"] = args.model
    os.environ["UNIFIED_SERVER_BATCH_SIZE"] = str(args.batch_size)
    os.environ["UNIFIED_SERVER_BATCH_TIMEOUT"] = str(args.batch_timeout)
    os.environ["UNIFIED_SERVER_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["UNIFIED_SERVER_TEMPERATURE"] = str(args.temperature)
    os.environ["UNIFIED_SERVER_TOP_P"] = str(args.top_p)

    if args.codec_model:
        os.environ["UNIFIED_SERVER_CODEC_MODEL_PATH"] = args.codec_model

    if args.debug:
        os.environ["DEBUG"] = "1"

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    # Build extra config for backend-specific options
    extra_config = {}

    if args.prompt_format:
        extra_config["prompt_format"] = args.prompt_format

    if args.backend == "tts":
        extra_config["decoder_only_model"] = args.decoder_only_model
        extra_config["phoneme_input_type"] = args.phoneme_input_type
        extra_config["use_local_transformer"] = args.use_local_transformer
        if args.top_k:
            extra_config["top_k"] = args.top_k

    # S2S backend options
    if args.backend in ("s2s", "s2s_voicechat", "s2s_incremental", "s2s_incremental_v2", "s2s_session"):
        extra_config["ignore_system_prompt"] = args.ignore_system_prompt
        if args.silence_padding_sec != 5.0:
            extra_config["silence_padding_sec"] = args.silence_padding_sec

    # S2S offline backend specific options
    if args.backend == "s2s":
        if args.extra_decoding_seconds:
            extra_config["extra_decoding_seconds"] = args.extra_decoding_seconds
        if args.config_path:
            extra_config["config_path"] = args.config_path
        if args.tts_ckpt_path:
            extra_config["tts_ckpt_path"] = args.tts_ckpt_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference
        if args.code_path:
            extra_config["code_path"] = args.code_path
        if args.inference_pad_boost is not None:
            extra_config["inference_pad_boost"] = args.inference_pad_boost
        if args.inference_bos_boost is not None:
            extra_config["inference_bos_boost"] = args.inference_bos_boost
        if args.inference_eos_boost is not None:
            extra_config["inference_eos_boost"] = args.inference_eos_boost

    # s2s_voicechat backend specific options
    if args.backend == "s2s_voicechat":
        if args.extra_decoding_seconds:
            extra_config["extra_decoding_seconds"] = args.extra_decoding_seconds
        if args.config_path:
            extra_config["config_path"] = args.config_path
        if args.tts_ckpt_path:
            extra_config["tts_ckpt_path"] = args.tts_ckpt_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference
        if args.code_path:
            extra_config["code_path"] = args.code_path
        if args.inference_pad_boost is not None:
            extra_config["inference_pad_boost"] = args.inference_pad_boost
        if args.inference_bos_boost is not None:
            extra_config["inference_bos_boost"] = args.inference_bos_boost
        if args.inference_eos_boost is not None:
            extra_config["inference_eos_boost"] = args.inference_eos_boost
        if args.decode_audio:
            extra_config["decode_audio"] = True
        if args.output_dir:
            extra_config["output_dir"] = args.output_dir
        if args.save_artifacts:
            extra_config["save_artifacts"] = True
        if args.trim_leading_silence:
            extra_config["trim_leading_silence"] = True
        if args.trim_leading_silence_padding_sec != 0.01:
            extra_config["trim_leading_silence_padding_sec"] = args.trim_leading_silence_padding_sec
        if args.merge_user_channel:
            extra_config["merge_user_channel"] = True

    # S2S Incremental/Session backend options (shared config)
    if args.backend in ("s2s_incremental", "s2s_session"):
        if args.config_path:
            extra_config["config_path"] = args.config_path
        if args.llm_checkpoint_path:
            extra_config["llm_checkpoint_path"] = args.llm_checkpoint_path
        if args.tts_checkpoint_path:
            extra_config["tts_checkpoint_path"] = args.tts_checkpoint_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference
        if args.num_frames_per_inference != 1:
            extra_config["num_frames_per_inference"] = args.num_frames_per_inference
        if args.no_decode_audio:
            extra_config["decode_audio"] = False
        # Response end detection (text-only mode uses eos)
        extra_config["response_end_detection_mode"] = args.response_end_detection_mode
        extra_config["eos_detection_window"] = args.eos_detection_window
        # Artifacts and alignment (available for both backends)
        if args.session_artifacts_dir:
            extra_config["session_artifacts_dir"] = args.session_artifacts_dir
        extra_config["save_session_artifacts"] = not args.no_save_session_artifacts
        extra_config["output_frame_alignment"] = args.output_frame_alignment

    # S2S Incremental V2 backend options
    if args.backend == "s2s_incremental_v2":
        if args.llm_checkpoint_path:
            extra_config["llm_checkpoint_path"] = args.llm_checkpoint_path
        if args.tts_checkpoint_path:
            extra_config["tts_checkpoint_path"] = args.tts_checkpoint_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference
        extra_config["num_frames_per_inference"] = args.num_frames_per_inference if args.num_frames_per_inference != 1 else 3
        extra_config["engine_type"] = args.engine_type
        extra_config["use_perception_cache"] = args.use_perception_cache
        extra_config["use_perception_cudagraph"] = args.use_perception_cudagraph
        extra_config["use_codec_cache"] = args.use_codec_cache
        extra_config["codec_token_history_size"] = args.codec_token_history_size
        extra_config["repetition_penalty"] = args.repetition_penalty
        extra_config["top_p"] = args.top_p
        extra_config["temperature"] = args.temperature
        if args.inference_pad_boost is not None:
            extra_config["inference_pad_boost"] = args.inference_pad_boost
        if args.inference_bos_boost is not None:
            extra_config["inference_bos_boost"] = args.inference_bos_boost
        if args.inference_eos_boost is not None:
            extra_config["inference_eos_boost"] = args.inference_eos_boost
        if args.inference_user_pad_boost is not None:
            extra_config["inference_user_pad_boost"] = args.inference_user_pad_boost
        if args.inference_user_bos_boost is not None:
            extra_config["inference_user_bos_boost"] = args.inference_user_bos_boost
        if args.inference_user_eos_boost is not None:
            extra_config["inference_user_eos_boost"] = args.inference_user_eos_boost
        extra_config["force_turn_taking"] = args.force_turn_taking
        extra_config["force_turn_taking_threshold"] = args.force_turn_taking_threshold
        extra_config["force_turn_taking_pad_window"] = args.force_turn_taking_pad_window
        extra_config["matmul_precision"] = args.matmul_precision
        if args.buffer_size_frames is not None:
            extra_config["buffer_size_frames"] = args.buffer_size_frames
        else:
            extra_config["buffer_size_frames"] = 20 if args.use_perception_cache else 71
        if args.pad_to_duration_secs is not None:
            extra_config["pad_to_duration_secs"] = args.pad_to_duration_secs
            extra_config["silence_padding_sec"] = 0.0
        if args.system_prompt:
            extra_config["system_prompt"] = args.system_prompt
        if args.tts_system_prompt:
            extra_config["tts_system_prompt"] = args.tts_system_prompt
        if args.decode_audio:
            extra_config["decode_audio"] = True
        if args.no_decode_audio:
            extra_config["decode_audio"] = False
        if args.output_dir:
            extra_config["session_artifacts_dir"] = args.output_dir
        elif args.session_artifacts_dir:
            extra_config["session_artifacts_dir"] = args.session_artifacts_dir
        extra_config["save_session_artifacts"] = not args.no_save_session_artifacts
        if args.merge_user_channel_v2:
            extra_config["merge_user_channel"] = True
        if args.use_asr_as_response:
            extra_config["use_asr_as_response"] = True
        extra_config["inference_guidance_enabled"] = not args.no_inference_guidance_enabled
        if args.inference_guidance_scale is not None:
            extra_config["inference_guidance_scale"] = args.inference_guidance_scale
        if args.inference_top_p_or_k is not None:
            extra_config["inference_top_p_or_k"] = args.inference_top_p_or_k
        if args.inference_noise_scale is not None:
            extra_config["inference_noise_scale"] = args.inference_noise_scale
        if args.tts_sliding_window is not None:
            extra_config["tts_sliding_window"] = args.tts_sliding_window
        # Build vLLM configs when using a vLLM engine
        if "vllm" in args.engine_type:
            model_path = args.model
            llm_path = args.llm_checkpoint_path or args.model
            extra_config["vllm_llm_config"] = {
                "model_path": model_path,
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "dtype": "bfloat16",
                "engine_path": None,
                "pretrained_llm": llm_path,
            }
            extra_config["vllm_tts_config"] = {
                "model_path": model_path,
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "dtype": "float32",
                "engine_path": None,
                "pretrained_llm": None,
                "skip_tokenizer_init": True,
            }

    # S2S Session backend options
    if args.backend == "s2s_session":
        extra_config["session_ttl"] = args.session_ttl
        extra_config["max_sessions"] = args.max_sessions

    # Print configuration
    print("=" * 60)
    print("[serve_unified] Starting Unified NeMo Inference Server")
    print("=" * 60)
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model}")
    if args.codec_model:
        print(f"  Codec Model: {args.codec_model}")
    print(f"  Port: {args.port}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Batch Timeout: {args.batch_timeout}s")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    if args.backend == "s2s":
        if args.config_path:
            print(f"  Config Path: {args.config_path}")
        if args.tts_ckpt_path:
            print(f"  TTS Checkpoint: {args.tts_ckpt_path}")
        if args.speaker_reference:
            print(f"  Speaker Reference: {args.speaker_reference}")
        print(f"  Extra Decoding Seconds: {args.extra_decoding_seconds}")
        print(f"  Inference Boosts: pad={args.inference_pad_boost}, bos={args.inference_bos_boost}, eos={args.inference_eos_boost}")
    if args.backend in ("s2s_incremental", "s2s_session"):
        if args.config_path:
            print(f"  Config Path: {args.config_path}")
        if args.llm_checkpoint_path:
            print(f"  LLM Checkpoint: {args.llm_checkpoint_path}")
        if args.speaker_reference:
            print(f"  Speaker Reference: {args.speaker_reference}")
        print(f"  Frames per Inference: {args.num_frames_per_inference}")
        print(f"  Decode Audio: {not args.no_decode_audio}")
        print(f"  Response End Mode: {args.response_end_detection_mode}")
        if args.response_end_detection_mode == "eos":
            print(f"  EOS Detection Window: {args.eos_detection_window} frames")
        print(f"  Save Artifacts: {not args.no_save_session_artifacts}")
        if args.session_artifacts_dir:
            print(f"  Artifacts Dir: {args.session_artifacts_dir}")
        else:
            print("  Artifacts Dir: /tmp/s2s_sessions (default)")
        print(f"  Output Frame Alignment: {args.output_frame_alignment}")
    if args.backend == "s2s_incremental_v2":
        print(f"  Engine Type: {args.engine_type}")
        print(f"  Perception Cache: {args.use_perception_cache}")
        print(f"  Perception CUDAGraph: {args.use_perception_cudagraph}")
        print(f"  Codec Cache: {args.use_codec_cache}")
        print(f"  Buffer Size Frames: {extra_config.get('buffer_size_frames')}")
        print(f"  Frames per Inference: {extra_config.get('num_frames_per_inference')}")
        print(f"  Decode Audio: {extra_config.get('decode_audio', True)}")
        if args.llm_checkpoint_path:
            print(f"  LLM Checkpoint: {args.llm_checkpoint_path}")
        if args.speaker_reference:
            print(f"  Speaker Reference: {args.speaker_reference}")
        if args.pad_to_duration_secs:
            print(f"  Pad to Duration: {args.pad_to_duration_secs}s")
        if args.system_prompt:
            print(f"  System Prompt: {args.system_prompt[:80]}...")
        print(f"  Force Turn Taking: {args.force_turn_taking}")
        print(f"  Save Artifacts: {not args.no_save_session_artifacts}")
    if args.backend == "s2s_session":
        print(f"  Session TTL: {args.session_ttl}s")
        print(f"  Max Sessions: {args.max_sessions}")
    if extra_config:
        print(f"  Extra Config: {extra_config}")
    print("=" * 60)

    # Import and run the unified server
    try:
        import uvicorn

        from recipes.multimodal.server.unified_server import create_app

        app = create_app(
            backend_type=args.backend,
            model_path=args.model,
            codec_model_path=args.codec_model or "",
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            device=args.device,
            dtype=args.dtype,
            extra_config=extra_config if extra_config else None,
        )

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except ImportError as e:
        print(f"[serve_unified] Error: Failed to import unified server: {e}")
        print("[serve_unified] Make sure the recipes.multimodal.server package is in PYTHONPATH")
        sys.exit(1)
    except Exception as e:
        print(f"[serve_unified] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
