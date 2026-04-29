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
Incremental Speech-to-Speech (S2S) backend using NemotronVoiceChat.

This backend processes audio frame-by-frame (80ms frames), simulating real-time
streaming behavior. It produces both text output (agent response + ASR) and
audio output via TTS.

Based on: niva_s2s/niva/core/s2s/inference_streaming_realtime.py
Config: nanov2_demo_model_eartts_updated.yaml
"""

import io
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from transformers import DynamicCache

from .base import (
    BackendConfig,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    Modality,
)

# Streaming parameters
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples
TTS_SAMPLE_RATE = 22050

# Default hyper-parameters
DEFAULT_BUFFER_SIZE_FRAMES = 70
DEFAULT_NUM_FRAMES_PER_INFERENCE = 1
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 60


@dataclass
class S2SIncrementalConfig(BackendConfig):
    """Configuration for incremental S2S backend."""

    # Config file path (YAML)
    config_path: Optional[str] = None

    # Model paths (can override config)
    llm_checkpoint_path: Optional[str] = None
    tts_checkpoint_path: Optional[str] = None
    speaker_reference: Optional[str] = None

    # Frame processing
    buffer_size_frames: int = DEFAULT_BUFFER_SIZE_FRAMES
    num_frames_per_inference: int = DEFAULT_NUM_FRAMES_PER_INFERENCE
    codec_token_history_size: int = DEFAULT_CODEC_TOKEN_HISTORY_SIZE
    silence_padding_sec: float = 5.0

    # Turn-taking
    force_turn_taking: bool = True
    force_turn_taking_threshold: int = 40
    force_turn_taking_pad_window: int = 25

    # Audio decoding
    decode_audio: bool = True

    # Session artifacts saving
    save_session_artifacts: bool = True  # Whether to save input/output artifacts per session
    session_artifacts_dir: str = "/tmp/s2s_sessions"  # Directory to save session artifacts

    # Per-frame alignment output
    output_frame_alignment: bool = False  # Whether to include per-frame alignment in debug output

    # Response end detection (for session backend)
    response_end_detection_mode: str = "audio_energy"  # "audio_energy" or "eos"
    audio_energy_threshold: float = 0.01  # RMS threshold for audio energy detection
    audio_energy_window_sec: float = 0.5  # Window size for audio energy calculation
    max_response_duration_sec: float = 30.0  # Maximum response duration before forced stop
    eos_detection_window: int = 10  # Consecutive PAD tokens to detect EOS

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "S2SIncrementalConfig":
        """Create config from dictionary."""
        known_fields = {
            "model_path",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "config_path",
            "llm_checkpoint_path",
            "tts_checkpoint_path",
            "speaker_reference",
            "buffer_size_frames",
            "num_frames_per_inference",
            "codec_token_history_size",
            "silence_padding_sec",
            "force_turn_taking",
            "force_turn_taking_threshold",
            "force_turn_taking_pad_window",
            "decode_audio",
            "save_session_artifacts",
            "session_artifacts_dir",
            "output_frame_alignment",
            "response_end_detection_mode",
            "audio_energy_threshold",
            "audio_energy_window_sec",
            "max_response_duration_sec",
            "eos_detection_window",
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


class S2SIncrementalBackend(InferenceBackend):
    """
    Incremental Speech-to-Speech backend using NemotronVoiceChat.

    Processes audio frame-by-frame and generates synchronized text + audio output.
    """

    @property
    def name(self) -> str:
        return "s2s_incremental"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT, Modality.AUDIO_IN, Modality.AUDIO_OUT}

    def __init__(self, config: BackendConfig):
        if isinstance(config, S2SIncrementalConfig):
            self.inc_config = config
        else:
            self.inc_config = S2SIncrementalConfig.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in ["model_path", "device", "dtype", "max_new_tokens", "temperature", "top_p", "top_k"]
                    },
                    **config.extra_config,
                }
            )

        super().__init__(self.inc_config)

        self._tokenizer = None
        self._model_cfg = None

        # TTS state
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None
        self.target_sample_rate = TTS_SAMPLE_RATE
        self.target_fps = None

    def _resolve_dtype(self, compute_dtype):
        """Resolve dtype string to torch dtype."""
        if isinstance(compute_dtype, torch.dtype):
            return compute_dtype
        if compute_dtype is None:
            return torch.bfloat16
        if isinstance(compute_dtype, str):
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return mapping.get(compute_dtype.lower(), torch.bfloat16)
        return torch.bfloat16

    def _load_and_merge_configs(self):
        """Load and merge configurations from checkpoint and YAML config."""
        import json

        model_path = self.inc_config.tts_checkpoint_path or self.config.model_path
        llm_path = self.inc_config.llm_checkpoint_path or self.config.model_path

        # Load nano's config (for LLM, perception)
        nano_config_file = os.path.join(llm_path, "config.json")
        print(f"[S2SIncremental] Loading nano config: {nano_config_file}")
        with open(nano_config_file, "r") as f:
            nano_cfg_dict = json.load(f)
        nano_cfg = DictConfig(nano_cfg_dict)

        # Load eartts's config (for TTS) if different path
        if model_path != llm_path:
            eartts_config_file = os.path.join(model_path, "config.json")
            print(f"[S2SIncremental] Loading eartts config: {eartts_config_file}")
            with open(eartts_config_file, "r") as f:
                eartts_cfg_dict = json.load(f)
            eartts_cfg = DictConfig(eartts_cfg_dict)

            # Merge TTS config
            if "model" in eartts_cfg and "speech_generation" in eartts_cfg.model:
                nano_cfg.model.speech_generation = eartts_cfg.model.speech_generation
            if "data" not in nano_cfg:
                nano_cfg.data = eartts_cfg.data

        # Set speaker reference
        speaker_ref = self.inc_config.speaker_reference
        if not speaker_ref and self.inc_config.config_path:
            # Try to get from YAML config
            yaml_cfg = OmegaConf.load(self.inc_config.config_path)
            speaker_ref = yaml_cfg.get("model", {}).get("inference_speaker_reference")

        if speaker_ref:
            if "model" not in nano_cfg:
                nano_cfg.model = {}
            nano_cfg.model.inference_speaker_reference = speaker_ref

        return nano_cfg

    def load_model(self) -> None:
        """Load the NemotronVoiceChat model with TTS support."""
        from safetensors.torch import load_file

        print(f"[S2SIncremental] Loading model from {self.config.model_path}...")

        try:
            from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
            from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import NemotronVoiceChat. Make sure NeMo with speechlm2 "
                f"collection is installed. Error: {e}"
            )

        # Set precision settings
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Load and merge configs
        cfg = self._load_and_merge_configs()
        self._model_cfg = cfg

        # Don't use pretrained paths - we'll load weights manually
        cfg.model.stt.model.pretrained_s2s_model = None
        if hasattr(cfg.model, "speech_generation") and hasattr(cfg.model.speech_generation, "model"):
            cfg.model.speech_generation.model.pretrained_model = None

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Initialize model structure
        print("[S2SIncremental] Initializing model structure...")
        self._model = NemotronVoiceChat(cfg_dict)

        # Load LLM + perception weights
        model_path = self.config.model_path
        llm_path = self.inc_config.llm_checkpoint_path or model_path
        tts_path = self.inc_config.tts_checkpoint_path or model_path

        if llm_path:
            safetensors_path = os.path.join(llm_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"[S2SIncremental] Loading LLM weights from: {llm_path}")
                nano_state_dict = load_file(safetensors_path)

                # Filter out TTS weights
                tts_keys = ["tts_model.", "speech_generation."]
                nano_filtered = {
                    k: v for k, v in nano_state_dict.items() if not any(k.startswith(prefix) for prefix in tts_keys)
                }

                nano_filtered = set_model_dict_for_partial_init(nano_filtered, self._model.state_dict())
                self._model.load_state_dict(nano_filtered, strict=False)

        # Load TTS weights (always load, even if from same path as LLM since we filtered them out above)
        if tts_path:
            safetensors_path = os.path.join(tts_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"[S2SIncremental] Loading TTS weights from: {tts_path}")
                tts_state_dict = load_file(safetensors_path)

                tts_keys_filter = ["tts_model."]
                tts_only = {
                    k: v for k, v in tts_state_dict.items() if any(k.startswith(prefix) for prefix in tts_keys_filter)
                }
                print(f"[S2SIncremental] Loading {len(tts_only)} TTS parameters")

                self._model.load_state_dict(tts_only, strict=False)

        # Setup model
        self.dtype = self._resolve_dtype(self.config.dtype)
        self._model.to(self.config.device)
        self._model.eval()

        # Convert S2S components to configured dtype (keep TTS in float32)
        print(f"[S2SIncremental] Converting S2S components to {self.dtype}")
        self._model.stt_model.llm = self._model.stt_model.llm.to(self.dtype)
        self._model.stt_model.lm_head = self._model.stt_model.lm_head.to(self.dtype)
        self._model.stt_model.embed_tokens = self._model.stt_model.embed_tokens.to(self.dtype)
        self._model.stt_model.asr_head = self._model.stt_model.asr_head.to(self.dtype)
        self._model.stt_model.embed_asr_tokens = self._model.stt_model.embed_asr_tokens.to(self.dtype)

        self._model.on_train_epoch_start()
        self._tokenizer = self._model.stt_model.tokenizer

        # Get TTS info
        if hasattr(self._model, "tts_model") and self.inc_config.decode_audio:
            self.target_fps = self._model.tts_model.target_fps
            self.target_sample_rate = self._model.tts_model.target_sample_rate
            print(f"[S2SIncremental] TTS: fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            self._prepare_tts_initial_state()

        self._is_loaded = True
        print("[S2SIncremental] Model loaded successfully")

    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self._model.stt_model.text_pad_id, device=self.config.device)
        input_embeds = self._model.stt_model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _get_asr_bos_embedding(self):
        """Get ASR BOS embedding."""
        text_bos = torch.full((1,), fill_value=self._model.stt_model.text_pad_id, device=self.config.device)
        input_embeds = self._model.stt_model.embed_asr_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _clone_cache(self, cache):
        """Deep clone cache structures."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        if hasattr(cache, "__dict__"):
            import copy

            return copy.deepcopy(cache)
        return cache

    def _decode_single_token(self, token_id: int, pad_id: int) -> str:
        """Decode a single token to text."""
        try:
            # Use ids_to_tokens which properly handles special tokens
            tokens = self._tokenizer.ids_to_tokens([token_id])
            if tokens:
                token_str = tokens[0]
                # Replace Ġ with space for readability
                token_str = token_str.replace("Ġ", " ")
                return token_str
            return f"<SPEC_{token_id}>"
        except Exception:
            return f"<SPEC_{token_id}>"

    def _init_frame_alignment(self) -> Dict[str, list]:
        """Initialize frame alignment as dict of lists for space efficiency."""
        return {
            "frame_idx": [],
            "user_stream": [],
            "agent_stream_token": [],
            "agent_stream_decoded": [],
            "asr_stream_token": [],
            "asr_stream_decoded": [],
            "is_tts_stop": [],
        }

    def _append_frame_alignment(
        self,
        frame_alignment: Dict[str, list],
        frame_idx: int,
        phase: str,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        pad_id: int,
        is_tts_stop: bool = False,
    ) -> None:
        """Append per-frame alignment information to dict of lists."""
        agent_token = gen_text[0, frame_idx].item() if frame_idx < gen_text.shape[1] else pad_id
        asr_token = gen_asr_text[0, frame_idx].item() if frame_idx < gen_asr_text.shape[1] else pad_id
        frame_alignment["frame_idx"].append(frame_idx)
        frame_alignment["user_stream"].append(phase)
        frame_alignment["agent_stream_token"].append(agent_token)
        frame_alignment["agent_stream_decoded"].append(self._decode_single_token(agent_token, pad_id))
        frame_alignment["asr_stream_token"].append(asr_token)
        frame_alignment["asr_stream_decoded"].append(self._decode_single_token(asr_token, pad_id))
        frame_alignment["is_tts_stop"].append(is_tts_stop)

    def _get_artifacts_dir(self, request_id: str) -> Optional[str]:
        """Get or create artifacts directory for this request."""
        if not self.inc_config.save_session_artifacts:
            return None
        base_dir = self.inc_config.session_artifacts_dir
        artifacts_dir = os.path.join(base_dir, request_id)
        os.makedirs(artifacts_dir, exist_ok=True)
        return artifacts_dir

    def _save_artifacts(
        self,
        artifacts_dir: str,
        input_audio_path: str,
        output_text: str,
        output_audio_bytes: Optional[bytes],
        debug_info: Dict[str, Any],
        generation_time_ms: float,
    ) -> Dict[str, str]:
        """Save input/output artifacts to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Copy input audio
        input_dest = os.path.join(artifacts_dir, f"{timestamp}_input.wav")
        shutil.copy2(input_audio_path, input_dest)

        # Save output audio
        output_audio_path = None
        if output_audio_bytes:
            output_audio_path = os.path.join(artifacts_dir, f"{timestamp}_output.wav")
            with open(output_audio_path, "wb") as f:
                f.write(output_audio_bytes)

        # Save output JSON
        output_json_path = os.path.join(artifacts_dir, f"{timestamp}_output.json")
        with open(output_json_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "text": output_text,
                    "audio_path": output_audio_path,
                    "debug_info": debug_info,
                    "generation_time_ms": generation_time_ms,
                },
                f,
                indent=2,
            )

        return {"artifacts_dir": artifacts_dir, "input_path": input_dest, "output_path": output_audio_path}

    def _generate_dual_channel_audio(
        self,
        artifacts_dir: str,
        input_audio_path: str,
        output_audio_bytes: Optional[bytes],
    ) -> Optional[str]:
        """Generate 2-channel audio (user=ch0, agent=ch1)."""
        import soundfile as sf

        if not output_audio_bytes:
            return None

        output_sr = TTS_SAMPLE_RATE

        # Load user audio
        try:
            user_audio, user_sr = sf.read(input_audio_path)
            if user_sr != output_sr:
                import scipy.signal

                user_audio = scipy.signal.resample(user_audio, int(len(user_audio) * output_sr / user_sr))
            if len(user_audio.shape) > 1:
                user_audio = user_audio[:, 0]
        except Exception as e:
            print(f"[S2SIncremental] Error reading user audio: {e}")
            return None

        # Load agent audio
        try:
            agent_audio, agent_sr = sf.read(io.BytesIO(output_audio_bytes))
            if agent_sr != output_sr:
                import scipy.signal

                agent_audio = scipy.signal.resample(agent_audio, int(len(agent_audio) * output_sr / agent_sr))
            if len(agent_audio.shape) > 1:
                agent_audio = agent_audio[:, 0]
        except Exception as e:
            print(f"[S2SIncremental] Error reading agent audio: {e}")
            return None

        # Create 2-channel audio (zero-padded to max length)
        max_len = max(len(user_audio), len(agent_audio))
        stereo = np.zeros((max_len, 2), dtype=np.float32)
        stereo[: len(user_audio), 0] = user_audio
        stereo[: len(agent_audio), 1] = agent_audio

        # Normalize
        max_val = np.abs(stereo).max()
        if max_val > 0:
            stereo = stereo / max_val * 0.95

        output_path = os.path.join(artifacts_dir, "dual_channel.wav")
        sf.write(output_path, stereo, output_sr)
        print(f"[S2SIncremental] Generated dual-channel audio: {output_path}")
        return output_path

    def _prepare_tts_initial_state(self):
        """Prepare TTS warmup state with speaker reference."""
        from nemo.collections.audio.parts.utils.resampling import resample
        from nemo.collections.speechlm2.parts.precision import fp32_precision

        if not hasattr(self._model, "tts_model"):
            return

        speaker_ref = None
        if self._model_cfg and hasattr(self._model_cfg, "model"):
            speaker_ref = self._model_cfg.model.get("inference_speaker_reference")
        if not speaker_ref:
            speaker_ref = self.inc_config.speaker_reference

        if not speaker_ref:
            print("[S2SIncremental] Warning: No speaker reference, TTS disabled")
            return

        print(f"[S2SIncremental] Preparing TTS with speaker: {speaker_ref}")

        with fp32_precision():
            speaker_audio, speaker_sr = torchaudio.load(speaker_ref)
            speaker_audio = resample(speaker_audio, speaker_sr, self._model.tts_model.target_sample_rate)

        speaker_audio = speaker_audio.to(self.config.device)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.config.device).long()

        self._model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
        )
        init_inputs = self._model.tts_model.get_init_inputs(B=1)

        self.generation_config = self._model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        # Debug: print generation config
        print(f"[S2SIncremental] TTS generation_config: {self.generation_config}")

        with torch.no_grad():
            outputs = self._model.tts_model.tts_model(**init_inputs)
            code = init_inputs["code"][:, -1:]

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)

        # Debug: print TTS state shapes
        print(f"[S2SIncremental] first_context_subword_id shape: {self.first_context_subword_id.shape}")
        print(f"[S2SIncremental] first_context_subword_id value: {self.first_context_subword_id}")
        print(f"[S2SIncremental] first_tts_code_input shape: {self.first_tts_code_input.shape}")
        print(f"[S2SIncremental] first_tts_code_input value: {self.first_tts_code_input}")
        print(f"[S2SIncremental] codec_silence_tokens: {self._model.tts_model.codec_silence_tokens}")
        print(f"[S2SIncremental] codec_token_history_size: {self.inc_config.codec_token_history_size}")

        print("[S2SIncremental] TTS warmup state prepared")

    def _samples_per_audio_output_frame(self):
        """Calculate samples per audio output frame."""
        rate = self.target_sample_rate or TTS_SAMPLE_RATE
        return int(float(rate) * FRAME_SIZE_SEC)

    def _update_audio_buffer(self, audio_buffer, buffer_fill_level, new_audio, buffer_size_samples):
        """Update sliding window audio buffer."""
        if new_audio.shape[1] == 0:
            current_buffer = audio_buffer[:, :buffer_fill_level]
            return audio_buffer, buffer_fill_level, current_buffer

        remaining = new_audio

        if buffer_fill_level < buffer_size_samples and remaining.shape[1] > 0:
            warmup_take = min(buffer_size_samples - buffer_fill_level, remaining.shape[1])
            if warmup_take > 0:
                audio_buffer[:, buffer_fill_level : buffer_fill_level + warmup_take] = remaining[:, :warmup_take]
                buffer_fill_level += warmup_take
                remaining = remaining[:, warmup_take:]

        if remaining.shape[1] > 0:
            if remaining.shape[1] >= buffer_size_samples:
                audio_buffer = remaining[:, -buffer_size_samples:]
            else:
                audio_buffer = torch.cat([audio_buffer[:, remaining.shape[1] :], remaining], dim=1)
            buffer_fill_level = buffer_size_samples

        current_buffer = (
            audio_buffer if buffer_fill_level == buffer_size_samples else audio_buffer[:, :buffer_fill_level]
        )
        return audio_buffer, buffer_fill_level, current_buffer

    def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr):
        """Apply forced turn-taking rules based on ASR tokens."""
        if not self.inc_config.force_turn_taking:
            return

        threshold = self.inc_config.force_turn_taking_threshold
        pad_window_steps = self.inc_config.force_turn_taking_pad_window

        B = gen_text.size(0)
        for batch_idx in range(B):
            lookback_start = max(0, t - threshold)
            agent_text_window = gen_text[batch_idx, lookback_start:t]

            if t < pad_window_steps:
                continue

            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = gen_asr[batch_idx, pad_lookback_start:t]
            has_pad_window = (
                (asr_recent_tokens == self._model.stt_model.text_pad_id).all() if len(asr_recent_tokens) > 0 else False
            )

            if has_pad_window and pad_lookback_start > 0:
                token_before_window = gen_asr[batch_idx, pad_lookback_start - 1]
                has_pad_window = token_before_window != self._model.stt_model.text_pad_id
            elif has_pad_window and pad_lookback_start == 0:
                has_pad_window = False

            if has_pad_window:
                if not (agent_text_window == self._model.stt_model.text_bos_id).any():
                    gen_text[batch_idx, t] = self._model.stt_model.text_bos_id

    def infer_one_step(
        self,
        audio_input,
        num_frames_per_inference,
        frame_idx,
        gen_text,
        audio_toks_buffer,
        input_embeds_history,
        dynamic_cache,
        embedding_position,
        past_key_values,
        code,
        subword_mask,
        gen_asr_text,
    ):
        """Process one inference step (potentially multiple frames)."""
        from nemo.collections.speechlm2.parts.precision import fp32_precision

        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]
        device = self.config.device

        predicted_tokens = torch.empty((batch_size, num_frames_per_inference), dtype=gen_text.dtype, device=device)
        asr_predicted_tokens = torch.empty((batch_size, num_frames_per_inference), dtype=gen_text.dtype, device=device)
        tts_silence_mask = torch.ones((batch_size, num_frames_per_inference), dtype=torch.bool, device=device)

        # Perception step
        buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=device)
        source_encoded, _, _ = self._model.stt_model.perception(
            input_signal=audio_input,
            input_signal_length=buffer_len,
            return_encoder_emb=True,
        )
        source_encoded = source_encoded.to(self.dtype)
        total_encoded_frames = source_encoded.shape[1]

        if embedding_position < 0:
            newest_frame_index = total_encoded_frames + embedding_position
        else:
            newest_frame_index = embedding_position

        base_frame_index = newest_frame_index - (num_frames_per_inference - 1)
        base_frame_index = max(base_frame_index, 0)

        new_input_embeds = []
        decode_audio = self.inc_config.decode_audio and hasattr(self._model, "tts_model")

        for chunk_offset in range(num_frames_per_inference):
            current_frame_idx = frame_idx + chunk_offset
            current_frame_index = min(base_frame_index + chunk_offset, total_encoded_frames - 1)
            current_frame_embedding = source_encoded[:, current_frame_index : current_frame_index + 1, :]

            current_input_emb = current_frame_embedding.clone()
            current_input_emb *= self._model.stt_model.cfg.get("duplex_nano_channel_weight", 1.0)

            if current_frame_idx == 0:
                current_input_emb += self._get_bos_embedding()
                current_input_emb += self._get_asr_bos_embedding()
            else:
                last_token_emb = self._model.stt_model.embed_tokens(gen_text[:, current_frame_idx - 1])
                current_input_emb += last_token_emb
                last_asr_token_emb = self._model.stt_model.embed_asr_tokens(gen_asr_text[:, current_frame_idx - 1])
                current_input_emb += last_asr_token_emb

            # Forward pass
            if use_cache:
                ans = self._model.stt_model(current_input_emb, cache=dynamic_cache)
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self._model.stt_model(full_input_embeds, cache=None)

            # Sample tokens
            predicted_token = ans["text_logits"][:, -1].argmax(dim=-1)
            asr_predicted_token = ans["asr_logits"][:, -1].argmax(dim=-1)

            gen_text[:, current_frame_idx] = predicted_token
            predicted_tokens[:, chunk_offset] = predicted_token
            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, chunk_offset] = asr_predicted_token

            # Apply turn-taking
            self._maybe_apply_forced_turn_taking(current_frame_idx, gen_text, gen_asr_text)
            predicted_tokens[:, chunk_offset] = gen_text[:, current_frame_idx]

            # TTS step
            if decode_audio and self.generation_config is not None:
                current_subword_id = gen_text[:, current_frame_idx].unsqueeze(-1)

                if current_frame_idx == 0:
                    prev_subword_id = self.first_context_subword_id
                else:
                    prev_subword_id = gen_text[:, current_frame_idx - 1].unsqueeze(-1)

                current_subword_mask = subword_mask[:, current_frame_idx].unsqueeze(-1)

                # Debug TTS inputs for first few frames
                if current_frame_idx < 3:
                    print(
                        f"[DEBUG TTS frame {current_frame_idx}] current_subword_id: {current_subword_id.item()}, prev_subword_id: {prev_subword_id.item()}"
                    )
                    print(f"[DEBUG TTS frame {current_frame_idx}] current_subword_mask: {current_subword_mask}")
                    print(
                        f"[DEBUG TTS frame {current_frame_idx}] prev_audio_tokens shape: {code.shape}, values: {code[0, 0, :5]}"
                    )

                try:
                    code, past_key_values = self._model.tts_model.infer_codes_one_step(
                        current_subword_id=current_subword_id,
                        prev_subword_id=prev_subword_id,
                        current_subword_mask=current_subword_mask,
                        prev_audio_tokens=code,
                        past_key_values=past_key_values,
                        guidance_enabled=True,
                        generation_config=self.generation_config,
                        ignore_eos_flag_stop=True,
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"[S2SIncremental] TTS infer_codes_one_step failed at frame {current_frame_idx}: {e}"
                    ) from e

                # Debug TTS output for first few frames
                if current_frame_idx < 3:
                    print(f"[DEBUG TTS frame {current_frame_idx}] NEW code: {code[0, 0, :5]}")

                audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:], code], dim=1)

                # Handle silence on EOS
                if self._model.cfg.get("inference_force_speech_silence_on_eos", None):
                    silence_codes = self._model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self._model.tts_model.text_eos_id,
                        silence_codes,
                        code,
                    )

                # Mark whether this frame's generated codec tokens correspond to silence
                # (shape of `code` is [B, 1, C]; silence_codes is broadcast-compatible).
                try:
                    silence_codes = self._model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    tts_silence_mask[:, chunk_offset] = (code == silence_codes).all(dim=-1).squeeze(1)
                except Exception:
                    # If something goes wrong (unexpected shapes), keep default True (treat as silence)
                    tts_silence_mask[:, chunk_offset] = True

        # Decode audio
        decoded_audio_new = None
        if decode_audio and audio_toks_buffer is not None:
            samples_per_frame = self._samples_per_audio_output_frame()
            len_audio_toks_buffer = torch.tensor(
                [self.inc_config.codec_token_history_size], dtype=torch.long, device=device
            )

            # Debug: print audio_toks_buffer info
            if frame_idx == 0:
                print(f"[DEBUG] audio_toks_buffer shape: {audio_toks_buffer.shape}")
                print(f"[DEBUG] audio_toks_buffer dtype: {audio_toks_buffer.dtype}")
                print(f"[DEBUG] audio_toks_buffer sample values: {audio_toks_buffer[0, :3, :5]}")

            try:
                with fp32_precision(), torch.no_grad():
                    decoded_audio, _ = self._model.tts_model.audio_codec.decode(
                        audio_toks_buffer, len_audio_toks_buffer
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"[S2SIncremental] Audio codec decode failed at frame {frame_idx}: {e}"
                ) from e

            # Debug: print decoded audio info
            if frame_idx == 0:
                print(f"[DEBUG] decoded_audio shape: {decoded_audio.shape}")
                print(f"[DEBUG] decoded_audio dtype: {decoded_audio.dtype}")
                print(f"[DEBUG] decoded_audio min/max: {decoded_audio.min():.4f} / {decoded_audio.max():.4f}")

            decoded_audio_new = decoded_audio[:, :, -samples_per_frame * num_frames_per_inference :]

        # Convert tokens to text
        predicted_text_strs = []
        for predicted_tok_ids_b in predicted_tokens:
            toks = self._tokenizer.ids_to_tokens(predicted_tok_ids_b.tolist())
            toks = [t.replace("<SPECIAL_12>", "").replace("Ġ", " ") for t in toks]
            predicted_text_strs.append("".join(toks))

        asr_predicted_text_strs = []
        for asr_tok_ids_b in asr_predicted_tokens:
            toks = self._tokenizer.ids_to_tokens(asr_tok_ids_b.tolist())
            toks = [t.replace("<SPECIAL_12>", "").replace("Ġ", " ") for t in toks]
            asr_predicted_text_strs.append("".join(toks))

        return {
            "predicted_text_tokens": predicted_tokens,
            "asr_predicted_text_tokens": asr_predicted_tokens,
            "audio_toks_buffer": audio_toks_buffer,
            "decoded_audio_new": decoded_audio_new,
            "predicted_text_strs": predicted_text_strs,
            "asr_predicted_text_strs": asr_predicted_text_strs,
            "tts_silence_mask": tts_silence_mask,
            "input_embeds_history": input_embeds_history + new_input_embeds if not use_cache else input_embeds_history,
            "dynamic_cache": dynamic_cache if use_cache else None,
            "past_key_values": past_key_values,
            "code": code,
        }

    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, num_frames_per_inference: int = None):
        """
        Perform incremental streaming inference on audio file.

        Args:
            audio_path: Path to input audio file
            num_frames_per_inference: Frames to process per step (default: 1)

        Returns:
            Dict with 'text', 'asr_text', 'audio' outputs
        """
        import librosa
        from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str

        if num_frames_per_inference is None:
            num_frames_per_inference = self.inc_config.num_frames_per_inference

        device = self.config.device
        buffer_size_frames = self.inc_config.buffer_size_frames
        buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES

        # Load audio
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Add silence padding
        if self.inc_config.silence_padding_sec > 0:
            silence_samples = int(self.inc_config.silence_padding_sec * SAMPLE_RATE)
            audio_signal = np.concatenate([audio_signal, np.zeros(silence_samples)])

        total_samples = len(audio_signal)

        # Calculate frames
        total_frames_maybe = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES))
        num_inference_steps = total_frames_maybe // num_frames_per_inference
        if total_frames_maybe % num_frames_per_inference != 0:
            num_inference_steps += 1
        total_frames = num_inference_steps * num_frames_per_inference

        # Pad audio
        padded_samples = num_inference_steps * num_frames_per_inference * FRAME_SIZE_SAMPLES
        if padded_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_samples - total_samples))

        # Audio must be float32 for AudioPreprocessor accuracy
        audio_tensor = torch.tensor(audio_signal, dtype=torch.float32, device=device).unsqueeze(0)

        # Check cache support
        use_cache = "Nemotron" not in self._model.stt_model.cfg.pretrained_llm

        # Initialize buffers (float32 for audio preprocessing)
        audio_buffer = torch.zeros(1, buffer_size_samples, dtype=torch.float32, device=device)
        buffer_fill_level = 0

        if use_cache:
            llm_cache = DynamicCache()
        else:
            llm_cache = None
            input_embeds_history = []

        # Initialize TTS state
        decode_audio = self.inc_config.decode_audio and hasattr(self._model, "tts_model")
        code = None
        past_key_values = None
        subword_mask = None
        audio_toks_buffer = None

        if decode_audio:
            audio_toks_buffer = (
                self._model.tts_model.codec_silence_tokens.view(1, 1, -1)
                .expand(-1, self.inc_config.codec_token_history_size, -1)
                .to(device)
            )

            if self.first_tts_past_key_values_input is not None:
                past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
                code = self.first_tts_code_input.detach().clone()
                subword_mask = torch.ones(1, total_frames, device=device, dtype=torch.bool)

        gen_text = torch.full((1, total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long)
        gen_asr_text = torch.full(
            (1, total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
        )

        audio_segments = []
        frame_alignment = self._init_frame_alignment() if self.inc_config.output_frame_alignment else None
        pad_id = self._model.stt_model.text_pad_id

        # Frame-by-frame processing
        frame_idx = 0
        while frame_idx < total_frames:
            slice_start = frame_idx * FRAME_SIZE_SAMPLES
            slice_n_samples = num_frames_per_inference * FRAME_SIZE_SAMPLES
            new_audio = audio_tensor[:, slice_start : slice_start + slice_n_samples]

            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, new_audio, buffer_size_samples
            )

            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_inference=num_frames_per_inference,
                frame_idx=frame_idx,
                gen_text=gen_text,
                audio_toks_buffer=audio_toks_buffer if decode_audio else None,
                input_embeds_history=input_embeds_history if not use_cache else [],
                dynamic_cache=llm_cache if use_cache else None,
                embedding_position=-1,
                past_key_values=past_key_values if decode_audio else None,
                code=code if decode_audio else None,
                subword_mask=subword_mask if decode_audio else None,
                gen_asr_text=gen_asr_text,
            )

            if not use_cache:
                input_embeds_history = result["input_embeds_history"]
            llm_cache = result["dynamic_cache"]

            if decode_audio:
                audio_toks_buffer = result["audio_toks_buffer"]
                if result["decoded_audio_new"] is not None:
                    audio_segments.append(result["decoded_audio_new"])
                past_key_values = result["past_key_values"]
                code = result["code"]

            # Collect frame alignment
            if frame_alignment is not None:
                for i in range(num_frames_per_inference):
                    fi = frame_idx + i
                    if fi < total_frames:
                        self._append_frame_alignment(frame_alignment, fi, "user_turn", gen_text, gen_asr_text, pad_id)

            frame_idx += num_frames_per_inference

        # Prepare outputs
        gen_text = gen_text[:, :total_frames]
        gen_asr_text = gen_asr_text[:, :total_frames]
        lengths = torch.tensor([total_frames], dtype=torch.long, device=device)

        text_output = tokens_to_str(
            gen_text,
            lengths,
            tokenizer=self._tokenizer,
            pad_id=self._model.stt_model.text_pad_id,
            eval_text_turn_taking=True,
        )
        asr_text_output = tokens_to_str(
            gen_asr_text,
            lengths,
            tokenizer=self._tokenizer,
            pad_id=self._model.stt_model.text_pad_id,
            eval_text_turn_taking=True,
        )

        output_audio = None
        if audio_segments:
            output_audio = torch.cat(audio_segments, dim=-1)
            print(f"[DEBUG] Final output_audio shape: {output_audio.shape}")
            print(f"[DEBUG] Final output_audio min/max: {output_audio.min():.4f} / {output_audio.max():.4f}")
            print(f"[DEBUG] Final output_audio mean/std: {output_audio.mean():.6f} / {output_audio.std():.4f}")
            print(f"[DEBUG] Number of audio segments: {len(audio_segments)}")

        debug_info = {"total_frames": total_frames}
        if frame_alignment is not None:
            debug_info["frame_alignment"] = frame_alignment

        return {
            "text": text_output,
            "asr_text": asr_text_output,
            "audio": output_audio,
            "tokens_text": gen_text,
            "tokens_len": lengths,
            "debug_info": debug_info,
            "input_audio_path": audio_path,
        }

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Generate text + audio responses from audio inputs."""
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]

        if not requests:
            return []

        results = []

        for req in requests:
            start_time = time.time()
            temp_file_path = None

            try:
                # Get audio input
                audio_path = req.audio_path
                if req.audio_bytes:
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_file.write(req.audio_bytes)
                    temp_file.close()
                    temp_file_path = temp_file.name
                    audio_path = temp_file_path

                if not audio_path:
                    results.append(
                        GenerationResult(
                            error="Audio input required for s2s_incremental backend", request_id=req.request_id
                        )
                    )
                    continue

                # Run inference
                output = self.inference_realtime_streaming(
                    audio_path=audio_path,
                    num_frames_per_inference=self.inc_config.num_frames_per_inference,
                )

                # Encode audio to bytes
                audio_bytes = None
                if output["audio"] is not None:
                    audio_np = output["audio"].float().cpu().numpy().squeeze()
                    max_val = np.abs(audio_np).max()
                    if max_val > 0:
                        audio_np = audio_np / max_val * 0.95
                    wav_buffer = io.BytesIO()
                    import soundfile as sf

                    sf.write(wav_buffer, audio_np, self.target_sample_rate, format="WAV")
                    audio_bytes = wav_buffer.getvalue()

                elapsed_ms = (time.time() - start_time) * 1000
                output_text = output["text"][0] if output["text"] else ""
                debug_info = output.get("debug_info", {})

                # Save artifacts if enabled
                request_id = req.request_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                artifacts_dir = self._get_artifacts_dir(request_id)
                response_audio_bytes = audio_bytes
                response_sample_rate = self.target_sample_rate

                if artifacts_dir:
                    self._save_artifacts(artifacts_dir, audio_path, output_text, audio_bytes, debug_info, elapsed_ms)
                    # Generate dual-channel audio and use it as the response
                    dual_path = self._generate_dual_channel_audio(artifacts_dir, audio_path, audio_bytes)
                    if dual_path:
                        debug_info["dual_channel_audio_path"] = dual_path
                        # Read dual-channel audio to return to client
                        with open(dual_path, "rb") as f:
                            response_audio_bytes = f.read()
                        response_sample_rate = TTS_SAMPLE_RATE  # Dual-channel uses TTS sample rate

                results.append(
                    GenerationResult(
                        text=output_text,
                        audio_bytes=response_audio_bytes,
                        audio_sample_rate=response_sample_rate,
                        request_id=req.request_id,
                        generation_time_ms=elapsed_ms,
                        debug_info=debug_info,
                    )
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                results.append(GenerationResult(error=str(e), request_id=req.request_id))

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        return results

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        """Validate request for incremental S2S."""
        if not request.audio_bytes and not request.audio_path:
            return "Audio input is required for s2s_incremental backend"
        return None

    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        base = super().health_check()
        if self._is_loaded:
            base.update(
                {
                    "buffer_size_frames": self.inc_config.buffer_size_frames,
                    "num_frames_per_inference": self.inc_config.num_frames_per_inference,
                    "decode_audio": self.inc_config.decode_audio,
                    "target_sample_rate": self.target_sample_rate,
                    "tts_enabled": self.generation_config is not None,
                }
            )
        return base
