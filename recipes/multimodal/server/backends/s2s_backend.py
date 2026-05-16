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
Speech-to-Speech (S2S) offline backend using NemotronVoiceChat.

Based on inference pattern from:
/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s/NeMo/scripts/training/iad/s2s/sdv2_hf/conv/nano_9b/inf/infer_nano9b_s2s.sh
Config: infer_nano_eartts_Fullduplexbench_Jan22_2026.yaml

This backend takes audio input and produces frame-synchronized TEXT output.
Uses NemotronVoiceChat model with decode_audio=False for text-only output.

Output format:
- "text": Agent's generated response text
- "asr_hyps": User's transcribed text (via ASR scoring)
- "tokens_text": Raw text token IDs (frame-synchronized)

Key parameters matching the latest inference recipe:
- extra_decoding_seconds: Additional decoding time (default 0 for FDB, 20 for Voicebench)
- inference_pad_boost, inference_bos_boost, inference_eos_boost: Token logit adjustments
"""

import io
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch

from .base import (
    BackendConfig,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    Modality,
)


@dataclass
class S2SConfig(BackendConfig):
    """S2S-specific configuration matching the latest inference recipe."""

    # Frame-based processing parameters
    frame_length: float = 0.08  # 80ms frames
    source_sample_rate: int = 16000
    target_sample_rate: int = 22050  # TTS sample rate

    # Role configuration
    input_roles: List[str] = field(default_factory=lambda: ["user", "User"])
    output_roles: List[str] = field(default_factory=lambda: ["agent", "Assistant", "assistant", "Agent"])

    # Model behavior
    predict_user_text: bool = True  # Also transcribe user speech
    decode_audio: bool = True  # Text-only output

    # Extra decoding time - duplex models need additional time to generate response
    # Default 0 for Fullduplexbench, 20 for Voicebench
    extra_decoding_seconds: float = 0.0

    # Legacy parameter (kept for backward compatibility)
    silence_padding_sec: float = 5.0

    # Inference boost parameters (matching the latest recipe)
    inference_pad_boost: float = 0.0
    inference_bos_boost: float = 0.0
    inference_eos_boost: float = 0.0

    # Config path for model configuration YAML
    config_path: Optional[str] = None

    # Speaker reference for TTS (required if decode_audio=True)
    speaker_reference: Optional[str] = None

    # Checkpoint paths
    stt_ckpt_path: Optional[str] = None  # Separate STT checkpoint
    tts_ckpt_path: Optional[str] = None  # Separate TTS checkpoint

    # Code path to add to PYTHONPATH
    code_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "S2SConfig":
        """Create S2S config from dictionary."""
        known_fields = {
            "model_path",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frame_length",
            "source_sample_rate",
            "target_sample_rate",
            "input_roles",
            "output_roles",
            "predict_user_text",
            "decode_audio",
            "extra_decoding_seconds",
            "silence_padding_sec",
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "config_path",
            "speaker_reference",
            "stt_ckpt_path",
            "tts_ckpt_path",
            "code_path",
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


class S2SBackend(InferenceBackend):
    """
    Speech-to-Text inference backend using NemotronVoiceChat.

    This model processes audio frame-by-frame and generates synchronized text.
    With decode_audio=False (default), it produces text output only.

    Supports:
    - Audio input (required)
    - Text output (agent response, synchronized with audio frames)
    - Optionally: User transcription (predict_user_text=True)

    The model generates text tokens at each audio frame, allowing for
    real-time synchronization between input speech and generated response.
    """

    @property
    def name(self) -> str:
        return "s2s"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT, Modality.AUDIO_IN}

    def __init__(self, config: BackendConfig):
        # Convert to S2S-specific config if needed
        if isinstance(config, S2SConfig):
            self.s2s_config = config
        else:
            self.s2s_config = S2SConfig.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in ["model_path", "device", "dtype", "max_new_tokens", "temperature", "top_p", "top_k"]
                    },
                    **config.extra_config,
                }
            )

        super().__init__(self.s2s_config)

        self._tokenizer = None
        self._model_config = None

    def _add_code_path(self):
        """Add code path to PYTHONPATH if specified."""
        import sys

        code_path = self.s2s_config.code_path
        if code_path and code_path not in sys.path:
            sys.path.insert(0, code_path)
            print(f"[S2SBackend] Added {code_path} to PYTHONPATH")

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        from omegaconf import OmegaConf

        config_path = self.s2s_config.config_path
        if config_path and os.path.exists(config_path):
            print(f"[S2SBackend] Loading config from {config_path}")
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)
        return None

    def _build_model_config(self) -> Dict[str, Any]:
        """Build model configuration dict for NemotronVoiceChat."""
        # Start with loaded config or create minimal config
        if self._model_config:
            cfg = self._model_config
        else:
            # Build minimal config
            cfg = {
                "model": {
                    "stt": {
                        "model": {
                            "pretrained_s2s_model": self.config.model_path,
                            "predict_user_text": self.s2s_config.predict_user_text,
                            "inference_pad_boost": self.s2s_config.inference_pad_boost,
                            "inference_bos_boost": self.s2s_config.inference_bos_boost,
                            "inference_eos_boost": self.s2s_config.inference_eos_boost,
                        },
                        "data": {
                            "source_sample_rate": self.s2s_config.source_sample_rate,
                            "target_sample_rate": self.s2s_config.target_sample_rate,
                        },
                        "exp_manager": {"explicit_log_dir": "/tmp/s2s_inference"},
                    },
                    "speech_generation": {
                        "model": {},
                        "data": {
                            "source_sample_rate": self.s2s_config.target_sample_rate,
                            "target_sample_rate": self.s2s_config.target_sample_rate,
                        },
                        "exp_manager": {"explicit_log_dir": "/tmp/s2s_inference"},
                    },
                    "inference_speaker_reference": self.s2s_config.speaker_reference
                    or "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/inference_references/Emma_S3_A1_SC7_singleturntarget_21_channel_1_audio_in.wav",
                    "extra_decoding_seconds": self.s2s_config.extra_decoding_seconds,
                },
                "data": {
                    "frame_length": self.s2s_config.frame_length,
                    "source_sample_rate": self.s2s_config.source_sample_rate,
                    "target_sample_rate": self.s2s_config.target_sample_rate,
                    "input_roles": self.s2s_config.input_roles,
                    "output_roles": self.s2s_config.output_roles,
                },
                "exp_manager": {"explicit_log_dir": "/tmp/s2s_inference"},
            }

        # Override with CLI parameters
        if self.s2s_config.stt_ckpt_path:
            cfg["model"]["stt"]["model"]["pretrained_s2s_model"] = self.s2s_config.stt_ckpt_path
        elif self.config.model_path:
            cfg.setdefault("model", {}).setdefault("stt", {}).setdefault("model", {})[
                "pretrained_s2s_model"
            ] = self.config.model_path

        if self.s2s_config.tts_ckpt_path:
            cfg["model"]["speech_generation"]["model"]["pretrained_model"] = self.s2s_config.tts_ckpt_path

        # Apply inference boosts
        stt_model_cfg = cfg.get("model", {}).get("stt", {}).get("model", {})
        if self.s2s_config.inference_pad_boost:
            stt_model_cfg["inference_pad_boost"] = self.s2s_config.inference_pad_boost
        if self.s2s_config.inference_bos_boost:
            stt_model_cfg["inference_bos_boost"] = self.s2s_config.inference_bos_boost
        if self.s2s_config.inference_eos_boost:
            stt_model_cfg["inference_eos_boost"] = self.s2s_config.inference_eos_boost

        # Apply extra_decoding_seconds
        if self.s2s_config.extra_decoding_seconds:
            cfg["model"]["extra_decoding_seconds"] = self.s2s_config.extra_decoding_seconds

        return cfg

    def load_model(self) -> None:
        """Load the NemotronVoiceChat model."""
        print(f"[S2SBackend] Loading S2S model from {self.config.model_path}...")

        # Add code path if specified
        self._add_code_path()

        try:
            # Load config first
            self._model_config = self._load_model_config()
            model_config = self._build_model_config()

            from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat

            print("[S2SBackend] Using NemotronVoiceChat model")
            self._model = NemotronVoiceChat(model_config)

            # Move to device and set eval mode
            self._model = self._model.eval()

            # Handle dtype
            dtype = getattr(torch, self.config.dtype, torch.bfloat16)
            if hasattr(self._model, "to"):
                try:
                    self._model = self._model.to(dtype)
                except Exception as e:
                    print(f"[S2SBackend] Warning: Could not convert to {dtype}: {e}")

            # Move to device
            self._model = self._model.to(self.config.device)

            # Cache tokenizer
            self._tokenizer = self._model.stt_model.tokenizer

            self._is_loaded = True

            print("[S2SBackend] Model loaded successfully")
            print(f"  Model path: {self.config.model_path}")
            print(f"  Device: {self.config.device}")
            print(f"  decode_audio: {self.s2s_config.decode_audio}")
            print(f"  Frame length: {self.s2s_config.frame_length}s")
            print(f"  Source sample rate: {self.s2s_config.source_sample_rate}")
            print(f"  Extra decoding seconds: {self.s2s_config.extra_decoding_seconds}")
            print(f"  Inference boosts: pad={self.s2s_config.inference_pad_boost}, "
                  f"bos={self.s2s_config.inference_bos_boost}, eos={self.s2s_config.inference_eos_boost}")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import NemotronVoiceChat. Make sure NeMo with speechlm2 "
                f"collection is installed. Error: {e}"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to load S2S model: {e}")

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Generate text responses from audio inputs with batching support."""
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]

        if not requests:
            return []

        start_time = time.time()
        temp_files = []
        results: List[GenerationResult] = [None] * len(requests)
        valid_indices = []
        audio_list = []
        system_prompts = []

        try:
            # Step 1: Load and preprocess all audio and collect system prompts
            for i, req in enumerate(requests):
                try:
                    audio = self._load_and_preprocess_audio(req, temp_files)
                    audio_list.append(audio)
                    valid_indices.append(i)
                    # Collect system prompt (may be None)
                    system_prompts.append(req.system_prompt)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    results[i] = GenerationResult(error=str(e), request_id=req.request_id)

            if not audio_list:
                return results

            # Step 2: Pad to max length and create batch tensor
            max_len = max(a.shape[0] for a in audio_list)
            batch_size = len(audio_list)

            batch_tensor = torch.zeros(batch_size, max_len, dtype=torch.float32)
            lengths = torch.zeros(batch_size, dtype=torch.long)

            for i, audio in enumerate(audio_list):
                batch_tensor[i, : len(audio)] = torch.from_numpy(audio).float()
                lengths[i] = len(audio)

            # Move to device
            batch_tensor = batch_tensor.to(self.config.device)
            lengths = lengths.to(self.config.device)

            # Step 2.5: Tokenize system prompts if any are provided
            prompt_tokens, prompt_token_lens = self._tokenize_system_prompts(system_prompts, batch_size)

            # Set seed if provided
            first_seed = next((r.seed for r in requests if r.seed is not None), None)
            if first_seed is not None:
                self._set_seed(first_seed)

            # Calculate input_pad_len from extra_decoding_seconds
            input_pad_len = int(self.s2s_config.extra_decoding_seconds * self.s2s_config.source_sample_rate)

            # Step 3: Run batched inference
            with torch.no_grad():
                outputs = self._model.offline_inference(
                    input_signal=batch_tensor,
                    input_signal_lens=lengths,
                    prompt_tokens=prompt_tokens,
                    prompt_token_lens=prompt_token_lens,
                    decode_audio=self.s2s_config.decode_audio,
                    input_pad_len=input_pad_len,
                )

            # Diagnostic: when decode_audio=True, log whether model returned audio
            if self.s2s_config.decode_audio:
                out_keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__
                has_audio = isinstance(outputs, dict) and outputs.get("audio") is not None
                print(f"[S2SBackend] decode_audio=True | outputs keys: {out_keys} | has 'audio': {has_audio}")

            # Step 4: Parse outputs back to individual results
            for batch_idx, req_idx in enumerate(valid_indices):
                try:
                    result = self._parse_batch_output(outputs, batch_idx, requests[req_idx])
                    results[req_idx] = result
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    results[req_idx] = GenerationResult(error=str(e), request_id=requests[req_idx].request_id)

            elapsed_ms = (time.time() - start_time) * 1000

            # Update timing info
            for result in results:
                if result is not None and result.is_success():
                    result.generation_time_ms = elapsed_ms / len(requests)

        finally:
            # Clean up temp files
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

        return results

    def _load_and_preprocess_audio(self, request: GenerationRequest, temp_files: List[str]) -> np.ndarray:
        """Load and preprocess audio from a request."""
        if not request.audio_bytes and not request.audio_path:
            raise ValueError("Audio input is required for S2S backend")

        # Handle audio input
        audio_path = request.audio_path
        if request.audio_bytes:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(request.audio_bytes)
            temp_file.close()
            audio_path = temp_file.name
            temp_files.append(audio_path)

        # Load audio
        audio, sr = sf.read(audio_path)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.s2s_config.source_sample_rate:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.s2s_config.source_sample_rate)
            except ImportError:
                import torchaudio

                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.s2s_config.source_sample_rate)
                audio = audio_tensor.squeeze(0).numpy()

        # Legacy: add silence padding if extra_decoding_seconds is not set but silence_padding_sec is
        if self.s2s_config.extra_decoding_seconds <= 0 and self.s2s_config.silence_padding_sec > 0:
            audio = self._add_silence_padding(audio, self.s2s_config.source_sample_rate)

        return audio

    def _clean_special_tokens(self, text: str) -> str:
        """Remove special timing/frame tokens from model output.

        The S2S model outputs special tokens like:
        - <$X.XX$> - energy/confidence markers
        - <|X.XX|> - timing/duration markers

        These should be stripped for clean text output.
        """
        if not text:
            return text
        # Remove <$X.XX$> patterns (energy/confidence)
        text = re.sub(r'<\$[\d.]+\$>', '', text)
        # Remove <|X.XX|> patterns (timing)
        text = re.sub(r'<\|[\d.]+\|>', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _parse_batch_output(
        self, outputs: Dict[str, Any], batch_idx: int, request: GenerationRequest
    ) -> GenerationResult:
        """Parse output for a specific batch index."""
        # Extract text output
        text_output = ""
        if "text" in outputs and outputs["text"]:
            if isinstance(outputs["text"], list) and len(outputs["text"]) > batch_idx:
                text_output = outputs["text"][batch_idx]
            elif not isinstance(outputs["text"], list):
                text_output = outputs["text"]

        # Clean special tokens from output
        text_output = self._clean_special_tokens(text_output)

        # Count tokens if available
        num_tokens = 0
        if "tokens_text" in outputs and outputs["tokens_text"] is not None:
            try:
                tokens = outputs["tokens_text"][batch_idx]
                if hasattr(tokens, "cpu"):
                    tokens = tokens.cpu()
                num_tokens = len(tokens) if hasattr(tokens, "__len__") else tokens.shape[0]
            except (IndexError, TypeError):
                pass

        # Audio output (when decode_audio=True); same contract as s2s_voicechat_infer_backend
        out_audio_bytes = None
        out_sr = int(self.s2s_config.target_sample_rate)
        if self.s2s_config.decode_audio:
            audio_out = outputs.get("audio")
            audio_len = outputs.get("audio_len")
            if audio_out is None and batch_idx == 0:
                print("[S2SBackend] decode_audio=True but outputs has no 'audio' key; model may not return audio.")
            if audio_out is not None:
                try:
                    wav = audio_out[batch_idx]
                    if hasattr(wav, "detach"):
                        wav = wav.detach().float().cpu().numpy()
                    wav = np.asarray(wav).squeeze()
                    if audio_len is not None:
                        try:
                            n = int(audio_len[batch_idx].item())
                            wav = wav[:n]
                        except Exception:
                            pass
                    max_val = float(np.max(np.abs(wav))) if wav.size else 0.0
                    if max_val > 0:
                        wav = wav / max_val * 0.95
                    buf = io.BytesIO()
                    sf.write(buf, wav, out_sr, format="WAV")
                    out_audio_bytes = buf.getvalue()
                except Exception as e:
                    print(f"[S2SBackend] Warning: failed encoding audio for {request.request_id}: {e}")

        return GenerationResult(
            text=text_output,
            audio_bytes=out_audio_bytes,
            audio_sample_rate=out_sr,
            request_id=request.request_id,
            num_tokens_generated=num_tokens,
        )

    def _tokenize_system_prompts(
        self, system_prompts: List[Optional[str]], batch_size: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Tokenize system prompts for the batch.

        Args:
            system_prompts: List of system prompts (may contain None values)
            batch_size: Batch size

        Returns:
            Tuple of (prompt_tokens, prompt_token_lens) or (None, None) if no prompts
        """
        # Check if any system prompts are provided
        if not any(p for p in system_prompts):
            return None, None

        if self._tokenizer is None:
            print("[S2SBackend] Warning: No tokenizer available, skipping system prompts")
            return None, None

        # Tokenize all prompts
        tokenized = []
        for prompt in system_prompts:
            if prompt:
                # Use tokenizer to convert text to token IDs
                if hasattr(self._tokenizer, "text_to_ids"):
                    tokens = self._tokenizer.text_to_ids(prompt)
                elif hasattr(self._tokenizer, "encode"):
                    tokens = self._tokenizer.encode(prompt)
                else:
                    tokens = []
                tokenized.append(tokens)
            else:
                tokenized.append([])

        # Get max length and pad
        max_prompt_len = max(len(t) for t in tokenized) if tokenized else 0
        if max_prompt_len == 0:
            return None, None

        # Get pad token ID
        if hasattr(self._tokenizer, "pad_id"):
            pad_id = self._tokenizer.pad_id
        elif hasattr(self._tokenizer, "pad_token_id"):
            pad_id = self._tokenizer.pad_token_id
        else:
            pad_id = 0

        # Create padded tensor
        prompt_tokens = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.long)
        prompt_token_lens = torch.zeros(batch_size, dtype=torch.long)

        for i, tokens in enumerate(tokenized):
            if tokens:
                prompt_tokens[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                prompt_token_lens[i] = len(tokens)

        # Move to device
        prompt_tokens = prompt_tokens.to(self.config.device)
        prompt_token_lens = prompt_token_lens.to(self.config.device)

        return prompt_tokens, prompt_token_lens

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _add_silence_padding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Append silence to audio (legacy, prefer extra_decoding_seconds)."""
        if self.s2s_config.silence_padding_sec <= 0:
            return audio
        silence_samples = int(self.s2s_config.silence_padding_sec * sample_rate)
        return np.concatenate([audio, np.zeros(silence_samples, dtype=audio.dtype)])

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        """Validate S2S request."""
        if not request.audio_bytes and not request.audio_path:
            return "Audio input is required for S2S backend"
        return None

    def health_check(self) -> Dict[str, Any]:
        """Return health status with S2S-specific info."""
        base = super().health_check()
        if self._is_loaded:
            base.update(
                {
                    "frame_length": self.s2s_config.frame_length,
                    "source_sample_rate": self.s2s_config.source_sample_rate,
                    "extra_decoding_seconds": self.s2s_config.extra_decoding_seconds,
                    "inference_pad_boost": self.s2s_config.inference_pad_boost,
                    "inference_bos_boost": self.s2s_config.inference_bos_boost,
                    "inference_eos_boost": self.s2s_config.inference_eos_boost,
                    "decode_audio": self.s2s_config.decode_audio,
                }
            )
        return base
