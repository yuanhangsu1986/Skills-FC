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
NemotronVoiceChat offline backend that mirrors `examples/speechlm2/nemotron_voicechat_infer.py`.

This backend:
- Loads a NeMo OmegaConf YAML (`--config_path`)
- Applies script-style overrides (checkpoint paths, boosts, speaker ref, extra decoding)
- Resolves config and instantiates `NemotronVoiceChat(OmegaConf.to_container(..., resolve=True))`
- Runs `offline_inference` per request (supports batch/padded inference)

Default behavior is text-only (decode_audio=False), but audio output can be enabled.
Artifacts (input.wav / output.json / output.wav) can be written under nemo-skills `output_dir/`.
"""

import io
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality


@dataclass
class S2SVoiceChatInferConfig(BackendConfig):
    # NeMo config + code injection
    config_path: Optional[str] = None
    code_path: Optional[str] = None

    # Inference knobs (match infer_nano9b_s2s.sh overrides)
    extra_decoding_seconds: float = 0.0
    speaker_reference: Optional[str] = None
    tts_ckpt_path: Optional[str] = None
    inference_pad_boost: float = 0.0
    inference_bos_boost: float = 0.0
    inference_eos_boost: float = 0.0

    # Decoding knobs — passed to NemotronVoiceChat.offline_inference per call.
    # Defaults match offline_inference's own defaults (greedy).
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    # Turn-taking — landed on model.stt.model.force_turn_taking before model construction,
    # then read by DuplexSTTModel internals (duplex_stt_model.py:3566).
    force_turn_taking: bool = False

    # Output behavior
    decode_audio: bool = False  # default text-only; can be enabled
    output_dir: Optional[str] = None
    save_artifacts: bool = False
    # Merge user (input) + model (pred) into two-channel WAV like NeMo ResultsLogger (same timeline for FDB)
    merge_user_channel: bool = False

    # Prompt handling
    ignore_system_prompt: bool = False
    # If set, used as the system prompt for any request that doesn't carry its own
    # (mirrors --system_prompt behavior on s2s_incremental_v2).
    system_prompt: Optional[str] = None

    # Function-channel / tool-call extraction (BFCL eval).
    # decode_function_channel: decode tokens_function_pred → function_channel_text
    #                          (off by default so non-FC pipelines are unaffected).
    # use_function_channel_for_tool_calls + tool_call_parser: read by the server
    #                          framework (unified_server.py) to convert
    #                          function_channel_text into structured tool_calls.
    decode_function_channel: bool = False
    use_function_channel_for_tool_calls: bool = False
    tool_call_parser: Optional[str] = None

    # Audio preprocessing defaults (will be overridden from YAML if present)
    source_sample_rate: int = 16000
    target_sample_rate: int = 22050

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "S2SVoiceChatInferConfig":
        known_fields = {
            "model_path",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "config_path",
            "code_path",
            "extra_decoding_seconds",
            "speaker_reference",
            "tts_ckpt_path",
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "repetition_penalty",
            "force_turn_taking",
            "decode_audio",
            "output_dir",
            "save_artifacts",
            "merge_user_channel",
            "ignore_system_prompt",
            "system_prompt",
            "decode_function_channel",
            "use_function_channel_for_tool_calls",
            "tool_call_parser",
            "source_sample_rate",
            "target_sample_rate",
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


class S2SVoiceChatInferBackend(InferenceBackend):
    @property
    def name(self) -> str:
        return "s2s_voicechat"

    @property
    def supported_modalities(self) -> Set[Modality]:
        # AUDIO_OUT is optional (decode_audio flag), but supported.
        return {Modality.TEXT, Modality.AUDIO_IN, Modality.AUDIO_OUT}

    def __init__(self, config: BackendConfig):
        if isinstance(config, S2SVoiceChatInferConfig):
            self.vc_config = config
        else:
            self.vc_config = S2SVoiceChatInferConfig.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in ["model_path", "device", "dtype", "max_new_tokens", "temperature", "top_p", "top_k"]
                    },
                    **config.extra_config,
                }
            )

        super().__init__(self.vc_config)
        self._tokenizer = None

    def _add_code_path(self) -> None:
        import sys

        code_path = self.vc_config.code_path
        if code_path and code_path not in sys.path:
            sys.path.insert(0, code_path)
            print(f"[S2SVoiceChat] Added {code_path} to PYTHONPATH")

    def _load_cfg(self) -> Any:
        config_path = self.vc_config.config_path
        if not config_path:
            raise RuntimeError("s2s_voicechat requires --config_path pointing to a NeMo YAML (OmegaConf)")
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config path does not exist: {config_path}")
        return OmegaConf.load(config_path)

    def _apply_overrides(self, cfg: Any) -> Any:
        # Use the --model path as pretrained_s2s_model (like infer_nano9b_s2s.sh does).
        if self.config.model_path:
            OmegaConf.update(cfg, "model.stt.model.pretrained_s2s_model", self.config.model_path, force_add=True)

        # TTS override semantics (match Kevin's inference recipe):
        # - `pretrained_model`: checkpoint file (.ckpt/.nemo)
        # - `pretrained_tts_model`: exported model directory (expects config.json inside)
        if self.vc_config.tts_ckpt_path:
            tts_path = self.vc_config.tts_ckpt_path
            if os.path.isdir(tts_path):
                OmegaConf.update(cfg, "model.speech_generation.model.pretrained_tts_model", tts_path, force_add=True)
                # Avoid accidentally also loading weights from an old checkpoint path.
                OmegaConf.update(cfg, "model.speech_generation.model.pretrained_model", None, force_add=True)
            else:
                OmegaConf.update(cfg, "model.speech_generation.model.pretrained_model", tts_path, force_add=True)
                # Ensure we don't trigger directory-based loading by mistake.
                OmegaConf.update(cfg, "model.speech_generation.model.pretrained_tts_model", None, force_add=True)

        if self.vc_config.speaker_reference:
            OmegaConf.update(cfg, "model.inference_speaker_reference", self.vc_config.speaker_reference, force_add=True)

        if self.vc_config.extra_decoding_seconds:
            OmegaConf.update(cfg, "model.extra_decoding_seconds", float(self.vc_config.extra_decoding_seconds), force_add=True)

        # Script defaults / common inference overrides
        OmegaConf.update(cfg, "model.use_asr_timestamps", True, force_add=True)
        OmegaConf.update(cfg, "model.stt.model.eval_text_turn_taking", True, force_add=True)

        # Boosts
        if self.vc_config.inference_pad_boost:
            OmegaConf.update(cfg, "model.stt.model.inference_pad_boost", float(self.vc_config.inference_pad_boost), force_add=True)
        if self.vc_config.inference_bos_boost:
            OmegaConf.update(cfg, "model.stt.model.inference_bos_boost", float(self.vc_config.inference_bos_boost), force_add=True)
        if self.vc_config.inference_eos_boost:
            OmegaConf.update(cfg, "model.stt.model.inference_eos_boost", float(self.vc_config.inference_eos_boost), force_add=True)

        # Forced turn-taking — landed on the STT model's cfg; consumed by
        # DuplexSTTModel internals via self.cfg.get("force_turn_taking", False).
        if self.vc_config.force_turn_taking:
            OmegaConf.update(cfg, "model.stt.model.force_turn_taking", True, force_add=True)

        # Pull sample rates for preprocessing (data.* is what nemotron_voicechat_infer.py uses)
        try:
            sr_in = OmegaConf.select(cfg, "data.source_sample_rate")
            sr_out = OmegaConf.select(cfg, "data.target_sample_rate")
            if sr_in is not None:
                self.vc_config.source_sample_rate = int(sr_in)
            if sr_out is not None:
                self.vc_config.target_sample_rate = int(sr_out)
        except Exception:
            pass

        return cfg

    def load_model(self) -> None:
        print(f"[S2SVoiceChat] Loading NemotronVoiceChat. model={self.config.model_path}")
        self._add_code_path()

        # Match script's inference-safe numerical defaults
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True

        try:
            cfg = self._load_cfg()
            cfg = self._apply_overrides(cfg)
            OmegaConf.resolve(cfg)
            model_config = OmegaConf.to_container(cfg, resolve=True)

            from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat

            self._model = NemotronVoiceChat(model_config).eval()

            dtype = getattr(torch, self.config.dtype, torch.bfloat16)
            try:
                self._model = self._model.to(dtype)
            except Exception as e:
                print(f"[S2SVoiceChat] Warning: dtype conversion to {dtype} failed: {e}")

            self._model = self._model.to(self.config.device)
            self._tokenizer = getattr(getattr(self._model, "stt_model", None), "tokenizer", None)
            self._is_loaded = True
            print("[S2SVoiceChat] Model loaded successfully")
            print(f"  device={self.config.device} dtype={self.config.dtype}")
            print(f"  source_sr={self.vc_config.source_sample_rate} target_sr={self.vc_config.target_sample_rate}")
            print(f"  decode_audio={self.vc_config.decode_audio} extra_decoding_seconds={self.vc_config.extra_decoding_seconds}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to load s2s_voicechat backend: {e}")

    def _clean_special_tokens(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"<\\$[\\d.]+\\$>", "", text)
        text = re.sub(r"<\\|[\\d.]+\\|>", "", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text

    def _tokenize_system_prompts(
        self, system_prompts: List[Optional[str]], batch_size: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.vc_config.ignore_system_prompt or not any(p for p in system_prompts):
            return None, None
        if self._tokenizer is None:
            return None, None

        tokenized: List[List[int]] = []
        for prompt in system_prompts:
            if not prompt:
                tokenized.append([])
                continue
            if hasattr(self._tokenizer, "text_to_ids"):
                tokens = self._tokenizer.text_to_ids(prompt)
            elif hasattr(self._tokenizer, "encode"):
                tokens = self._tokenizer.encode(prompt)
            else:
                tokens = []
            tokenized.append(tokens)

        max_len = max((len(t) for t in tokenized), default=0)
        if max_len == 0:
            return None, None

        pad_id = 0
        if hasattr(self._tokenizer, "pad_id"):
            pad_id = int(self._tokenizer.pad_id)
        elif hasattr(self._tokenizer, "pad_token_id"):
            pad_id = int(self._tokenizer.pad_token_id)

        prompt_tokens = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        prompt_lens = torch.zeros(batch_size, dtype=torch.long)
        for i, tokens in enumerate(tokenized):
            if tokens:
                prompt_tokens[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                prompt_lens[i] = len(tokens)

        return prompt_tokens.to(self.config.device), prompt_lens.to(self.config.device)

    def _load_and_preprocess_audio(self, request: GenerationRequest, temp_files: List[str]) -> Tuple[np.ndarray, str]:
        if not request.audio_bytes and not request.audio_path:
            raise ValueError("Audio input is required for s2s_voicechat backend")

        audio_path = request.audio_path
        if request.audio_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(request.audio_bytes)
            tmp.close()
            audio_path = tmp.name
            temp_files.append(audio_path)

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_sr = int(self.vc_config.source_sample_rate)
        if sr != target_sr:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                import torchaudio

                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
                audio = audio_tensor.squeeze(0).numpy()

        return audio, audio_path

    def _merge_user_model_audio(
        self,
        user_audio: np.ndarray,
        user_sr: int,
        pred_audio: np.ndarray,
        pred_sr: int,
    ) -> np.ndarray:
        """Merge user (input) and model (pred) into a two-channel WAV on one timeline.
        Mirrors NeMo ResultsLogger.merge_and_save_audio: channel 0 = user, channel 1 = pred,
        same length (padded), so FDB ASR (--stereo) gets a single timeline."""
        user = np.asarray(user_audio, dtype=np.float64).squeeze()
        pred = np.asarray(pred_audio, dtype=np.float64).squeeze()
        if user.ndim > 1:
            user = user.mean(axis=1)
        if pred.ndim > 1:
            pred = pred.mean(axis=1)
        # Resample user to pred_sr
        if user_sr != pred_sr:
            try:
                import librosa
                user = librosa.resample(user, orig_sr=user_sr, target_sr=pred_sr)
            except ImportError:
                import torchaudio
                ut = torch.from_numpy(user).float().unsqueeze(0)
                ut = torchaudio.functional.resample(ut, user_sr, pred_sr)
                user = ut.squeeze(0).numpy()
        T1, T2 = user.shape[0], pred.shape[0]
        max_len = max(T1, T2)
        user_pad = np.pad(user, (0, max_len - T1), mode="constant", constant_values=0)
        pred_pad = np.pad(pred, (0, max_len - T2), mode="constant", constant_values=0)
        # (samples, 2): ch0 = user, ch1 = pred (like NeMo .T after cat)
        merged = np.stack([user_pad, pred_pad], axis=1).astype(np.float32)
        return merged

    def _artifact_root(self) -> Optional[str]:
        if not (self.vc_config.save_artifacts and self.vc_config.output_dir):
            return None
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or "local"
        return os.path.join(self.vc_config.output_dir, "artifacts", job_id)

    def _save_artifacts(
        self,
        request_id: str,
        input_audio_path: str,
        output_text: str,
        output_audio_bytes: Optional[bytes],
        debug_info: Dict[str, Any],
        elapsed_ms: float,
    ) -> Optional[Dict[str, str]]:
        root = self._artifact_root()
        if root is None:
            return None
        req_dir = os.path.join(root, request_id)
        os.makedirs(req_dir, exist_ok=True)

        out: Dict[str, str] = {}
        try:
            with open(input_audio_path, "rb") as fin, open(os.path.join(req_dir, "input.wav"), "wb") as fout:
                fout.write(fin.read())
            out["input_wav"] = os.path.join(req_dir, "input.wav")
        except Exception as e:
            print(f"[S2SVoiceChat] Warning: failed saving input.wav: {e}")

        try:
            meta = {
                "request_id": request_id,
                "text": output_text,
                "generation_time_ms": elapsed_ms,
                "debug_info": debug_info,
            }
            meta_path = os.path.join(req_dir, "output.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            out["output_json"] = meta_path
        except Exception as e:
            print(f"[S2SVoiceChat] Warning: failed saving output.json: {e}")

        if output_audio_bytes:
            wav_path = os.path.join(req_dir, "output.wav")
            try:
                with open(wav_path, "wb") as f:
                    f.write(output_audio_bytes)
                out["output_wav"] = wav_path
            except Exception as e:
                print(f"[S2SVoiceChat] Warning: failed saving output.wav: {e}")

        return out or None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        start_time = time.time()
        temp_files: List[str] = []
        results: List[Optional[GenerationResult]] = [None] * len(requests)

        valid_indices: List[int] = []
        audio_list: List[np.ndarray] = []
        system_prompts: List[Optional[str]] = []
        input_audio_paths: Dict[int, str] = {}

        try:
            for i, req in enumerate(requests):
                try:
                    audio, audio_path = self._load_and_preprocess_audio(req, temp_files)
                    audio_list.append(audio)
                    valid_indices.append(i)
                    # Per-request system_prompt wins; fall back to backend default
                    # set via --system_prompt at server-launch time.
                    system_prompts.append(req.system_prompt or self.vc_config.system_prompt)
                    input_audio_paths[i] = audio_path
                except Exception as e:
                    results[i] = GenerationResult(error=str(e), request_id=req.request_id)

            if not audio_list:
                return [r if r is not None else GenerationResult(error="No valid requests") for r in results]

            max_len = max(a.shape[0] for a in audio_list)
            batch_size = len(audio_list)
            batch = torch.zeros(batch_size, max_len, dtype=torch.float32)
            lengths = torch.zeros(batch_size, dtype=torch.long)
            for bi, audio in enumerate(audio_list):
                batch[bi, : len(audio)] = torch.from_numpy(audio).float()
                lengths[bi] = len(audio)

            batch = batch.to(self.config.device)
            lengths = lengths.to(self.config.device)

            prompt_tokens, prompt_lens = self._tokenize_system_prompts(system_prompts, batch_size)

            first_seed = next((r.seed for r in requests if r.seed is not None), None)
            if first_seed is not None:
                random.seed(first_seed)
                np.random.seed(first_seed)
                torch.manual_seed(first_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(first_seed)

            input_pad_len = int(float(self.vc_config.extra_decoding_seconds) * int(self.vc_config.source_sample_rate))

            with torch.no_grad():
                outputs = self._model.offline_inference(
                    input_signal=batch,
                    input_signal_lens=lengths,
                    prompt_tokens=prompt_tokens,
                    prompt_token_lens=prompt_lens,
                    input_pad_len=input_pad_len,
                    decode_audio=bool(self.vc_config.decode_audio),
                    temperature=float(self.vc_config.temperature),
                    top_p=float(self.vc_config.top_p),
                    repetition_penalty=float(self.vc_config.repetition_penalty),
                )

            elapsed_ms = (time.time() - start_time) * 1000.0

            texts = outputs.get("text") or []
            tokens_len = outputs.get("tokens_len")
            asr_hyps = outputs.get("asr_hyps")
            audio_out = outputs.get("audio")
            audio_len = outputs.get("audio_len")

            for bi, req_i in enumerate(valid_indices):
                req = requests[req_i]
                request_id = req.request_id or f"req_{bi}"

                # Text
                text = ""
                if isinstance(texts, list) and bi < len(texts):
                    text = texts[bi] or ""
                elif isinstance(texts, str):
                    text = texts
                text = self._clean_special_tokens(text)

                # Token count
                num_tokens = 0
                if tokens_len is not None:
                    try:
                        num_tokens = int(tokens_len[bi].item())
                    except Exception:
                        num_tokens = 0

                # Audio bytes (optional)
                out_audio_bytes = None
                out_sr = int(self.vc_config.target_sample_rate)
                if self.vc_config.decode_audio and audio_out is not None:
                    try:
                        wav = audio_out[bi]
                        if hasattr(wav, "detach"):
                            wav = wav.detach().float().cpu().numpy()
                        wav = np.asarray(wav).squeeze()
                        # Trim to per-sample input duration to remove batch-padding artifacts.
                        # When multiple samples are batched, shorter ones get padded to the longest,
                        # and the model generates audio for all frames including padding. Trimming to the
                        # input duration (converted to target SR) matches NeMo ResultsLogger behavior and
                        # ensures the output WAV only contains audio corresponding to the actual input.
                        source_sr = int(self.vc_config.source_sample_rate)
                        per_sample_input_len = int(lengths[bi].item()) + input_pad_len
                        per_sample_pred_len = int(per_sample_input_len / source_sr * out_sr)
                        if per_sample_pred_len > 0 and per_sample_pred_len < len(wav):
                            wav = wav[:per_sample_pred_len]
                        elif audio_len is not None:
                            try:
                                n = int(audio_len[bi].item())
                                wav = wav[:n]
                            except Exception:
                                pass
                        max_val = float(np.max(np.abs(wav))) if wav.size else 0.0
                        if max_val > 0:
                            wav = wav / max_val * 0.95
                        if self.vc_config.merge_user_channel:
                            user_audio = audio_list[bi]
                            merged = self._merge_user_model_audio(
                                user_audio,
                                int(self.vc_config.source_sample_rate),
                                wav,
                                out_sr,
                            )
                            buf = io.BytesIO()
                            sf.write(buf, merged, out_sr, format="WAV")
                            out_audio_bytes = buf.getvalue()
                        else:
                            buf = io.BytesIO()
                            sf.write(buf, wav, out_sr, format="WAV")
                            out_audio_bytes = buf.getvalue()
                    except Exception as e:
                        print(f"[S2SVoiceChat] Warning: failed encoding audio for {request_id}: {e}")

                per_req_ms = elapsed_ms / max(1, len(requests))
                debug_info: Dict[str, Any] = {
                    "tokens_len": num_tokens,
                    "decode_audio": bool(self.vc_config.decode_audio),
                    "extra_decoding_seconds": float(self.vc_config.extra_decoding_seconds),
                    "source_sample_rate": int(self.vc_config.source_sample_rate),
                    "target_sample_rate": int(self.vc_config.target_sample_rate),
                }

                # User ASR (if model returns it). We attach the per-request hypothesis to debug_info.
                # Expected shape: list[str] aligned to batch, or a single str.
                user_asr = None
                if asr_hyps is not None:
                    if isinstance(asr_hyps, list):
                        if bi < len(asr_hyps):
                            user_asr = asr_hyps[bi]
                    elif isinstance(asr_hyps, str):
                        user_asr = asr_hyps
                if user_asr:
                    debug_info["asr_hyp"] = user_asr

                artifacts = self._save_artifacts(
                    request_id=request_id,
                    input_audio_path=input_audio_paths.get(req_i, ""),
                    output_text=text,
                    output_audio_bytes=out_audio_bytes,
                    debug_info=debug_info,
                    elapsed_ms=per_req_ms,
                )
                if artifacts:
                    debug_info["artifacts"] = artifacts

                function_channel_text = None
                if self.vc_config.decode_function_channel:
                    function_channel_text = self._decode_function_channel(outputs, bi)

                results[req_i] = GenerationResult(
                    text=text,
                    function_channel_text=function_channel_text,
                    audio_bytes=out_audio_bytes,
                    audio_sample_rate=out_sr,
                    request_id=req.request_id,
                    num_tokens_generated=num_tokens,
                    generation_time_ms=per_req_ms,
                    debug_info=debug_info,
                )

            return [r if r is not None else GenerationResult(error="Unknown error") for r in results]

        finally:
            for p in temp_files:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Function channel decoder (BFCL/FD3/BBA FC eval; off by default)
    # ------------------------------------------------------------------
    def _decode_function_channel(self, output: Dict[str, Any], idx: int = 0) -> Optional[str]:
        """Decode tokens_function_pred → raw text (e.g. '<TOOLCALL>[...]</TOOLCALL>').

        Mirrors the helper in s2s_incremental_backend_v2.  Only called when
        decode_function_channel=True.  Returns None if the model produced no
        function tokens or decoding fails.

        For batched s2s_voicechat output, ``idx`` selects the row to decode;
        tokens_to_str handles batch tensors and returns a list of length B,
        from which we return the requested row.
        """
        func_tokens = output.get("tokens_function_pred")
        if func_tokens is None:
            func_tokens = output.get("tokens_function")
        tokens_len = output.get("tokens_len")
        if func_tokens is None or tokens_len is None:
            return None
        try:
            stt = getattr(self._model, "stt_model", None)
            if stt is None:
                return None
            from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
            # DSFTS STT model exposes user_bos_id; DRIRF (incremental wrapper) uses
            # user_bos_id_text. Pick whichever is present so this works on both forks.
            user_bos_id = getattr(stt, "user_bos_id_text", None)
            if user_bos_id is None:
                user_bos_id = getattr(stt, "user_bos_id", None)
            texts = tokens_to_str(
                func_tokens,
                tokens_len,
                tokenizer=stt.tokenizer,
                pad_id=stt.text_pad_id,
                user_bos_id=user_bos_id,
                eval_text_turn_taking=False,
                sil_id=None,
            )
            if not texts:
                return None
            if idx < len(texts):
                return texts[idx]
            return texts[0]
        except Exception as e:
            print(f"[S2SVoiceChat] Warning: function channel decode failed: {e}")
            return None

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        if not request.audio_bytes and not request.audio_path:
            return "Audio input is required for s2s_voicechat backend"
        return None

