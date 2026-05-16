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
Session-aware Speech-to-Speech (S2S) backend.

Extends S2SIncrementalBackend to support multi-turn conversations by
persisting LLM KV cache and other state between HTTP requests.
"""

import io
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.signal
import torch

from ..session_manager import SessionState, TurnData
from .base import GenerationRequest, GenerationResult
from .s2s_incremental_backend import (
    FRAME_SIZE_SAMPLES,
    FRAME_SIZE_SEC,
    SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    S2SIncrementalBackend,
)


class S2SSessionBackend(S2SIncrementalBackend):
    """
    Session-aware S2S backend that persists state between requests.

    Extends S2SIncrementalBackend to:
    - Accept session state from SessionManager
    - Restore LLM cache, frame index, and buffers from session
    - Return updated state for saving back to session
    """

    @property
    def name(self) -> str:
        return "s2s_session"

    def _initialize_fresh_state(self, total_frames: int, device: str) -> Dict[str, Any]:
        """Initialize fresh state for a new session."""
        from transformers import DynamicCache

        use_cache = "Nemotron" not in self._model.stt_model.cfg.pretrained_llm

        state = {
            "frame_idx": 0,
            "gen_text": torch.full(
                (1, total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
            ),
            "gen_asr_text": torch.full(
                (1, total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
            ),
            "audio_buffer": torch.zeros(
                1, self.inc_config.buffer_size_frames * FRAME_SIZE_SAMPLES, dtype=torch.float32, device=device
            ),
            "buffer_fill_level": 0,
            "llm_cache": DynamicCache() if use_cache else None,
            "input_embeds_history": [] if not use_cache else None,
        }
        return state

    def _restore_state_from_session(
        self, session_state: SessionState, total_frames: int, device: str
    ) -> Dict[str, Any]:
        """Restore state from session, extending buffers if needed."""
        frame_idx = session_state.frame_idx

        # We need to extend gen_text and gen_asr_text to accommodate new frames
        new_total_frames = frame_idx + total_frames

        # Restore or create gen_text
        if session_state.gen_text is not None:
            old_gen_text = session_state.gen_text.to(device)
            if old_gen_text.shape[1] < new_total_frames:
                gen_text = torch.full(
                    (1, new_total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
                )
                gen_text[:, : old_gen_text.shape[1]] = old_gen_text
            else:
                gen_text = old_gen_text
        else:
            gen_text = torch.full(
                (1, new_total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
            )

        # Restore or create gen_asr_text
        if session_state.gen_asr_text is not None:
            old_gen_asr_text = session_state.gen_asr_text.to(device)
            if old_gen_asr_text.shape[1] < new_total_frames:
                gen_asr_text = torch.full(
                    (1, new_total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
                )
                gen_asr_text[:, : old_gen_asr_text.shape[1]] = old_gen_asr_text
            else:
                gen_asr_text = old_gen_asr_text
        else:
            gen_asr_text = torch.full(
                (1, new_total_frames), self._model.stt_model.text_pad_id, device=device, dtype=torch.long
            )

        # Restore audio buffer
        buffer_size_samples = self.inc_config.buffer_size_frames * FRAME_SIZE_SAMPLES
        if session_state.audio_buffer is not None:
            audio_buffer = session_state.audio_buffer.to(device)
        else:
            audio_buffer = torch.zeros(1, buffer_size_samples, dtype=torch.float32, device=device)

        state = {
            "frame_idx": frame_idx,
            "gen_text": gen_text,
            "gen_asr_text": gen_asr_text,
            "audio_buffer": audio_buffer,
            "buffer_fill_level": session_state.buffer_fill_level,
            "llm_cache": session_state.llm_cache,
            "input_embeds_history": session_state.input_embeds_history,
        }

        return state

    def _save_state_to_session(self, session_state: SessionState, state: Dict[str, Any]):
        """Save current state back to session."""
        session_state.frame_idx = state["frame_idx"]
        session_state.gen_text = state["gen_text"].detach().cpu()
        session_state.gen_asr_text = state["gen_asr_text"].detach().cpu()
        session_state.audio_buffer = state["audio_buffer"].detach().cpu()
        session_state.buffer_fill_level = state["buffer_fill_level"]
        session_state.llm_cache = state["llm_cache"]
        session_state.input_embeds_history = state.get("input_embeds_history")

    def _generate_dual_channel_audio_for_turn(
        self,
        input_audio_path: str,
        output_audio_bytes: Optional[bytes],
    ) -> tuple[Optional[bytes], int]:
        """
        Generate 2-channel audio (user=ch0, agent=ch1) for a single turn.

        Returns:
            Tuple of (audio_bytes, sample_rate) or (None, 0) if failed.
        """
        import soundfile as sf

        if not output_audio_bytes:
            return None, 0

        output_sr = TTS_SAMPLE_RATE

        # Load user audio
        try:
            user_audio, user_sr = sf.read(input_audio_path)
            if user_sr != output_sr:
                user_audio = scipy.signal.resample(user_audio, int(len(user_audio) * output_sr / user_sr))
            if len(user_audio.shape) > 1:
                user_audio = user_audio[:, 0]
        except Exception as e:
            print(f"[S2SSession] Error reading user audio: {e}")
            return None, 0

        # Load agent audio
        try:
            agent_audio, agent_sr = sf.read(io.BytesIO(output_audio_bytes))
            if agent_sr != output_sr:
                agent_audio = scipy.signal.resample(agent_audio, int(len(agent_audio) * output_sr / agent_sr))
            if len(agent_audio.shape) > 1:
                agent_audio = agent_audio[:, 0]
        except Exception as e:
            print(f"[S2SSession] Error reading agent audio: {e}")
            return None, 0

        # Create 2-channel audio (zero-padded to max length)
        max_len = max(len(user_audio), len(agent_audio))
        stereo = np.zeros((max_len, 2), dtype=np.float32)
        stereo[: len(user_audio), 0] = user_audio
        stereo[: len(agent_audio), 1] = agent_audio

        # Normalize
        max_val = np.abs(stereo).max()
        if max_val > 0:
            stereo = stereo / max_val * 0.95

        # Encode to bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, stereo, output_sr, format="WAV")
        print(
            f"[S2SSession] Generated dual-channel audio: user={len(user_audio)} samples, agent={len(agent_audio)} samples"
        )
        return wav_buffer.getvalue(), output_sr

    def _get_session_artifacts_dir(self, session_id: str) -> Optional[str]:
        """Get or create the artifacts directory for a session."""
        if not self.inc_config.save_session_artifacts:
            return None

        base_dir = self.inc_config.session_artifacts_dir
        if base_dir is None:
            # Default to /tmp/s2s_sessions
            base_dir = "/tmp/s2s_sessions"

        session_dir = os.path.join(base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _save_session_artifacts(
        self,
        session_id: str,
        turn_idx: int,
        input_audio_path: str,
        request_info: Dict[str, Any],
        output_text: str,
        output_audio_bytes: Optional[bytes],
        debug_info: Dict[str, Any],
        generation_time_ms: float,
    ):
        """Save session artifacts (input/output) to disk."""
        session_dir = self._get_session_artifacts_dir(session_id)
        if session_dir is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        turn_prefix = f"turn{turn_idx:02d}_{timestamp}"

        # Save input audio
        input_audio_dest = os.path.join(session_dir, f"{turn_prefix}_input.wav")
        shutil.copy2(input_audio_path, input_audio_dest)

        # Save output audio
        output_audio_dest = None
        if output_audio_bytes:
            output_audio_dest = os.path.join(session_dir, f"{turn_prefix}_output.wav")
            with open(output_audio_dest, "wb") as f:
                f.write(output_audio_bytes)

        # Build input JSON (with path instead of base64)
        input_json = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "timestamp": timestamp,
            "request": {
                **request_info,
                "audio_path": input_audio_dest,
            },
        }

        # Build output JSON
        output_json = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "timestamp": timestamp,
            "text": output_text,
            "audio_path": output_audio_dest,
            "debug_info": debug_info,
            "generation_time_ms": generation_time_ms,
        }

        # Save JSON files
        input_json_path = os.path.join(session_dir, f"{turn_prefix}_input.json")
        output_json_path = os.path.join(session_dir, f"{turn_prefix}_output.json")

        with open(input_json_path, "w") as f:
            json.dump(input_json, f, indent=2)

        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=2)

        print(f"[S2SSession] Saved artifacts to {session_dir}")

        return {
            "session_dir": session_dir,
            "input_audio_path": input_audio_dest,
            "output_audio_path": output_audio_dest,
            "input_json_path": input_json_path,
            "output_json_path": output_json_path,
        }

    @torch.no_grad()
    def inference_with_session(
        self,
        audio_path: str,
        session_state: Optional[SessionState] = None,
        num_frames_per_inference: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform inference with session state support.

        Args:
            audio_path: Path to input audio file
            session_state: Optional existing session state to restore from
            num_frames_per_inference: Frames to process per step

        Returns:
            Dict with 'text', 'asr_text', 'audio' outputs and 'session_state'
        """
        import librosa
        from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str

        if num_frames_per_inference is None:
            num_frames_per_inference = self.inc_config.num_frames_per_inference

        device = self.config.device
        buffer_size_frames = self.inc_config.buffer_size_frames
        buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES

        # Load audio for this turn
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Add silence padding for response
        if self.inc_config.silence_padding_sec > 0:
            silence_samples = int(self.inc_config.silence_padding_sec * SAMPLE_RATE)
            audio_signal = np.concatenate([audio_signal, np.zeros(silence_samples)])

        total_samples = len(audio_signal)

        # Calculate frames for this turn's audio
        total_frames_maybe = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES))
        num_inference_steps = total_frames_maybe // num_frames_per_inference
        if total_frames_maybe % num_frames_per_inference != 0:
            num_inference_steps += 1
        turn_frames = num_inference_steps * num_frames_per_inference

        # Pad audio to align with frames
        padded_samples = num_inference_steps * num_frames_per_inference * FRAME_SIZE_SAMPLES
        if padded_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_samples - total_samples))

        audio_tensor = torch.tensor(audio_signal, dtype=torch.float32, device=device).unsqueeze(0)

        # Initialize or restore state
        # Check if session has existing state (frame_idx > 0 indicates prior turns)
        has_existing_state = session_state is not None and session_state.frame_idx > 0

        if has_existing_state:
            print(f"[S2SSession] Restoring state from session, frame_idx={session_state.frame_idx}")
            state = self._restore_state_from_session(session_state, turn_frames, device)
        else:
            print("[S2SSession] Initializing fresh state")
            if session_state is None:
                session_state = SessionState(session_id="temp")
            # For fresh state, we need to estimate total frames
            max_frames = 50000  # Large buffer
            state = self._initialize_fresh_state(max_frames, device)

        # Check cache support (Mamba models use input_embeds_history, others use DynamicCache)
        use_cache = state["llm_cache"] is not None
        if not use_cache:
            input_embeds_history = state.get("input_embeds_history") or []
            print(f"[S2SSession] Using input_embeds_history (Mamba mode), history length: {len(input_embeds_history)}")
        else:
            input_embeds_history = []
            print("[S2SSession] Using DynamicCache mode")

        # Get starting frame index
        start_frame_idx = state["frame_idx"]
        gen_text = state["gen_text"]
        gen_asr_text = state["gen_asr_text"]
        audio_buffer = state["audio_buffer"]
        buffer_fill_level = state["buffer_fill_level"]
        llm_cache = state["llm_cache"]

        # Initialize TTS state for this turn (fresh each turn)
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
                # Create subword_mask with same size as gen_text to avoid index errors
                subword_mask = torch.ones(1, gen_text.shape[1], device=device, dtype=torch.bool)

        audio_segments = []

        # Response detection config
        response_end_detection_mode = self.inc_config.response_end_detection_mode
        audio_energy_threshold = self.inc_config.audio_energy_threshold
        audio_energy_window_sec = self.inc_config.audio_energy_window_sec
        max_response_frames = int(self.inc_config.max_response_duration_sec / FRAME_SIZE_SEC)
        audio_energy_window_samples = int(audio_energy_window_sec * TTS_SAMPLE_RATE)

        # Per-frame alignment tracking (use same format as incremental backend)
        output_frame_alignment = self.inc_config.output_frame_alignment
        frame_alignment = self._init_frame_alignment() if output_frame_alignment else None
        pad_id = self._model.stt_model.text_pad_id

        # Phase 1: Process audio frames
        local_frame_idx = 0
        while local_frame_idx < turn_frames:
            global_frame_idx = start_frame_idx + local_frame_idx

            slice_start = local_frame_idx * FRAME_SIZE_SAMPLES
            slice_n_samples = num_frames_per_inference * FRAME_SIZE_SAMPLES
            new_audio = audio_tensor[:, slice_start : slice_start + slice_n_samples]

            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, new_audio, buffer_size_samples
            )

            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_inference=num_frames_per_inference,
                frame_idx=global_frame_idx,
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

            # Collect frame alignment info (same format as incremental backend)
            if frame_alignment is not None:
                self._append_frame_alignment(
                    frame_alignment=frame_alignment,
                    frame_idx=global_frame_idx,
                    phase="user_turn",
                    gen_text=gen_text,
                    gen_asr_text=gen_asr_text,
                    pad_id=pad_id,
                    is_tts_stop=False,
                )

            local_frame_idx += num_frames_per_inference

        # Phase 2: Feed silence until response completes (energy-based detection)
        silence_audio = torch.zeros(
            1, num_frames_per_inference * FRAME_SIZE_SAMPLES, dtype=torch.float32, device=device
        )

        audio_energy_response_started = False
        consecutive_low_energy_samples = 0
        response_frames = 0
        recent_audio_buffer = []
        pad_id = self._model.stt_model.text_pad_id
        response_started = False
        consecutive_pad_count = 0
        stop_reason = "max_duration"  # Default, will be updated if stopped earlier

        print(f"[S2SSession] Waiting for response (mode={response_end_detection_mode})...")

        while response_frames < max_response_frames:
            global_frame_idx = start_frame_idx + local_frame_idx

            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, silence_audio, buffer_size_samples
            )

            # Ensure gen_text/gen_asr_text have enough space
            if global_frame_idx >= gen_text.shape[1]:
                new_size = global_frame_idx + 1000
                new_gen_text = torch.full((1, new_size), pad_id, device=device, dtype=torch.long)
                new_gen_text[:, : gen_text.shape[1]] = gen_text
                gen_text = new_gen_text

                new_gen_asr_text = torch.full((1, new_size), pad_id, device=device, dtype=torch.long)
                new_gen_asr_text[:, : gen_asr_text.shape[1]] = gen_asr_text
                gen_asr_text = new_gen_asr_text

                if subword_mask is not None:
                    new_subword_mask = torch.ones(1, new_size, device=device, dtype=torch.bool)
                    new_subword_mask[:, : subword_mask.shape[1]] = subword_mask
                    subword_mask = new_subword_mask

            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_inference=num_frames_per_inference,
                frame_idx=global_frame_idx,
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
                    recent_audio_buffer.append(result["decoded_audio_new"])
                past_key_values = result["past_key_values"]
                code = result["code"]

            # Audio energy detection
            if decode_audio and recent_audio_buffer:
                recent_audio_cat = torch.cat(recent_audio_buffer, dim=-1)
                if recent_audio_cat.shape[-1] > audio_energy_window_samples:
                    recent_audio_cat = recent_audio_cat[..., -audio_energy_window_samples:]
                    recent_audio_buffer = [recent_audio_cat]

                rms = torch.sqrt(torch.mean(recent_audio_cat**2)).item()

                if rms > audio_energy_threshold:
                    audio_energy_response_started = True
                    consecutive_low_energy_samples = 0
                else:
                    if audio_energy_response_started:
                        consecutive_low_energy_samples += (
                            result["decoded_audio_new"].shape[-1] if result.get("decoded_audio_new") is not None else 0
                        )

            # Text EOS detection
            current_token = gen_text[0, global_frame_idx].item()
            if current_token != pad_id:
                response_started = True
                consecutive_pad_count = 0
            else:
                if response_started:
                    consecutive_pad_count += 1

            # Check stopping condition
            should_stop = False
            is_tts_stop = False
            if response_end_detection_mode == "audio_energy":
                if (
                    decode_audio
                    and audio_energy_response_started
                    and consecutive_low_energy_samples >= audio_energy_window_samples
                ):
                    should_stop = True
                    is_tts_stop = True
                    stop_reason = "audio_energy"
                    print("[S2SSession] Response completed (audio energy)")
            elif response_end_detection_mode == "eos":
                if response_started and consecutive_pad_count >= self.inc_config.eos_detection_window:
                    should_stop = True
                    is_tts_stop = True
                    stop_reason = "eos"
                    print("[S2SSession] Response completed (EOS)")

            # Collect frame alignment info for response phase (same format as incremental backend)
            if frame_alignment is not None:
                self._append_frame_alignment(
                    frame_alignment=frame_alignment,
                    frame_idx=global_frame_idx,
                    phase="agent_response",
                    gen_text=gen_text,
                    gen_asr_text=gen_asr_text,
                    pad_id=pad_id,
                    is_tts_stop=is_tts_stop,
                )

            local_frame_idx += num_frames_per_inference
            response_frames += num_frames_per_inference

            if should_stop:
                break

        if response_frames >= max_response_frames:
            stop_reason = "max_duration"
            print("[S2SSession] Response hit max duration")

        # Update frame index for next turn
        final_frame_idx = start_frame_idx + local_frame_idx

        # Prepare outputs
        total_frames = final_frame_idx

        # For current turn text output: decode only the new frames from this turn
        current_turn_frames = final_frame_idx - start_frame_idx
        gen_text_current_turn = gen_text[:, start_frame_idx:final_frame_idx]
        gen_asr_text_current_turn = gen_asr_text[:, start_frame_idx:final_frame_idx]
        current_turn_lengths = torch.tensor([current_turn_frames], dtype=torch.long, device=device)

        text_output = tokens_to_str(
            gen_text_current_turn,
            current_turn_lengths,
            tokenizer=self._tokenizer,
            pad_id=self._model.stt_model.text_pad_id,
            eval_text_turn_taking=True,
        )
        asr_text_output = tokens_to_str(
            gen_asr_text_current_turn,
            current_turn_lengths,
            tokenizer=self._tokenizer,
            pad_id=self._model.stt_model.text_pad_id,
            eval_text_turn_taking=True,
        )

        # Keep full trimmed tensors for session state (needed for next turn)
        gen_text_trimmed = gen_text[:, :total_frames]
        lengths = torch.tensor([total_frames], dtype=torch.long, device=device)

        output_audio = None
        if audio_segments:
            output_audio = torch.cat(audio_segments, dim=-1)

        # Save state back to session
        state["frame_idx"] = final_frame_idx
        state["gen_text"] = gen_text
        state["gen_asr_text"] = gen_asr_text
        state["audio_buffer"] = audio_buffer
        state["buffer_fill_level"] = buffer_fill_level
        state["llm_cache"] = llm_cache
        state["input_embeds_history"] = input_embeds_history

        self._save_state_to_session(session_state, state)

        debug_info = {
            "start_frame_idx": start_frame_idx,
            "final_frame_idx": final_frame_idx,
            "turn_frames": turn_frames,
            "response_frames": response_frames,
            "stop_reason": stop_reason,
            "audio_energy_response_started": audio_energy_response_started,
        }

        # Add frame alignment if enabled
        if frame_alignment:
            debug_info["frame_alignment"] = frame_alignment

        return {
            "text": text_output,
            "asr_text": asr_text_output,
            "audio": output_audio,
            "tokens_text": gen_text_trimmed,
            "tokens_len": lengths,
            "session_state": session_state,
            "debug_info": debug_info,
        }

    def generate_with_session(self, request: GenerationRequest, session_state: Optional[SessionState] = None) -> tuple:
        """
        Generate with session support.

        Args:
            request: Generation request
            session_state: Optional session state to restore

        Returns:
            Tuple of (GenerationResult, updated SessionState)
        """
        if not self._is_loaded:
            return (
                GenerationResult(error="Model not loaded", request_id=request.request_id),
                session_state,
            )

        start_time = time.time()
        temp_files = []
        saved_input_audio_path = None

        try:
            # Get audio path
            audio_path = request.audio_path
            if request.audio_bytes:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(request.audio_bytes)
                temp_file.close()
                temp_files.append(temp_file.name)
                audio_path = temp_file.name

            if not audio_path:
                return (
                    GenerationResult(error="Audio input required", request_id=request.request_id),
                    session_state,
                )

            saved_input_audio_path = audio_path

            # Determine turn index from session state
            turn_idx = 0
            if session_state is not None and session_state.frame_idx > 0:
                # Estimate turn index from frame_idx (rough approximation)
                # Better: track turn count in session state
                turn_idx = getattr(session_state, "turn_count", 0)

            # Run inference with session
            output = self.inference_with_session(
                audio_path=audio_path,
                session_state=session_state,
                num_frames_per_inference=self.inc_config.num_frames_per_inference,
            )

            # Encode agent audio to bytes (single-channel for storage)
            agent_audio_bytes = None
            if output["audio"] is not None:
                audio_np = output["audio"].float().cpu().numpy().squeeze()
                max_val = np.abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / max_val * 0.95

                wav_buffer = io.BytesIO()
                import soundfile as sf

                sf.write(wav_buffer, audio_np, self.target_sample_rate, format="WAV")
                agent_audio_bytes = wav_buffer.getvalue()

            elapsed_ms = (time.time() - start_time) * 1000

            # Generate dual-channel audio (user=ch0, agent=ch1) for response
            response_audio_bytes = agent_audio_bytes
            response_sample_rate = self.target_sample_rate
            if saved_input_audio_path and agent_audio_bytes:
                dual_audio_bytes, dual_sr = self._generate_dual_channel_audio_for_turn(
                    saved_input_audio_path, agent_audio_bytes
                )
                if dual_audio_bytes:
                    response_audio_bytes = dual_audio_bytes
                    response_sample_rate = dual_sr

            updated_session = output["session_state"]
            session_id = updated_session.session_id if updated_session else "unknown"

            # Update turn count in session
            if updated_session is not None:
                if not hasattr(updated_session, "turn_count"):
                    updated_session.turn_count = 0
                updated_session.turn_count += 1
                turn_idx = updated_session.turn_count - 1

            # Save session artifacts
            debug_info = output.get("debug_info", {})
            output_text = output["text"][0] if output["text"] else ""

            # Read input audio bytes for session storage
            input_audio_bytes = None
            user_duration_sec = 0.0
            if saved_input_audio_path and os.path.exists(saved_input_audio_path):
                with open(saved_input_audio_path, "rb") as f:
                    input_audio_bytes = f.read()
                # Calculate duration from debug_info
                user_duration_sec = debug_info.get("turn_frames", 0) * FRAME_SIZE_SEC

            # Calculate agent audio duration
            agent_duration_sec = 0.0
            if agent_audio_bytes:
                agent_duration_sec = debug_info.get("response_frames", 0) * FRAME_SIZE_SEC

            # Store turn data in session (use single-channel agent audio for storage)
            if updated_session is not None:
                if not hasattr(updated_session, "turns") or updated_session.turns is None:
                    updated_session.turns = []
                turn_data = TurnData(
                    turn_idx=turn_idx,
                    user_audio_bytes=input_audio_bytes,
                    agent_audio_bytes=agent_audio_bytes,
                    agent_text=output_text,
                    user_duration_sec=user_duration_sec,
                    agent_duration_sec=agent_duration_sec,
                )
                updated_session.turns.append(turn_data)

            artifacts_info = self._save_session_artifacts(
                session_id=session_id,
                turn_idx=turn_idx,
                input_audio_path=saved_input_audio_path,
                request_info={
                    "request_id": request.request_id,
                    "text": request.text,
                    "user_prompt": request.user_prompt,
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                },
                output_text=output_text,
                output_audio_bytes=agent_audio_bytes,
                debug_info=debug_info,
                generation_time_ms=elapsed_ms,
            )

            # Add artifacts info to debug_info
            if artifacts_info:
                debug_info["artifacts"] = artifacts_info

            # Add total_frames to match incremental backend debug format
            debug_info["total_frames"] = debug_info.get("final_frame_idx", 0)

            # Add per-turn text responses to debug_info
            if updated_session is not None and updated_session.turns:
                debug_info["turn_texts"] = [t.agent_text for t in updated_session.turns]

            # Add ASR text (user speech transcription) to debug_info
            asr_text_output = output.get("asr_text", [""])[0] if output.get("asr_text") else ""
            debug_info["asr_text"] = asr_text_output

            result = GenerationResult(
                text=output_text,
                audio_bytes=response_audio_bytes,
                audio_sample_rate=response_sample_rate,
                request_id=request.request_id,
                generation_time_ms=elapsed_ms,
                debug_info=debug_info,
            )

            return (result, updated_session)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return (
                GenerationResult(error=str(e), request_id=request.request_id),
                session_state,
            )

        finally:
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Generate without session support (falls back to parent).

        For session support, use generate_with_session() instead.
        """
        return super().generate(requests)

    def generate_session_audio(
        self,
        session_state: SessionState,
        pause_between_turns_sec: float = 0.5,
    ) -> Optional[str]:
        """
        Generate a 2-channel WAV file for the entire session.

        Channel 0: User (speaker) audio
        Channel 1: Agent audio

        Both channels include pauses to align with the conversation flow.

        Args:
            session_state: The session state with turn data
            pause_between_turns_sec: Pause duration between turns (seconds)

        Returns:
            Path to the generated session audio file, or None if no turns
        """
        if not session_state.turns:
            print("[S2SSession] No turns to generate session audio")
            return None

        import soundfile as sf

        session_dir = self._get_session_artifacts_dir(session_state.session_id)
        if session_dir is None:
            print("[S2SSession] Session artifacts disabled, skipping session audio")
            return None

        os.makedirs(session_dir, exist_ok=True)

        # Target sample rate for output
        output_sr = self.target_sample_rate
        pause_samples = int(pause_between_turns_sec * output_sr)

        # Collect all audio segments with timing
        user_segments = []  # List of (start_sample, audio_array)
        agent_segments = []  # List of (start_sample, audio_array)

        current_sample = 0
        for turn in session_state.turns:
            # Process user audio
            if turn.user_audio_bytes:
                try:
                    user_audio, user_sr = sf.read(io.BytesIO(turn.user_audio_bytes))
                    # Resample if needed
                    if user_sr != output_sr:
                        # Simple resampling via linear interpolation
                        import scipy.signal

                        num_samples = int(len(user_audio) * output_sr / user_sr)
                        user_audio = scipy.signal.resample(user_audio, num_samples)
                    # Ensure mono
                    if len(user_audio.shape) > 1:
                        user_audio = user_audio[:, 0]
                    user_segments.append((current_sample, user_audio))
                    current_sample += len(user_audio)
                except Exception as e:
                    print(f"[S2SSession] Error reading user audio: {e}")

            # Add pause after user speaks
            current_sample += pause_samples

            # Process agent audio
            if turn.agent_audio_bytes:
                try:
                    agent_audio, agent_sr = sf.read(io.BytesIO(turn.agent_audio_bytes))
                    # Resample if needed
                    if agent_sr != output_sr:
                        import scipy.signal

                        num_samples = int(len(agent_audio) * output_sr / agent_sr)
                        agent_audio = scipy.signal.resample(agent_audio, num_samples)
                    # Ensure mono
                    if len(agent_audio.shape) > 1:
                        agent_audio = agent_audio[:, 0]
                    agent_segments.append((current_sample, agent_audio))
                    current_sample += len(agent_audio)
                except Exception as e:
                    print(f"[S2SSession] Error reading agent audio: {e}")

            # Add pause after agent speaks
            current_sample += pause_samples

        if not user_segments and not agent_segments:
            print("[S2SSession] No audio segments found")
            return None

        # Create 2-channel array
        total_samples = current_sample
        stereo_audio = np.zeros((total_samples, 2), dtype=np.float32)

        # Fill user channel (channel 0)
        for start_sample, audio in user_segments:
            end_sample = min(start_sample + len(audio), total_samples)
            stereo_audio[start_sample:end_sample, 0] = audio[: end_sample - start_sample]

        # Fill agent channel (channel 1)
        for start_sample, audio in agent_segments:
            end_sample = min(start_sample + len(audio), total_samples)
            stereo_audio[start_sample:end_sample, 1] = audio[: end_sample - start_sample]

        # Normalize
        max_val = np.abs(stereo_audio).max()
        if max_val > 0:
            stereo_audio = stereo_audio / max_val * 0.95

        # Save to file
        output_path = os.path.join(session_dir, "session_audio.wav")
        sf.write(output_path, stereo_audio, output_sr)

        duration_sec = total_samples / output_sr
        print(
            f"[S2SSession] Generated session audio: {output_path} "
            f"({duration_sec:.2f}s, {len(session_state.turns)} turns)"
        )

        return output_path

    def on_session_close(self, session_state: SessionState) -> Dict[str, Any]:
        """
        Called when a session is closed/deleted.

        Generates the final session audio and returns summary info.

        Args:
            session_state: The session being closed

        Returns:
            Dict with session summary info
        """
        result = {
            "session_id": session_state.session_id,
            "turn_count": len(session_state.turns) if session_state.turns else 0,
            "turn_texts": [t.agent_text for t in session_state.turns] if session_state.turns else [],
        }

        # Generate session audio
        audio_path = self.generate_session_audio(session_state)
        if audio_path:
            result["session_audio_path"] = audio_path

        return result

    def warmup(self):
        """
        Run a warmup inference to pre-compile Triton kernels.

        This prevents race conditions when multiple requests arrive simultaneously
        before kernels are compiled.
        """
        import tempfile

        import soundfile as sf

        print("[S2SSession] Running warmup inference...")

        # Create a short silence audio for warmup (0.5 seconds)
        warmup_duration_sec = 0.5
        warmup_samples = int(warmup_duration_sec * SAMPLE_RATE)
        warmup_audio = np.zeros(warmup_samples, dtype=np.float32)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, warmup_audio, SAMPLE_RATE)
            warmup_path = f.name

        try:
            # Run inference with minimal settings
            from ..session_manager import SessionState

            session_state = SessionState(session_id="warmup")
            _ = self.inference_with_session(
                audio_path=warmup_path,
                session_state=session_state,
                num_frames_per_inference=self.inc_config.num_frames_per_inference,
            )
            print("[S2SSession] Warmup complete")
        finally:
            if os.path.exists(warmup_path):
                os.unlink(warmup_path)
