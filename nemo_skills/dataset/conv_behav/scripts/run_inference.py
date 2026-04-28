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
Offline inference for conv_behav evaluation.

Uses NemotronVoiceChat via RealtimeStreamingInference from vtrinh's NeMo2
for per-frame autoregressive streaming, which is required for turn-taking
timing metrics (barge-in detection, latency, etc.).

Produces under <output_dir>/:
  validation_logs/pred_wavs/{dataset_name}_{recording_id}.wav
  validation_logs/metadatas/{dataset_name}.json

Usage:
    python run_inference.py \
        --model_path /path/to/vtrinh_combined_ckpt \
        --llm_checkpoint_path /path/to/vtrinh_combined_ckpt \
        --shar_input_dir /path/to/lhotse_shar \
        --output_dir /path/to/output \
        --nemo_code_path /path/to/vtrinh/NeMo2 \
        --speaker_reference /path/to/speaker_ref.wav \
        [--dataset_name team_20251124] \
        [--num_frames_per_inference 3] \
        [--buffer_size_frames 21] \
        [--codec_token_history_size 60] \
        [extra inference overrides, e.g. --force_turn_taking false --temperature 0.8]
"""

import argparse
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import soundfile as sf
import torchaudio
from omegaconf import OmegaConf


_INFER_SCRIPT_REL = (
    "scripts/training/iad/s2s/sdv2_hf/conv/nano_9b/eartts/"
    "inference_streaming_realtime_niva_json.py"
)
_TTS_SAMPLE_RATE = 22050


def _import_realtime_engine(nemo_code_path: str):
    """Import RealtimeStreamingInference from vtrinh's NeMo2.

    Pre-importing nemo from nemo_code_path fills sys.modules before the
    script's own sys.path.insert (hardcoded to cchen1's path) runs, so
    vtrinh's NeMo2 modules are used throughout.
    """
    if nemo_code_path not in sys.path:
        sys.path.insert(0, nemo_code_path)

    import nemo  # noqa: F401 — populates sys.modules from nemo_code_path
    import nemo.collections.speechlm2  # noqa: F401

    script_path = os.path.join(nemo_code_path, _INFER_SCRIPT_REL)
    spec = importlib.util.spec_from_file_location(
        "inference_streaming_realtime_niva_json", script_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RealtimeStreamingInference


def _iter_shar_recordings(shar_dir: str):
    """Yield (recording_id, audio_bytes) from all recording.*.tar files."""
    shar_path = Path(shar_dir)
    tar_files = sorted(shar_path.glob("recording.*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No recording.*.tar files in {shar_dir}")
    for tar_file in tar_files:
        with tarfile.open(tar_file) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".flac"):
                    f = tf.extractfile(member)
                    if f is not None:
                        yield Path(member.name).stem, f.read()


def _to_mono_wav(audio_bytes: bytes, target_sr: int = 16000) -> str:
    """Write flac bytes to a temp mono WAV at target_sr. Caller must unlink."""
    tmp_flac = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
    tmp_flac.write(audio_bytes)
    tmp_flac.close()

    waveform, sr = torchaudio.load(tmp_flac.name)
    os.unlink(tmp_flac.name)

    if waveform.shape[0] > 1:
        waveform = waveform[0:1]  # channel 0 = user audio
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp_wav.name, waveform, target_sr)
    tmp_wav.close()
    return tmp_wav.name


def _apply_overrides(cfg, extra_args: list) -> None:
    """Apply --key [value] pairs from extra_args as flat OmegaConf overrides."""
    def _coerce(v: str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token.lstrip("-")
        if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
            value = _coerce(extra_args[i + 1])
            i += 2
        else:
            value = True
            i += 1
        OmegaConf.update(cfg, key, value, merge=True)
        print(f"[inference] model.cfg override: {key} = {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="conv_behav inference via NemotronVoiceChat (RealtimeStreamingInference)"
    )
    parser.add_argument("--model_path", required=True,
                        help="vtrinh combined checkpoint (HF format); provides TTS weights")
    parser.add_argument("--llm_checkpoint_path", required=True,
                        help="Same combined checkpoint; provides LLM+perception weights")
    parser.add_argument("--shar_input_dir", required=True,
                        help="Lhotse shar directory (recording.*.tar + cuts.*.jsonl.gz)")
    parser.add_argument("--output_dir", required=True,
                        help="Root output dir; validation_logs/ written here")
    parser.add_argument("--nemo_code_path", required=True,
                        help="vtrinh NeMo2 root (…/code/eval_turn_taking/NeMo2)")
    parser.add_argument("--speaker_reference", required=True,
                        help="Speaker reference WAV for TTS voice cloning")
    parser.add_argument("--dataset_name", default="conv_behav")
    parser.add_argument("--num_frames_per_inference", type=int, default=3)
    parser.add_argument("--buffer_size_frames", type=int, default=21)
    parser.add_argument("--codec_token_history_size", type=int, default=60)
    # All remaining --key [value] pairs are applied as OmegaConf overrides on
    # engine.model.cfg after the engine is initialized (e.g. --force_turn_taking
    # false, --temperature 0.8, --inference_guidance_scale 0.2, …).
    args, extra_args = parser.parse_known_args()

    RealtimeStreamingInference = _import_realtime_engine(args.nemo_code_path)

    model_cfg = OmegaConf.create({
        "model_path": args.model_path,
        "llm_checkpoint_path": args.llm_checkpoint_path,
        "speaker_reference": args.speaker_reference,
        "buffer_size_frames": args.buffer_size_frames,
        "codec_token_history_size": args.codec_token_history_size,
        "decode_audio": True,
        "compute_dtype": "bfloat16",
    })

    engine = RealtimeStreamingInference(model_cfg=model_cfg)

    if extra_args:
        _apply_overrides(engine.model.cfg, extra_args)

    pred_wavs_dir = Path(args.output_dir) / "validation_logs" / "pred_wavs"
    metadata_dir = Path(args.output_dir) / "validation_logs" / "metadatas"
    pred_wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{args.dataset_name}.json"

    tts_sr = getattr(engine, "target_sample_rate", _TTS_SAMPLE_RATE)

    n_done = 0
    n_failed = 0
    with open(metadata_path, "w") as meta_out:
        for recording_id, audio_bytes in _iter_shar_recordings(args.shar_input_dir):
            tmp_wav = None
            try:
                tmp_wav = _to_mono_wav(audio_bytes)

                results = engine.inference_realtime_streaming(
                    tmp_wav,
                    num_frames_per_inference=args.num_frames_per_inference,
                    explore_padding=False,
                )

                wav_out = pred_wavs_dir / f"{args.dataset_name}_{recording_id}.wav"
                audio = results.get("audio")
                if audio is not None:
                    sf.write(str(wav_out), audio.float().cpu().numpy(), tts_sr)

                meta_entry = {
                    "audio_path": recording_id,
                    "pred_text": results["text"][0],
                    "pred_src_text": results["asr_text"][0],
                }
                meta_out.write(json.dumps(meta_entry) + "\n")
                meta_out.flush()
                n_done += 1
                print(f"[inference] {n_done} done: {recording_id}")

            except Exception as e:
                print(f"[inference] ERROR {recording_id}: {e}", file=sys.stderr)
                n_failed += 1
            finally:
                if tmp_wav and os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)

    print(f"[inference] Done. {n_done} ok, {n_failed} failed.")
    if n_failed > 0 and (n_done + n_failed) > 0:
        if n_failed / (n_done + n_failed) > 0.1:
            print(
                f"[inference] ERROR: failure rate too high "
                f"({n_failed}/{n_done + n_failed})",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
