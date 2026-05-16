# `s2s_voicechat` backend (NemotronVoiceChat offline inference)

## What this backend is

`s2s_voicechat` is a unified-server backend that mirrors the behavior of NeMo’s `nemotron_voicechat_infer.py` style offline inference:

- Loads a **single OmegaConf YAML** via `--config_path`
- Applies **script-like overrides** (S2S ckpt, TTS ckpt, speaker reference, boosts, extra decoding)
- Instantiates `NemotronVoiceChat` from the resolved config
- Runs `offline_inference(...)` for each request (supports **batched** inference)

Default output is **text-only**. Audio output can be enabled with `--decode_audio`.

Backend implementation: `recipes/multimodal/server/backends/s2s_voicechat_infer_backend.py`

## What we changed recently (VoiceBench S2S eval)

- **Sound-enabled full evaluation config**:
  - `nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_offline_sound_config.yaml`
  - Adds `--decode_audio` and runs the full VoiceBench pipeline on all subsets (48 chunks).
- **Server batching knob**:
  - We pass `--batch_size 4` to `serve_unified` in the sound config to reduce request batching pressure.
- **4-stage scoring output** (single `metrics.json` per subset):
  - Stage 1: generation → `output.jsonl` (+ audio saved under `eval-results/voicebench.<subtest>/audio/`)
  - Stage 2: agent-audio ASR/WER/CER → `output_asr.jsonl` + `agent_audio_metrics.json`
  - Stage 3: VoiceBench scoring on generated text → `metrics.json` keys like `gpt`, `panda`, `acc`, `final`, ...
  - Stage 4: VoiceBench scoring on ASR text → same `metrics.json`, but with `*_asr` suffix (e.g. `gpt_asr`, `panda_asr`, `acc_asr`, `final_asr`)

## How it differs from `s2s_backend`

`s2s_backend` (`--backend s2s`) is also NemotronVoiceChat-based, but it is more “nemo-skills shaped” and can build/patch a minimal config.

Key differences:

- **Config loading**
  - `s2s_voicechat`: requires `--config_path` and uses it as the source of truth (closest to Kevin’s recipe).
  - `s2s_backend`: can run with a minimal generated config; optional `config_path` is converted to a Python dict and then patched.

- **Audio output**
  - `s2s_voicechat`: supports **AUDIO_OUT** and can return audio when `--decode_audio` is set.
  - `s2s_backend`: text output only (its `decode_audio` is `False` and not exposed as a server flag).

- **Artifact saving**
  - `s2s_voicechat`: optional per-request artifacts when `--save_artifacts --output_dir ...` are set.
    - Writes under: `<output_dir>/artifacts/<SLURM_JOB_ID|JOB_ID|local>/<request_id>/`
    - Files: `input.wav`, `output.json`, and (if audio enabled) `output.wav`
  - `s2s_backend`: no artifact writer.

- **TTS override semantics (important)**
  - `s2s_voicechat` matches NeMo `DuplexEARTTS` semantics:
    - If `--tts_ckpt_path` is a **file** (e.g. `.ckpt`), it sets `model.speech_generation.model.pretrained_model`
    - If `--tts_ckpt_path` is a **directory** (exported model), it sets `model.speech_generation.model.pretrained_tts_model`
  - `s2s_backend` always sets `pretrained_model` when `tts_ckpt_path` is provided.

## CLI / server usage

`s2s_voicechat` is served via `nemo_skills.inference.server.serve_unified`:

```bash
python -m nemo_skills.inference.server.serve_unified \
  --model /path/to/s2s_stt_ckpt.ckpt \
  --num_gpus 1 \
  --port 8000 \
  --backend s2s_voicechat \
  --batch_size 4 \
  --config_path /path/to/nemotron_voicechat_omegaconf.yaml \
  --code_path /path/to/NeMo/source/tree \
  --speaker_reference /path/to/speaker_ref.wav \
  --tts_ckpt_path /path/to/tts.ckpt \
  --extra_decoding_seconds 20 \
  --dtype float32 \
  --output_dir /path/to/output_root \
  --save_artifacts

# Add `--decode_audio` to enable audio output.
```

Notes:

- `--code_path` must point at a NeMo tree that contains `nemo.collections.speechlm2...` with `NemotronVoiceChat`.
- `--model` is used as `model.stt.model.pretrained_s2s_model` (script-like override).
- `--decode_audio` is **off by default** (text-only).

## VoiceBench configs in this repo

- **Full VoiceBench (all subsets, 48 chunks)**:
  - `nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_offline_config.yaml`
  - Text-only by default (the config comment shows how to enable audio).

- **Full VoiceBench (all subsets, 48 chunks, with audio + 4-stage scoring)**:
  - `nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_offline_sound_config.yaml`

- **Smoke test (sd_qa, 10 samples, with audio)**:
  - `nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_sdqa_smoke10_sound.yaml`

## Commands to run VoiceBench evaluation

From repo root (ensure venv is active):

```bash
source .venv/bin/activate

# If you hit permission issues writing logs under $HOME, set HOME to a writable dir:
export HOME="$PWD/.tmp_home"
mkdir -p "$HOME"

python nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_offline_config.yaml \
  --output_dir /lustre/.../runs/voicebench_$(date +%Y%m%d_%H%M%S)
```

## Exact commands we ran (copy/paste)

Smoke test (sd_qa, 10 samples, audio enabled):

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-s2s-eval && bash -lc 'set -euo pipefail; source .venv/bin/activate; export HOME="/home/vmendelev/workspace/expressiveness/src/nemo-skills-s2s-eval/.tmp_home"; mkdir -p "$HOME"; NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py --config nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_sdqa_smoke10_sound.yaml --output_dir /lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_s2s_voicechat_smoke/runs/sdqa_smoke10_sound_postcommit_20260205_034004'
```

Full run (all VoiceBench subsets, 48 chunks, audio enabled, batch size 4):

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-s2s-eval && bash -lc 'set -euo pipefail; source .venv/bin/activate; export HOME="/home/vmendelev/workspace/expressiveness/src/nemo-skills-s2s-eval/.tmp_home"; mkdir -p "$HOME"; NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py --config nemo_skills/dataset/voicebench/scripts/voicebench_s2s_voicechat_offline_sound_config.yaml --output_dir /lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_s2s_voicechat_offline/runs/voicebench_sound_bs4_20260205_140917'
```

