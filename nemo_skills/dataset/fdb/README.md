# Full-Duplex-Bench (FDB) for nemo-skills

This package prepares the [Full-Duplex-Bench](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) dataset for speech-to-speech evaluation in nemo-skills.

## Dataset versions

- **v1.0** — writes to `fdb_v1/` (subtests: pause_candor, pause_synthetic, backchannel, turn_taking, interruption).
- **v1.5** — writes to `fdb_v1_5/` (subtests: background_speech, talking_to_other, backchannel, interruption).

Source layout (v1.0): `candor_pause_handling/{id}/input.wav`, `synthetic_pause_handling/{id}/input.wav`, `icc_backchannel/`, `candor_turn_taking/`, `synthetic_user_interruption/`.

## Prepare data

From the repo root (or with `PYTHONPATH` set so `nemo_skills` is importable):

```bash
# Download v1.0 to default path and prepare
python -m nemo_skills.dataset.fdb.prepare

# Use existing Full-Duplex-Bench data (no download)
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path /path/to/Full-Duplex-Bench-data

# Prepare v1.5 only
python -m nemo_skills.dataset.fdb.prepare --version v1.5 --fdb_data_path /path/to/Full-Duplex-Bench-data

# Skip copying audio (faster, for testing)
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path /path/to/data --no-audio
```

Audio files are copied into `fdb_v1/data/` (or `fdb_v1_5/data/`) with names like `pause_candor_18.wav`, `pause_synthetic_18.wav`, `backchannel_0.wav`, etc., so each subtest has distinct filenames.

## Output layout

After prepare:

- `fdb_v1/<subtest>/test.jsonl` — one JSONL per subtest (e.g. pause_candor, pause_synthetic).
- `fdb_v1/data/*.wav` — shared audio directory; entries in JSONL reference `fdb_v1/data/<audio_id>.wav`.

Each entry includes `messages`, `messages_text_audio`, and `messages_text` variants.

## Evaluation

Set `data_dir` in your eval config to the **fdb package directory** (the parent of `fdb_v1`/`fdb_v1_5`), e.g.:

```yaml
data_dir: /path/to/nemo_skills/dataset/fdb
```

### Running evaluation

Use `scripts/run_eval.py` with a YAML config:

```bash
# Full run (generate + score) for v1.0
python scripts/run_eval.py --config scripts/fdb_s2s_offline_v1.0_config.yaml

# Full run for v1.5
python scripts/run_eval.py --config scripts/fdb_s2s_offline_v1.5_config.yaml

# Score only (skip generation, re-run scoring on existing outputs)
python scripts/run_eval.py --config scripts/fdb_s2s_offline_v1.5_config.yaml \
  --scoring_only --scoring_force

# Score a specific subtest
python scripts/run_eval.py --config scripts/fdb_s2s_offline_v1.5_config.yaml \
  --scoring_only --scoring_force --subtests backchannel

# Override output directory
python scripts/run_eval.py --config scripts/fdb_s2s_offline_v1.0_config.yaml \
  --output_dir /path/to/output
```

Key config fields (see example YAML files):

| Field | Description |
|---|---|
| `cluster` | Cluster config name (e.g. `s2s_eval_oci_iad`) |
| `server_gpus` | GPUs for the inference server |
| `num_chunks` | Number of parallel inference chunks |
| `fdb_version` | `v1.0` or `v1.5` |
| `data_dir` | Path to the FDB nemo-skills package (parent of `fdb_v1`/`fdb_v1_5`) |
| `output_dir` | Where to write results |
| `fdb_repo_path` | Path to Full-Duplex-Bench clone (for ASR + evaluation scripts) |
| `fdb_data_path` | Path to raw FDB dataset (needed for turn_taking, interruption, behavior subtests) |
| `subtests` | List of subtests to run |

### Helper scripts

- `scripts/run_prepare_and_eval_both.sh` — prepare v1.0 and v1.5 then run offline eval for both (intended for cluster use with lustre paths).

## Source data

Download from Google Drive:  
https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3

If you pass `--fdb_data_path` and the requested version (v1.0 or v1.5) is missing, the script can attempt to download and extract it (requires `gdown`).

## output.wav pipeline (2-channel)

FDB expects **2-channel (stereo)** `output.wav`. Scoring uses the [Full-Duplex-Bench-NV](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV) repo (or a clone) for ASR and evaluation; see `STEREO_REFERENCE_EVIDENCE.md` for where 2-channel is created vs consumed in that repo and in this codebase.

1. **Server** (`s2s_voicechat` backend in `recipes/multimodal/server/backends/s2s_voicechat_infer_backend.py`) runs `offline_inference(..., decode_audio=True)`. The backend **outputs 2-channel WAV**: it duplicates the model’s mono output to both channels so the bytes sent in the API response are already stereo.
2. **Client** (`vllm_multimodal._process_audio_response`) decodes and writes those bytes to `output_dir/audio/<id>.wav` as-is (no channel change).
3. **Prepare** (`prepare_fdb_eval_dir`) copies that file to `fdb_prepared/<id>/output.wav` and **verifies** it is 2-channel; if mono, the script fails with a clear error.

So when using the **s2s_voicechat** backend with `--decode_audio`, the server already returns stereo; the prepare step only verifies and copies.

### Scoring consistency (stereo + backchannel)

The pipeline is **prepare (stereo output.wav) → ASR (--stereo, ch1) → evaluate**. For consistency:

- **fdb_repo_path** in your config must point to a Full-Duplex-Bench clone that has:
  1. **get_transcript/asr.py** with `--stereo` (uses channel 1 for transcription).
  2. **evaluation/eval_backchannel.py** that converts stereo to the model channel (ch1) before calling Silero VAD (Silero expects 1D audio).

The config `fdb_s2s_offline_v1.0_config.yaml` uses **our** Full-Duplex-Bench clone (`.../Full-Duplex-Bench`) for scoring so backchannel runs with the stereo-safe eval. On the cluster, ensure that clone has the updated `eval_backchannel.py` (and `asr.py` with `--stereo`) deployed.