# VoiceBench S2S Offline Evaluation

## Quick Start

Run full VoiceBench evaluation with S2S offline backend:

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-s2s-eval
source .venv/bin/activate
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py \
    --config nemo_skills/dataset/voicebench/scripts/voicebench_s2s_offline_config.yaml
```

## Backend Overview

The `s2s` backend (`recipes/multimodal/server/backends/s2s_backend.py`) provides offline Speech-to-Speech inference:

- **Model**: NemotronVoiceChat (STT + LLM + TTS combined)
- **Input**: Audio files (WAV)
- **Output**: Text response (audio is controlled by backend; `s2s_voicechat` supports `--decode_audio`)
- **Batching**: Server request batcher supports `--batch_size` (configured in `serve_unified.py`)
- **Key parameter**: `extra_decoding_seconds=20` - additional audio generation time per sample

## Configuration

Main config: `voicebench_s2s_offline_config.yaml`

Key settings:
- `cluster`: oci_iad
- `num_chunks`: 32-48 (parallel jobs)
- `server_args`: Backend-specific parameters including model paths, speaker reference, TTS checkpoint
- `subtests`: bbh, alpacaeval, ifeval, openbookqa, advbench, commoneval, wildvoice, mmsu, sd_qa, alpacaeval_speaker

## Retry Failed Chunks

If chunks fail (usually due to port collision), create a retry config pointing to the same `output_dir`:

```yaml
output_dir: /lustre/.../runs/voicebench_YYYYMMDD_HHMMSS  # Same as original
subtests:
  - <failed_subtest>
```

NeMo Skills automatically skips completed chunks and only runs missing ones.

## Check Progress

```bash
ssh draco-oci-login-02.draco-oci-iad.nvidia.com '
BASE="/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_s2s_offline/runs/<run_dir>/eval-results"
for subtest in bbh alpacaeval ifeval openbookqa advbench commoneval wildvoice mmsu sd_qa alpacaeval_speaker; do
    dir="$BASE/voicebench.$subtest"
    [ -f "$dir/metrics.json" ] && echo "$subtest: SCORED" && cat "$dir/metrics.json"
done
'
```

## Known Issues

1. **Port collision**: Jobs may fail with "address already in use" when multiple jobs land on same node. Solution: retry failed chunks.
2. **Slow throughput**: ~0.4 samples/min per job due to `extra_decoding_seconds=20` (audio generation is near-real-time).
3. **Special tokens in output**: Model outputs `<$X.XX$>` and `<|X.XX|>` timing tokens. These are cleaned in `convert_to_voicebench_format.py`.
