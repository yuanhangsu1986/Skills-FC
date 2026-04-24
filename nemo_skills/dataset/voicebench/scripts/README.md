# VoiceBench Evaluation (FC Checkpoint)

Evaluation of the Function Calling (FC) S2S checkpoint on VoiceBench. Uses the `s2s_incremental_v2` backend with speech output.

**Checkpoint**: Apr 2 2026 FC — `e70-step28008-tts-eartts-34014_asr_cand3_0.6b-PK_ep0_01988_300`

## Configs

| Config | Subtests | Output dir |
|--------|----------|------------|
| `vb_matched_demo_v2_02mar_config_fc.yaml` | sd_qa, alpacaeval_full, alpacaeval, ifeval, advbench, commoneval, wildvoice, alpacaeval_speaker | `.../voice_chat/vb` |
| `vb_matched_demo_v2_02mar_mcq_config_fc.yaml` | bbh, openbookqa, mmsu (MCQ with system prompt) | `.../voice_chat/vb_mcq` |

## Run

From repo root on the login node:

```bash
# Open-ended subtests
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 nohup python -u \
  nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc.yaml \
  > vb_fc_run.log 2>&1 &
echo "PID: $!"

# MCQ subtests
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 nohup python -u \
  nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_mcq_config_fc.yaml \
  > vb_fc_mcq_run.log 2>&1 &
echo "PID: $!"
```

Monitor logs:

```bash
tail -f vb_fc_run.log
```

## Re-score LLM-as-judge subtests

If scoring failed due to a missing/invalid `NV_INFERENCE_KEY`, re-run scoring only for the subtests that use GPT-4o-mini as judge (`sd_qa`, `alpacaeval`, `alpacaeval_full`, `commoneval`, `wildvoice`, `alpacaeval_speaker`). MCQ subtests (`bbh`, `openbookqa`, `mmsu`) and `ifeval`/`advbench` use deterministic scoring and don't need this.

```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 nohup python -u \
  nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc.yaml \
  --scoring_only \
  --scoring_force \
  --subtests sd_qa,alpacaeval_full,alpacaeval,commoneval,wildvoice,alpacaeval_speaker \
  > vb_fc_rescore.log 2>&1 &
echo "PID: $!"
```

Requires `NV_INFERENCE_KEY` to be set — see [build.nvidia.com](https://build.nvidia.com) or ask your team for a shared key.

## Output Structure

```
/lustre/.../voice_chat/vb/eval-results/voicebench.<subtest>/
├── output.jsonl                  # model generations
├── output.jsonl.done             # completion marker
├── output_asr.jsonl              # ASR transcriptions
├── voicebench_format.jsonl       # converted for VoiceBench scorer
├── result-voicebench_format.jsonl
├── metrics.json                  # final scores
└── summarized-results/           # job logs
```

Intermediate server artifacts (audio, per-request outputs) are written to `vb_artifacts/` and `vb_mcq_artifacts/`.

## Check Progress

```bash
BASE="/lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/vb/eval-results"
for subtest in sd_qa alpacaeval_full alpacaeval ifeval advbench commoneval wildvoice alpacaeval_speaker; do
    dir="$BASE/voicebench.$subtest"
    [ -f "$dir/metrics.json" ] && echo "$subtest: DONE" && cat "$dir/metrics.json" || echo "$subtest: pending"
done
```

## Cleanup

Delete all output files for a full reset:

```bash
rm -rf /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/vb/eval-results/ \
       /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/vb_artifacts/ \
       /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/vb_mcq/eval-results/ \
       /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/vb_mcq_artifacts/ \
       /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/code/nemo-run-fc/ \
       vb_fc_run.log vb_fc_mcq_run.log
```

- `vb/eval-results/` and `vb_mcq/eval-results/` — subtest outputs and scores
- `vb_artifacts/` and `vb_mcq_artifacts/` — intermediate server artifacts
- `nemo-run-fc/` — nemo_run job metadata, packaged code, sbatch files (grows over time)

## Related Docs

- [VOICEBENCH_S2S_EVAL.md](VOICEBENCH_S2S_EVAL.md) — S2S offline backend evaluation details
- [S2S_VOICECHAT_BACKEND.md](S2S_VOICECHAT_BACKEND.md) — `s2s_voicechat` backend reference
