# FDB v3 (FD3) — direct function-head tool-call evaluation

This module evaluates whether an S2S checkpoint can directly emit tool calls
from its function channel for FD3 audio prompts. It mirrors `cchen1`'s
`FD3_edge` pipeline, integrated as a single nemo-skills benchmark
`fdb_v3.tool_call`.

## Decisions baked in

| Item | Choice |
| --- | --- |
| Subtest granularity | Single benchmark `fdb_v3.tool_call`. Per-domain / per-difficulty breakdown comes from `evaluate_tool_calls.py` (4 domains: ecommerce, finance, housing, travel). |
| FD3 source | Vendored at `/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/NeMo/FDBV3_CHENCHEN`, referenced via `fdb_repo_path`. |
| LLM judge | `openai/openai/gpt-5.2` via NVIDIA inference API; auth uses `NV_INFERENCE_KEY` from cluster env. |
| Engine | `vllm_llm_vllm_eartts` (existing v2 backend already patches `nemotron_h.py`). |

## Files

```
dataset/fdb/
  fdb_v3/                           # benchmark registration only
    __init__.py
    tool_call/__init__.py
  scripts/fdb_v3/
    prepare.py                      # FD3 dataset -> dataset/fdb/fdb_v3/tool_call/test.jsonl
    render_prompt.py                # FD3 system prompt (Nemotron Jinja + FD3_TOOL_SPEC)
    run_scoring.py                  # output.jsonl -> FD3 layout -> evaluators -> metrics.json
    run_eval.py                     # nemo-skills generation + scoring orchestrator
    fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml
    README.md
```

## Usage

```bash
# 1) Prepare (one-time, on any node with /lustre access).
#    Copies input.wav into the dataset dir, renders FD3 system prompt, writes test.jsonl.
python nemo_skills/dataset/fdb/scripts/fdb_v3/prepare.py

# 2) Run eval (generation + scoring submitted as Slurm jobs).
python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml

# 3) Re-score only (after fixing the scorer).
python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml \
    --scoring_only --scoring_force
```

## Where outputs land

```
<output_dir>/eval-results/fdb_v3.tool_call/
  output.jsonl                                 # nemo-skills generation output
  fdb_v3_layout/                               # reconstructed FD3 layout
    {example_id}_{speaker_id}/
      result_<provider>.json                   # FD3-shaped per-sample result
      output_<provider>.wav                    # generated agent audio (when available)
  summarized-results/
    <provider>_eval.json                       # tool selection / argument acc / response qual + by_domain
    <provider>_pass_rate.json
    <provider>_latency.json
  metrics.json                                 # merged headline keyed by `fdb_v3.tool_call`
```
