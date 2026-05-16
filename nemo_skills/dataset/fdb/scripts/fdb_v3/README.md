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
| System prompt | **Nano v2 full-schema** (rendered by `render_prompt.py` from `vtrinh/.../template.jinja` over `FD3_TOOL_SPEC`). See [Prompt choice](#prompt-choice) for why and how to switch. |
| Output transcript | Parakeet ASR (`nvidia/parakeet-tdt-0.6b-v2`) on the generated `output_<provider>.wav`, matching the original `run_s2s_offline_benchmark.py` pipeline. Requires `scoring_gpus >= 1`. Falls back to the cleaned S2S text channel only when ASR returns no text. |

## Prompt choice

The original FD3 paper baselines use a **lightweight** system prompt — only
each tool's `name` and `description`, ending with "You can also directly
respond to the user." (see `FD3/release_code/run_s2s_offline_benchmark.py:204-217`
and `FD3/release_code/README.md`).

This Skills-FC pipeline deliberately uses a **different default**: the Nemotron
Nano v2 voicechat tool-calling format (full JSON schema with parameters, plus
explicit `<TOOLCALL>` / `<TOOL_RESPONSE>` framing), rendered by
`render_prompt.py` from `vtrinh/projects/function_calling_share/script/template.jinja`.

Implications:

- **Recommended default for paper-comparable numbers**: the original
  lightweight FD3 prompt. Use it whenever you need to compare against
  published FD3 baselines or against other agents that ran under that prompt.
- **Why we diverge here**: the Jinja template self-identifies as "adapted from
  the full `nano_v2_chat_template.jinja` released with Nemotron Nano v2." It
  matches the system-prompt shape the Nemotron Nano v2 LLM backbone expects
  for tool-calling. We have not separately verified which prompt shape the
  S2S checkpoints' FC fine-tuning data was formatted with, so this is the
  best-effort match to the LLM backbone's native format rather than a
  confirmed training-time prompt.
- **Numbers under the two prompts are not directly comparable.** Switch one
  prompt for the other and tool_selection / argument_acc can move several
  points independent of any model change.

To switch to the original FD3 lightweight prompt, edit `render_prompt.py` to
build the string from `FD3_TOOL_NAME_DESCRIPTION_SPEC` (name + description
only) and emit the EVA-style trailer, or point `--template_path` (in
`prepare.py`) at an alternative Jinja template that renders the lightweight
format. Re-run `prepare.py` after any change — the rendered prompt is baked
into `test.jsonl`.

## Files

```
dataset/fdb/
  fdb_v3/                           # benchmark registration only
    __init__.py
    tool_call/__init__.py
  scripts/fdb_v3/
    prepare.py                      # FD3 dataset -> $data_dir/fdb_v3/tool_call/test.jsonl
    render_prompt.py                # FD3 system prompt (Nemotron Jinja + FD3_TOOL_SPEC)
    run_scoring.py                  # output.jsonl -> FD3 layout -> evaluators -> metrics.json
    run_eval.py                     # nemo-skills generation + scoring orchestrator
    fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml
    README.md
```

## Usage

```bash
# 1) Prepare (one-time, on any node with /lustre access).
#    Copies input.wav under $data_dir/fdb_v3/data/, renders FD3 system prompt,
#    writes $data_dir/fdb_v3/tool_call/test.jsonl. --data_dir MUST point at a
#    Lustre path (or another shared dir) — nemo-run stages the source tree to
#    job_dir, so writing data into the repo would ship hundreds of MB of audio
#    on every job submission. Use the same path as the YAML's `data_dir` field.
python nemo_skills/dataset/fdb/scripts/fdb_v3/prepare.py \
    --data_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/data/fdb

# 2) Run eval (generation + scoring submitted as Slurm jobs).
python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml

# 3) Re-score only (after fixing the scorer).
python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml \
    --scoring_only --scoring_force

# 4) Re-score without ASR (debug only — collapses response_qual; skips GPU).
python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_s2s_incremental_v2_config_fc_greedy.yaml \
    --scoring_only --scoring_force --skip_asr
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
