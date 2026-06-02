# Vendored: Full-Duplex-Bench v3 official scoring

These scoring scripts are vendored verbatim from the official public benchmark
and used by the `fdb_v3_official` eval variant. Inference is unchanged from
`fdb_v3` (same `FDBv3GenerationTask`); only the **scoring** comes from upstream.

- **Upstream**: https://github.com/DanielLin94144/Full-Duplex-Bench (`v3/`)
- **Commit**: `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`
- **Vendored**: 2026-06-02

## Files

| File | Source | Notes |
|---|---|---|
| `evaluate_tool_calls.py` | `v3/evaluate_tool_calls.py` | tool-selection F1, argument accuracy, response quality |
| `evaluate_pass_rate.py` | `v3/evaluate_pass_rate.py` | strict binary pass/fail |
| `analyze_tool_latency.py` | `v3/analyze_tool_latency.py` | first-response / tool-call / task-completion latency |
| `benchmark_data_v2.json` | `v3/benchmark_data_v2.json` | byte-identical to `FDBV3_CHENCHEN/FD3/release_code/benchmark_data_v2.json` |
| `_nv_judge.py` | (new, ours) | NV-gateway judge routing helper — see below |

## The only behavioral patch: judge routing

Upstream calls the **gpt-4o** judge directly against the public OpenAI API
(`api.openai.com`). SLURM scoring jobs can't reach that, so the same gpt-4o
judge is routed through NVIDIA's OpenAI-compatible gateway
(`https://inference-api.nvidia.com/v1`) via `_nv_judge.py`. The decision
(gpt-4o, via NV gateway) was explicit; defaults preserve the official
methodology (judge = gpt-4o).

Exact edits applied to each upstream script (grep `vendored patch`):
- `_get_openai_client()` / direct `OpenAI(...)` → `_nv_judge.get_client()`
  (base_url + key from env: `OPENAI_BASE_URL`/`NVIDIA_BASE_URL`, `NVIDIA_API_KEY`/`OPENAI_API_KEY`).
- `model="gpt-4o"` → `model=_nv_judge_model()` (env `FD3_JUDGE_MODEL`, default
  `gpt-4o` → mapped to the gateway name `azure/openai/gpt-4o`).

No scoring *logic* was changed. To re-sync with upstream: re-copy the four
upstream files and re-apply the two `vendored patch` substitutions per script.

## Diff vs the private `FDBV3_CHENCHEN/FD3/release_code` copy

Same code lineage; the public release differs only in the judge layer
(CHENCHEN uses a pluggable `openai_compat` defaulting to `gpt-5.2`; upstream
hardcodes gpt-4o) plus minor scoring tweaks (`turn_take_success` condition, an
exact-then-LLM `judge_args` wrapper). `benchmark_data_v2.json` is identical.
