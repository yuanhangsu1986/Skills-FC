# FDB v3 ChenChen — upstream-end-to-end variant

A reference-mode variant of `fdb_v3.tool_call` that drives the **upstream FD3
orchestrator** end-to-end instead of in-tree inference + custom scoring. Useful
when you want numbers comparable to upstream FDBV3 reports (and as a baseline
to compare against `../fdb_v3/`).

Everything here is isolated to this folder. No edits to `fdb_v3/`,
nemo-skills core, or the upstream NeMo/FDBV3_CHENCHEN trees.

## Differences from `fdb_v3`

| Item | `fdb_v3` (in-tree) | `fdb_v3_chen_chen` (upstream) |
| --- | --- | --- |
| Inference | nemo-skills `serve_unified` w/ DSFTS backend | Upstream `run_s2s_offline_benchmark.py` driving DRIRF's `NemotronVoicechatInferenceWrapper` (vLLM LLM + vLLM EARTTS) |
| Agent loop | None (model emits tool calls directly to `output.jsonl`) | Backend Agent (FastAPI, `GRAPH_MODE=fd3`) + Qwen3-30B-A3B vLLM |
| Output WAV | Stereo (ch0=user, ch1=model) | Mono (model channel only) |
| Scoring | In-tree `run_scoring.py` (reconstructs FD3 layout, runs Parakeet ASR, calls evaluators) | Upstream `run_fd3_eval.sh` does ASR + evaluators inline; our `run_scoring.py` only ingests the reports |
| System prompt | Nano v2 full schema (`render_prompt.py` from `vtrinh/.../template.jinja`) | Upstream lightweight default (name+description) |
| Output WAV channel handling | Custom (`_to_model_channel_mono_wav`) | Not needed — upstream produces mono |
| Benchmark key | `fdb_v3.tool_call` | `fdb_v3_chen_chen.tool_call` |

Same checkpoint will produce **different numbers** under the two variants
because the inference wrapper (DRIRF vs DSFTS) is different code on a different
branch. That's the point — `chen_chen` is the upstream-comparable baseline.

## Prerequisites

- 2-GPU node access (`num_gpus: 2` — orchestrator pins Backend to GPU 0, S2S to GPU 1).
- Read access to `/lustre/fsw/portfolios/llmservice/users/cchen1/code/Backend_agent` and `.../containers/eval_agent.sqsh` (defaults; overridable in YAML).
- DRIRF checkout at `nemo_code_path` containing `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`.
- API key for the LLM judge (defaults to NVIDIA gateway). Set `NVIDIA_API_KEY` (or override `openai_api_key_env_var` in YAML).

## Usage

```bash
python nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/fdb_v3_chen_chen_config.yaml \
    --s2s_checkpoint_dir /path/to/s2s_ckpt
```

Common flags:

```bash
# Dry-run: print the submitted commands without executing.
... --dry_run

# Smoke test: 5 examples only.
... --max_examples 5

# Re-score from existing upstream reports without re-running inference.
... --scoring_only --scoring_force

# Skip scoring and just submit the upstream job.
... --generation_only

# Forward extra args to run_s2s_offline_benchmark.py (after --):
... -- --pad_audio_by_sec 12 --use_perception_cache
```

## What gets written where

```
<output_dir>/eval-results/fdb_v3_chen_chen.tool_call/
├── metrics.json                    ← key fdb_v3_chen_chen.tool_call (headline+eval+pass_rate+latency)
├── orchestrator-logs/              ← run_cmd logs for the upstream job
├── summarized-results/             ← run_cmd logs for scoring
└── fd3_run/                        ← FD3_RUN_ROOT passed to upstream
    ├── model_report.json           ← upstream consolidated summary
    ├── reports/
    │   ├── <provider>_eval.json
    │   ├── <provider>_pass_rate.json
    │   └── <provider>_latency.json
    ├── logs/                       ← vLLM, backend_agent, gpu_keepalive
    └── per_sample_results/         ← per-scenario result_<provider>.json files
```

## How the wrapper hooks DRIRF without symlinks/patches

The upstream `run_s2s_offline_benchmark.py` hardcodes
`sys.path.insert(0, REPO_ROOT/"NeMo")` (line 302). That directory doesn't exist
in FDBV3_CHENCHEN — by itself the import would fail. We just export
`PYTHONPATH=<DRIRF>:$PYTHONPATH` before invoking the orchestrator, so when
`from nemo.collections.speechlm2.inference.model_wrappers...` runs, the
nonexistent `FDBV3_CHENCHEN/NeMo` path is silently skipped and the DRIRF path
on `PYTHONPATH` resolves the import.

This keeps the variant fully isolated — no symlinks under FDBV3_CHENCHEN, no
patches to the upstream script.

## Troubleshooting

- **`ModuleNotFoundError: nemo.collections.speechlm2.inference...`**: `nemo_code_path` doesn't point at a DRIRF (or FC_Viet) checkout. Verify the file exists at `<nemo_code_path>/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`.
- **`ModuleNotFoundError: lightning`**: you're running outside the container. `server_container` / `scoring_container` must be the upstream's `eval_agent.sqsh` (or another container with NeMo + lightning + vllm installed).
- **Backend agent fails to start**: check `<fd3_run>/logs/backend_agent.log` and `<fd3_run>/logs/vllm.log`. The orchestrator retries 5x with fresh ports.
- **`response_qual: 0` across the board**: usually means the LLM judge couldn't reach the gateway. Check `OPENAI_BASE_URL` / `OPENAI_API_KEY` env vars and the gateway model name.
