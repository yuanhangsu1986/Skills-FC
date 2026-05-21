# FDB v3 ChenChen — upstream-end-to-end variant

A reference-mode variant of `fdb_v3.tool_call` that drives the **upstream FD3
orchestrator** end-to-end instead of in-tree inference + custom scoring. Useful
when you want numbers comparable to upstream FDBV3 reports (and as a baseline
to compare against `../fdb_v3/`).

Everything here is isolated to this folder. No edits to `fdb_v3/`,
nemo-skills core, or the FDBV3 / NeMo_fc upstream repos.

## Differences from `fdb_v3`

| Item | `fdb_v3` (in-tree) | `fdb_v3_chen_chen` (upstream) |
| --- | --- | --- |
| Inference | nemo-skills `serve_unified` w/ DSFTS backend | Upstream `run_s2s_offline_benchmark.py` driving cchen1's `NemotronVoicechatInferenceWrapper` (vendored at `FDBV3_CHENCHEN/NeMo`, the submodule tracking `github.com/yuanhangsu1986/NeMo_fc.git@fdb_v3_chen_chen`) |
| Agent loop | None (model emits tool calls directly to `output.jsonl`) | Backend Agent (FastAPI, `GRAPH_MODE=fd3`) + Qwen3-30B-A3B vLLM |
| Output WAV | Stereo (ch0=user, ch1=model) | Mono (model channel only) |
| Scoring | In-tree `run_scoring.py` (reconstructs FD3 layout, runs Parakeet ASR, calls evaluators) | Upstream `run_fd3_eval.sh` does ASR + evaluators inline; our `run_scoring.py` only ingests the reports |
| System prompt | Nano v2 full schema (`render_prompt.py` from `vtrinh/.../template.jinja`) | Same Nano v2 prompt (rendered at submission time and injected via `FD3_S2S_SYSTEM_PROMPT`) |
| Output WAV channel handling | Custom (`_to_model_channel_mono_wav`) | Not needed — upstream produces mono |
| Benchmark key | `fdb_v3.tool_call` | `fdb_v3_chen_chen.tool_call` |

Same checkpoint will produce **different numbers** under the two variants
because the inference wrapper (cchen1's chen_chen fork vs DSFTS) is different
code. That's the point — `chen_chen` is the upstream-comparable baseline.

## Prerequisites

- GPU allocation per shard depends on `backend_agent` config: `backend_agent.model: mock_api` (default) needs only 1 GPU (S2S on GPU 0, mock API in-process on CPU). LLM-driven backend with `backend_agent.gpu: 1` starts a local vLLM and needs 2 GPUs (vLLM on GPU 0, S2S on GPU 1).
- A clone of the FDBV3 repo at `fdb_repo_path` (default: `/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/FDBV3_CHENCHEN`, tracking `github.com/yuanhangsu1986/FDBV3@chen_chen`). Must contain `FD3/bin/run_fd3_audio_eval_job.sh`, `FD3/release_code/` (writable), `FD3/fdb_v3_data_released/` (writable, with the 100 input samples), `Backend_agent/`, and the `NeMo/` submodule.
- The `NeMo/` submodule **must be initialized** before running. Either clone with `git clone --recurse-submodules` or run `git submodule update --init --recursive` afterwards. The submodule tracks `github.com/yuanhangsu1986/NeMo_fc.git@fdb_v3_chen_chen` and ships a stripped-down NeMo tree (~30 MB) containing only the chen_chen-required modules (`nemo/collections/speechlm2/...`, `examples/speechlm2/...`). The full NeMo_fc.git default branch is much larger; the submodule deliberately ships only the chen_chen-compatible subset.
- Read access to the upstream container `.../containers/eval_agent.sqsh` (default; overridable in YAML).
- **`nemo_code_path` (default: `<fdb_repo_path>/NeMo`)** must point at the initialized submodule. Must contain `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`. **Do NOT point this at DRIRF or DSFTS** — the wrappers diverged (different FC state-machine + prefill logic) and the other forks cause the model to emit pure PAD/EOS tokens. To advance the submodule to a newer cchen1 snapshot, bump the pin: `cd <fdb_repo_path>/NeMo && git pull && cd .. && git add NeMo && git commit -m "bump NeMo"`.
- API key for the LLM judge (defaults to NVIDIA gateway). Set `NVIDIA_API_KEY` (or override `openai_api_key_env_var` in YAML).

## Usage

```bash
python nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_eval.py \
    --config nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/fdb_v3_chen_chen_config.yaml \
    --model /path/to/s2s_ckpt
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

## How the wrapper is wired

The upstream `run_s2s_offline_benchmark.py` hardcodes
`sys.path.insert(0, REPO_ROOT/"NeMo")` (line 302) and also has a precheck on
`NEMO_WRAPPER = REPO_ROOT / "NeMo" / nemo/collections/speechlm2/.../nemotron_voicechat_inference_wrapper.py`.
With `REPO_ROOT = fdb_repo_path` (= the FDBV3 repo checkout), `<REPO_ROOT>/NeMo` is the
**submodule** that ships cchen1's NeMo wrapper (tracking `NeMo_fc.git@fdb_v3_chen_chen`).
We also export `PYTHONPATH=<nemo_code_path>:$PYTHONPATH` before invoking the orchestrator
as belt-and-suspenders. `nemo_code_path` defaults to `<fdb_repo_path>/NeMo` — the same
submodule — so the two paths agree on a single checkout.

**Why not DRIRF or DSFTS?** The chen_chen run_s2s_offline_benchmark.py was developed against
cchen1's NeMo fork (now mirrored on `NeMo_fc.git@fdb_v3_chen_chen`). The other forks
have diverged: cchen1 has `_do_fc_prefill`, `get_fc_debug_counters`, a specific FC state
machine; DRIRF replaced these with `_create_silence_embedding_cache`,
`_apply_fc_state_machine`, `_convert_nums_in_json`, etc. (~857 lines diff between the
wrappers). Using DRIRF causes the model to emit pure PAD/EOS tokens during inference,
even though all the surrounding fixes (cuda-graph disable, Nano v2 prompt, mono audio)
are correct.

## Troubleshooting

- **`ModuleNotFoundError: nemo.collections.speechlm2.inference...`**: the `NeMo` submodule isn't initialized or `nemo_code_path` doesn't point at it. Verify the file exists at `<nemo_code_path>/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`. If the submodule directory is empty after cloning the FDBV3 repo, run `git submodule update --init --recursive` from `<fdb_repo_path>`. The expected source is the submodule tracking `NeMo_fc.git@fdb_v3_chen_chen`; DRIRF / DSFTS are incompatible (see "How the wrapper is wired").
- **Model emits only PAD / `</s>` tokens (empty user STT, no tool calls)**: most likely `nemo_code_path` is pointing at the wrong NeMo fork (e.g., DRIRF, FC_Viet, or DSFTS). Re-point at `<fdb_repo_path>/NeMo` (the submodule tracking `NeMo_fc.git@fdb_v3_chen_chen`).
- **`ModuleNotFoundError: lightning`**: you're running outside the container. `server_container` / `scoring_container` must be the upstream's `eval_agent.sqsh` (or another container with NeMo + lightning + vllm installed).
- **Backend agent fails to start**: check `<fd3_run>/logs/backend_agent.log` and `<fd3_run>/logs/vllm.log`. The orchestrator retries 5x with fresh ports.
- **`response_qual: 0` across the board**: usually means the LLM judge couldn't reach the gateway. Check `OPENAI_BASE_URL` / `OPENAI_API_KEY` env vars and the gateway model name.
