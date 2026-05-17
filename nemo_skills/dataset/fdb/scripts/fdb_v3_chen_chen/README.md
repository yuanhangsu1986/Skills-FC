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
| Inference | nemo-skills `serve_unified` w/ DSFTS backend | Upstream `run_s2s_offline_benchmark.py` driving cchen1's `NemotronVoicechatInferenceWrapper` (mirrored at `NeMo_Voice_Chat_Chen_Chen`) |
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

- 2-GPU node access (`num_gpus: 2` — orchestrator pins Backend to GPU 0, S2S to GPU 1).
- A self-contained local mirror of upstream's FD3 + Backend_agent at `fdb_repo_path` (default: `…/yuanhangs/codes/NeMo/FDBV3_CHENCHEN/`). Must contain: `FD3/bin/run_fd3_audio_eval_job.sh`, `FD3/release_code/` (writable), `FD3/fdb_v3_data_released/` (writable, with the 100 input samples), and `Backend_agent/` (the inner Python package with `setup_llm.sh`, `start_backend_agent.sh`, `langGraph/`).
- Read access to the upstream container `.../containers/eval_agent.sqsh` (default; overridable in YAML).
- Yuanhangs-owned mirror of cchen1's NeMo fork at `nemo_code_path` (default: `NeMo_Voice_Chat_Chen_Chen/`). Must contain `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`. **Do NOT point this at DRIRF** — the wrappers diverged (different FC state-machine + prefill logic) and DRIRF causes the model to emit pure PAD/EOS tokens. Re-mirror with `cp -a /lustre/.../cchen1/.../Backend_agent/NeMo  NeMo_Voice_Chat_Chen_Chen` if upstream advances.
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
With REPO_ROOT = FDBV3_CHENCHEN, we satisfy both by symlinking
`FDBV3_CHENCHEN/NeMo → NeMo_Voice_Chat_Chen_Chen` (yuanhangs-owned mirror of
cchen1's NeMo fork). We also export `PYTHONPATH=<NeMo_Voice_Chat_Chen_Chen>:$PYTHONPATH`
before invoking the orchestrator as belt-and-suspenders.

**Why not DRIRF?** Chen chen's run_s2s_offline_benchmark.py was developed against
cchen1's NeMo fork. The two have diverged: cchen1 has `_do_fc_prefill`,
`get_fc_debug_counters`, a specific FC state machine; DRIRF replaced these with
`_create_silence_embedding_cache`, `_apply_fc_state_machine`, `_convert_nums_in_json`,
etc. ~857 lines diff between the two wrappers. Using DRIRF causes the model
to emit pure PAD/EOS tokens during inference, even though all the surrounding
fixes (cuda-graph disable, Nano v2 prompt, mono audio) are correct.

## Troubleshooting

- **`ModuleNotFoundError: nemo.collections.speechlm2.inference...`**: `nemo_code_path` doesn't point at a valid checkout. Verify the file exists at `<nemo_code_path>/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`. The expected source is cchen1's NeMo fork (`NeMo_Voice_Chat_Chen_Chen`); DRIRF is incompatible (see the "How the wrapper is wired" section).
- **Model emits only PAD / `</s>` tokens (empty user STT, no tool calls)**: most likely the `nemo_code_path` is pointing at the wrong NeMo fork (e.g., DRIRF or FC_Viet). Switch back to `NeMo_Voice_Chat_Chen_Chen`.
- **`ModuleNotFoundError: lightning`**: you're running outside the container. `server_container` / `scoring_container` must be the upstream's `eval_agent.sqsh` (or another container with NeMo + lightning + vllm installed).
- **Backend agent fails to start**: check `<fd3_run>/logs/backend_agent.log` and `<fd3_run>/logs/vllm.log`. The orchestrator retries 5x with fresh ports.
- **`response_qual: 0` across the board**: usually means the LLM judge couldn't reach the gateway. Check `OPENAI_BASE_URL` / `OPENAI_API_KEY` env vars and the gateway model name.
