# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run FDB v3 ChenChen (upstream-end-to-end) eval.

Generation drives the upstream FD3 orchestrator (Backend Agent + vLLM + S2S via
cchen1's NemotronVoicechatInferenceWrapper, mirrored at FDBV3_CHENCHEN/NeMo via the
submodule github.com/yuanhangsu1986/NeMo_fc.git@fdb_v3_chen_chen)
on one 2-GPU node. Scoring just ingests the upstream reports into our metrics.json
schema.

Usage:
    python nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_eval.py \
        --config nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/fdb_v3_chen_chen_config.yaml
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

BENCHMARK = "fdb_v3_chen_chen.tool_call"
SCORING_SCRIPT = "nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_scoring.py"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _env_has_prompt_override(config: dict) -> bool:
    env = config.get("env") or {}
    return any(k in env for k in ("FD3_S2S_SYSTEM_PROMPT", "S2S_SYSTEM_PROMPT"))


def _maybe_render_fd3_system_prompt(config: dict) -> str | None:
    """Render the in-tree fdb_v3 Nano v2 FD3 system prompt (full schema)
    using fdb_v3/render_prompt.py. Returns None when:
      - user explicitly set `s2s_system_prompt` in YAML
      - user pinned FD3_S2S_SYSTEM_PROMPT / S2S_SYSTEM_PROMPT in `env:` dict
      - YAML toggle `render_fd3_system_prompt: false`
      - rendering fails (jinja2 missing on submission host, etc.) — falls
        back to upstream's lightweight default
    """
    if config.get("render_fd3_system_prompt") is False:
        return None
    if config.get("s2s_system_prompt"):
        return None
    if _env_has_prompt_override(config):
        return None
    try:
        from nemo_skills.dataset.fdb.scripts.fdb_v3.render_prompt import (
            render_fd3_system_prompt,
        )
        fdb_repo = Path(config["fdb_repo_path"])
        rendered = render_fd3_system_prompt(fdb_repo=fdb_repo)
        print(f"  rendered Nano v2 FD3 system prompt ({len(rendered)} chars) via fdb_v3/render_prompt.py")
        return rendered
    except Exception as e:
        print(
            f"  warning: could not render Nano v2 FD3 prompt ({type(e).__name__}: {e}); "
            f"upstream's lightweight default will apply"
        )
        return None


def _eval_results_dir(config: dict) -> str:
    return f"{config['output_dir']}/eval-results/{BENCHMARK}"


def _fd3_run_root(config: dict) -> str:
    # Parent root holding per-shard inference subtrees (shard_0/, shard_1/, ...)
    # plus the merged/ subtree where the scoring step assembles a single unified
    # per_sample_data view + reports. All under eval-results/<benchmark>/ so a
    # single rm -rf of the eval-results tree restores a clean slate.
    return f"{_eval_results_dir(config)}/fd3_run"


def _shard_run_root(config: dict, shard_id: int) -> str:
    return f"{_fd3_run_root(config)}/shard_{shard_id}"


def _merged_run_root(config: dict) -> str:
    return f"{_fd3_run_root(config)}/merged"


def _resolve_backend_agent(config: dict) -> dict:
    """Normalize the `backend_agent` config block.

    Returns a dict with keys: model (str), gpu (int 0/1), vllm_model_path (str|None),
    is_mock (bool), on_gpu (bool). Defaults to {model: mock_api, gpu: 0}.
    """
    ba = config.get("backend_agent") or {}
    model = str(ba.get("model", "mock_api"))
    gpu = int(ba.get("gpu", 0))
    if gpu not in (0, 1):
        print(f"  warning: backend_agent.gpu={gpu!r} not in (0,1); clamping to 0")
        gpu = 0
    resolved = {
        "model": model,
        "gpu": gpu,
        "vllm_model_path": ba.get("vllm_model_path"),
        # Model-family-specific token names. Forwarded to the wrapper via env
        # vars. The wrapper has NO hardcoded defaults — intent/ready/trigger
        # MUST be set in YAML or wrapper init will raise. response_open/close
        # may be empty (then no single-token wrapping is applied).
        "trigger_function_token": ba.get("trigger_function_token"),
        "intent_token": ba.get("intent_token"),
        "ready_token": ba.get("ready_token"),
        "response_open_token": ba.get("response_open_token"),
        "response_close_token": ba.get("response_close_token"),
        "is_mock": model == "mock_api",
        "on_gpu": gpu == 1,
    }
    config["backend_agent"] = resolved
    return resolved


def _validate_shard_config(config: dict) -> tuple[int, dict]:
    """Read num_shards + backend_agent from YAML; warn + clamp num_shards to [1,8]."""
    backend = _resolve_backend_agent(config)
    raw = config.get("num_shards", 1)
    try:
        num_shards = int(raw)
    except (TypeError, ValueError):
        print(f"  warning: num_shards={raw!r} not an int; defaulting to 1")
        num_shards = 1
    if num_shards < 1:
        print(f"  warning: num_shards={num_shards} < 1; clamping to 1")
        num_shards = 1
    if num_shards > 8:
        print(f"  warning: num_shards={num_shards} > 8; clamping to 8")
        num_shards = 8
    config["num_shards"] = num_shards
    return num_shards, backend


def _list_sample_folders(config: dict) -> list[str]:
    upstream_data = Path(config["fdb_repo_path"]) / "FD3" / "fdb_v3_data_released"
    if not upstream_data.exists():
        print(f"  warning: upstream data dir not found at {upstream_data}")
        return []
    return sorted(
        d.name for d in upstream_data.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _calls_by_folder(config: dict, folders: list[str]) -> dict[str, int]:
    """Map each sample folder name to the scenario's num_expected_calls.

    Folder names look like `<scenario_id>_<speaker_pid>` (e.g.
    `ecommerce_01_65e8cf8f...`). We resolve by matching against scenario IDs
    in benchmark_data_v2.json (longest prefix wins). Folders without a match
    fall back to 1 (single-call workload assumption).
    """
    bench_path = Path(config["fdb_repo_path"]) / "FD3" / "release_code" / "benchmark_data_v2.json"
    if not bench_path.exists():
        print(f"  warning: benchmark_data_v2.json not found at {bench_path}; assuming 1 call/sample")
        return {f: 1 for f in folders}
    with bench_path.open("r", encoding="utf-8") as fh:
        bench = json.load(fh)
    calls_by_sid = {s["id"]: int(s.get("num_expected_calls", 1)) for s in bench.get("scenarios", [])}
    # Longest scenario IDs first so e.g. `ecommerce_10_...` doesn't get matched
    # by `ecommerce_1` (no such id, but defensive).
    sorted_sids = sorted(calls_by_sid.keys(), key=len, reverse=True)

    def _lookup(folder: str) -> int:
        for sid in sorted_sids:
            if folder == sid or folder.startswith(sid + "_"):
                return calls_by_sid[sid]
        return 1
    return {f: _lookup(f) for f in folders}


def _bin_pack_shards(folders: list[str], loads: dict[str, int], num_shards: int) -> list[list[str]]:
    """Greedy bin-pack: assign the heaviest sample to the currently-lightest shard.
    Aims to minimize max-shard total `num_expected_calls`. Deterministic
    tie-breaks (sorted by load desc, then folder name asc; shard index when
    multiple shards tie on load).
    """
    if num_shards <= 1:
        return [list(folders)] if folders else [[]]
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    shard_loads = [0] * num_shards
    ordered = sorted(folders, key=lambda f: (-loads.get(f, 1), f))
    for folder in ordered:
        target = min(range(num_shards), key=lambda i: (shard_loads[i], i))
        shards[target].append(folder)
        shard_loads[target] += loads.get(folder, 1)
    return shards


def _stage_per_sample_data_dir(config: dict, run_root: str, sample_names: list[str] | None = None) -> str:
    """Create a per-shard mirror of FDBV3_CHENCHEN/FD3/fdb_v3_data_released under
    {run_root}/per_sample_data/. Each sample dir contains a symlink to the upstream
    input.wav + metadata.json; outputs (output_*.wav, result_*.json, _offline_*)
    will land here at inference time. If `sample_names` is given, only those
    samples are linked (shard subset); otherwise all are linked."""
    upstream_data = Path(config["fdb_repo_path"]) / "FD3" / "fdb_v3_data_released"
    staged = Path(run_root) / "per_sample_data"
    staged.mkdir(parents=True, exist_ok=True)
    if not upstream_data.exists():
        print(f"  warning: upstream data dir not found at {upstream_data}; per-sample staging skipped")
        return str(staged)
    if sample_names is None:
        sample_names = [d.name for d in upstream_data.iterdir() if d.is_dir()]
    n = 0
    for name in sample_names:
        src_sample = upstream_data / name
        if not src_sample.is_dir():
            continue
        dst_sample = staged / name
        dst_sample.mkdir(exist_ok=True)
        for fname in ("input.wav", "metadata.json"):
            src = src_sample / fname
            dst = dst_sample / fname
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError as e:
                    print(f"  warning: symlink {dst} -> {src} failed: {e}")
        n += 1
    print(f"  staged {n} samples at {staged}")
    return str(staged)


def build_generation_command(
    config: dict,
    extra_passthrough: list[str],
    shard_id: int,
    num_shards: int,
    shard_run_root: str,
    shard_per_sample_data_dir: str,
) -> str:
    """Compose the shell command that launches the upstream FD3 orchestrator
    for one shard.

    Exports the env vars upstream expects (per-shard FD3_RUN_ROOT/RESULTS_DIR/
    LOGS_DIR/REPORTS_DIR), prepends the cchen1-NeMo mirror to PYTHONPATH so the
    NemotronVoicechatInferenceWrapper import resolves, then execs the
    orchestrator. The shard runs on 1 GPU when `backend_agent.model == mock_api`
    OR `backend_agent.gpu == 0` (mock dispatch or hosted-LLM; no local vLLM),
    or on 2 GPUs when an LLM-driven Backend Agent runs locally (vLLM on GPU 0,
    S2S on GPU 1). Per-shard evaluators are skipped (--skip-tool-eval /
    --skip-pass-rate / --skip-latency); the scoring step runs them once over the
    merged result tree.
    """
    fdb_repo = config["fdb_repo_path"]
    drirf_path = config["nemo_code_path"]
    # Accept both `model` (rewritten by run_all_benchmarks.sh make_patched_config)
    # and `s2s_checkpoint_dir` (the YAML's original key) for backward compat.
    s2s_ckpt = config.get("model") or config.get("s2s_checkpoint_dir")
    if not s2s_ckpt:
        raise SystemExit("Config missing 'model' / 's2s_checkpoint_dir'")
    backend = _resolve_backend_agent(config)
    provider = config.get("provider", "fdb_v3_chen_chen")
    # Match upstream FDBV3_CHENCHEN/FD3/slurm/fd3_audio_eval.sh:35 — RUN_LABEL
    # defaults to basename(checkpoint). Drives `model_name` in model_report.json.
    run_label = config.get("run_label") or Path(s2s_ckpt).name

    # Render the Nano v2 FD3 system prompt — the same one the in-tree fdb_v3
    # pipeline uses and the format these S2S checkpoints were fine-tuned on.
    # Chen chen's run_fd3_eval.sh defaults to a lightweight (name+description
    # only) prompt; with that, the model emits pure PAD tokens. We inject the
    # rendered Nano v2 prompt via FD3_S2S_SYSTEM_PROMPT, which run_fd3_eval.sh
    # honors (line 39): S2S_SYSTEM_PROMPT > FD3_S2S_SYSTEM_PROMPT > default.
    rendered_prompt = _maybe_render_fd3_system_prompt(config)
    if rendered_prompt is not None:
        env_exports_extra_prompt = [f"export FD3_S2S_SYSTEM_PROMPT={shlex.quote(rendered_prompt)}"]
    else:
        env_exports_extra_prompt = []

    # Optional knobs forwarded to run_fd3_eval.sh -> run_s2s_offline_benchmark.py.
    # NOTE: do NOT pass --max_examples here — sample subset selection happens
    # at the per-shard staging step (only the assigned subset is symlinked into
    # FD3_RESULTS_DIR), so the inner script sees exactly the shard's samples.
    pass_args: list[str] = []
    if config.get("engine_type"):
        pass_args += ["--engine_type", str(config["engine_type"])]
    if config.get("temperature") is not None:
        pass_args += ["--temperature", str(config["temperature"])]
    if config.get("top_p") is not None:
        pass_args += ["--top_p", str(config["top_p"])]
    if config.get("repetition_penalty") is not None:
        pass_args += ["--repetition_penalty", str(config["repetition_penalty"])]
    if config.get("force_turn_taking") is True:
        pass_args += ["--force_turn_taking"]
    if config.get("speaker_reference"):
        pass_args += ["--speaker_reference", str(config["speaker_reference"])]
    if config.get("s2s_system_prompt"):
        pass_args += ["--s2s-system-prompt", str(config["s2s_system_prompt"])]
    if config.get("force"):
        pass_args += ["--force"]
    # Skip in-shard evaluators — scoring step runs them once on the merged tree.
    pass_args += ["--skip-tool-eval", "--skip-pass-rate", "--skip-latency"]
    pass_args += list(extra_passthrough)
    pass_args_str = " ".join(shlex.quote(a) for a in pass_args)

    logs_dir = f"{shard_run_root}/logs"
    reports_dir = f"{shard_run_root}/reports"

    # Derived env vars (depend on other YAML fields). These are the only ones
    # the script computes itself; everything else is user-controlled via
    # the `env:` dict in the YAML so we don't bake assumptions about which
    # variables the upstream Backend Agent / orchestrator needs.
    env_exports = [
        f"export S2S_CHECKPOINT_DIR={shlex.quote(s2s_ckpt)}",
        f"export RUN_LABEL={shlex.quote(run_label)}",
        f"export FD3_PROVIDER={shlex.quote(provider)}",
        # Used by agent_handler.py's mock_api dispatch to locate
        # Backend_agent/langGraph/fd3_tools._execute when Backend_agent isn't
        # directly importable (which it isn't from FD3/release_code/).
        f"export FDB_REPO_PATH={shlex.quote(str(fdb_repo))}",
        f"export FD3_RUN_ROOT={shlex.quote(shard_run_root)}",
        f"export FD3_RESULTS_DIR={shlex.quote(shard_per_sample_data_dir)}",
        f"export FD3_LOGS_DIR={shlex.quote(logs_dir)}",
        f"export FD3_REPORTS_DIR={shlex.quote(reports_dir)}",
        # Multi-shard runs write to the checkpoint dir concurrently — race-free
        # by gating the side-effect symlink + latest-report write.
        f"export FD3_SKIP_CHECKPOINT_DIR_WRITES=1",
        f"export FD3_SHARD_ID={shard_id}",
        f"export FD3_NUM_SHARDS={num_shards}",
        f"export PYTHONPATH={shlex.quote(drirf_path)}:${{PYTHONPATH:-}}",
    ]
    # Always export the backend_agent config to the orchestrator shell.
    # run_fd3_audio_eval_job.sh decides whether to start vLLM based on these.
    env_exports += [
        f"export BACKEND_AGENT_MODEL={shlex.quote(backend['model'])}",
        f"export BACKEND_AGENT_ON_GPU={1 if backend['on_gpu'] else 0}",
    ]
    if backend["vllm_model_path"]:
        env_exports.append(
            f"export BACKEND_AGENT_VLLM_MODEL_PATH={shlex.quote(str(backend['vllm_model_path']))}"
        )
    if backend["trigger_function_token"]:
        env_exports.append(
            f"export BACKEND_AGENT_TRIGGER_FUNC_TOKEN={shlex.quote(str(backend['trigger_function_token']))}"
        )
    if backend.get("intent_token"):
        env_exports.append(
            f"export BACKEND_AGENT_INTENT_TOKEN={shlex.quote(str(backend['intent_token']))}"
        )
    if backend.get("ready_token"):
        env_exports.append(
            f"export BACKEND_AGENT_READY_TOKEN={shlex.quote(str(backend['ready_token']))}"
        )
    # response_open/close can be intentionally empty (no single-token wrapping);
    # export the env var anyway so the wrapper distinguishes "explicitly empty"
    # from "not configured".
    if backend.get("response_open_token") is not None:
        env_exports.append(
            f"export BACKEND_AGENT_RESPONSE_OPEN_TOKEN={shlex.quote(str(backend['response_open_token']))}"
        )
    if backend.get("response_close_token") is not None:
        env_exports.append(
            f"export BACKEND_AGENT_RESPONSE_CLOSE_TOKEN={shlex.quote(str(backend['response_close_token']))}"
        )
    if not backend["on_gpu"]:
        # Single-GPU layout: bind S2S to GPU 0 (the only one this SLURM job
        # gets), and silence Backend Agent host fields so any leftover
        # references don't try to talk to a non-existent local vLLM.
        env_exports += [
            "export S2S_GPU=0",
            "export BACKEND_GPU=0",
        ]
    # Free-form env vars. Values are double-quoted so shell expansion (${...})
    # works — that's how the user redirects between upstream-expected names
    # and whatever the cluster actually injects.
    for k, v in (config.get("env") or {}).items():
        env_exports.append(f'export {k}="{v}"')
    # Append the rendered FD3 prompt last so that explicit user overrides
    # placed in the `env:` dict (FD3_S2S_SYSTEM_PROMPT or S2S_SYSTEM_PROMPT)
    # are NOT overridden when present; if not present, our render takes effect.
    if env_exports_extra_prompt and not _env_has_prompt_override(config):
        env_exports.extend(env_exports_extra_prompt)

    orchestrator = f"{fdb_repo}/FD3/bin/run_fd3_audio_eval_job.sh"

    # Optional container-specific setup the user controls via YAML, e.g. a
    # `python -> python3` symlink shim if the container only ships `python3`
    # while upstream Backend_agent/start_backend_agent.sh invokes `python`.
    # Empty by default — no assumptions baked into the script.
    pre_command = (config.get("pre_command") or "").strip()
    pre_command_block = (pre_command + "\n") if pre_command else ""

    # The orchestrator's REPO_ROOT is computed as `dirname(dirname(dirname(script)))`,
    # so cd-ing into REPO_ROOT first ensures it picks up the sibling Backend_agent/.
    return (
        pre_command_block
        + "\n".join(env_exports)
        + "\n"
        + f'cd {shlex.quote(fdb_repo)} && bash {shlex.quote(orchestrator)} {pass_args_str}'
    )


def build_score_command(config: dict, num_shards: int, force: bool = False) -> str:
    eval_results_dir = _eval_results_dir(config)
    fd3_run_root = _fd3_run_root(config)
    merged_root = _merged_run_root(config)
    shard_roots = [_shard_run_root(config, k) for k in range(num_shards)]
    python_exec = (
        config.get("scoring_container_python_exec")
        or config.get("server_container_python_exec")
        or "python"
    )
    cmd_args = [
        f"{python_exec} {SCORING_SCRIPT}",
        f"--eval_results_dir {shlex.quote(eval_results_dir)}",
        f"--fd3_run_root {shlex.quote(fd3_run_root)}",
        f"--merged_run_root {shlex.quote(merged_root)}",
        f"--fdb_repo_path {shlex.quote(config['fdb_repo_path'])}",
        f"--provider {shlex.quote(config.get('provider', 'fdb_v3_chen_chen'))}",
        f"--run_label {shlex.quote(config.get('run_label') or Path(config.get('model') or config.get('s2s_checkpoint_dir') or '').name)}",
        f"--shard_run_roots " + " ".join(shlex.quote(p) for p in shard_roots),
    ]
    if config.get("use_llm_judge", True):
        cmd_args.append("--use-llm")
    if force:
        cmd_args.append("--force")

    # Propagate the YAML `env:` block (e.g. OPENAI_API_KEY, NVIDIA_API_KEY,
    # JUDGE_MODEL) into the scoring container. Without this, the LLM judge's
    # openai_compat.api_key() check returns None — every per-sample
    # response_qual ends up "OpenAI client unavailable" / score 0. Bash
    # `export K="${V:-}"` lines expand against the cluster shell env at
    # runtime, same pattern used by build_generation_command.
    env_exports = [f'export {k}="{v}"' for k, v in (config.get("env") or {}).items()]
    cmd_str = " ".join(cmd_args)
    if env_exports:
        cmd_str = " && ".join(env_exports) + " && " + cmd_str
    return cmd_str


def run_fdb_v3_chen_chen_eval(config: dict, extra_passthrough: list[str]) -> None:
    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    num_shards, backend = _validate_shard_config(config)
    print(
        f"FDB v3 ChenChen: benchmark={BENCHMARK}, output_dir={config['output_dir']}, "
        f"num_shards={num_shards}, backend_agent={{model: {backend['model']!r}, gpu: {backend['gpu']}}}"
    )
    expname = config.get("expname", "fdb_v3_chen_chen")
    eval_results_path = _eval_results_dir(config)
    gpus_per_shard = 2 if backend["on_gpu"] else 1

    # Compute per-shard sample assignments. Greedy bin-pack over
    # `num_expected_calls` so that multi-call (and thus multi-turn) scenarios
    # are spread evenly — avoids one shard ending up with all the expensive
    # samples while others sit idle.
    all_samples = _list_sample_folders(config)
    if config.get("max_examples"):
        try:
            max_n = int(config["max_examples"])
            all_samples = all_samples[:max_n]
        except (TypeError, ValueError):
            print(f"  warning: max_examples={config['max_examples']!r} ignored (not an int)")
    loads = _calls_by_folder(config, all_samples)
    shard_samples = _bin_pack_shards(all_samples, loads, num_shards)
    shard_loads = [sum(loads.get(s, 1) for s in shard) for shard in shard_samples]
    print(
        f"  sample distribution: total={len(all_samples)} "
        f"(total expected_calls={sum(shard_loads)}), "
        f"per-shard sizes={[len(s) for s in shard_samples]}, "
        f"per-shard expected_calls={shard_loads}"
    )

    shard_expnames: list[str] = []
    any_shard_submitted = False
    if not scoring_only:
        for shard_id in range(num_shards):
            shard_root = _shard_run_root(config, shard_id)
            # Per-shard skip-if-completed marker. Each shard's inner orchestrator
            # writes its own model_report.json at FD3_RUN_ROOT on success.
            report_marker = Path(shard_root) / "model_report.json"
            if report_marker.exists() and not config.get("force"):
                print(f"Skipping shard {shard_id} (found {report_marker})")
                continue
            if not shard_samples[shard_id]:
                print(f"Skipping shard {shard_id} (no samples assigned)")
                continue
            print(
                f"Submitting shard {shard_id}/{num_shards} "
                f"({len(shard_samples[shard_id])} samples, {gpus_per_shard} GPU"
                f"{'s' if gpus_per_shard > 1 else ''})..."
            )
            shard_per_sample = _stage_per_sample_data_dir(config, shard_root, shard_samples[shard_id])
            gen_command = build_generation_command(
                config,
                extra_passthrough,
                shard_id=shard_id,
                num_shards=num_shards,
                shard_run_root=shard_root,
                shard_per_sample_data_dir=shard_per_sample,
            )
            container = config.get("server_container") or config.get("container") or "nemo-skills"
            shard_expname = f"{expname}_shard{shard_id}"
            shard_expnames.append(shard_expname)
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=gen_command,
                container=container,
                partition=config.get("partition"),
                num_gpus=gpus_per_shard,
                num_nodes=1,
                expname=shard_expname,
                installation_command=config.get("installation_command"),
                log_dir=f"{eval_results_path}/orchestrator-logs/shard_{shard_id}",
                mount_paths=config.get("mount_paths"),
                time_min=config.get("time_min"),
                reuse_code=False,
                dry_run=dry_run,
            )
            any_shard_submitted = True

    if not generation_only:
        print(f"Submitting scoring (merge {num_shards} shard(s) and run evaluators)...")
        score_command = build_score_command(config, num_shards=num_shards, force=config.get("scoring_force", False))
        scoring_container = (
            config.get("scoring_container") or config.get("server_container") or "nemo-skills"
        )
        scoring_gpus = config.get("scoring_gpus", 0)
        # GPU partitions reject 0-GPU jobs; fall back to cpu_partition for the
        # scoring step (judge-LLM API calls + JSON shuffling).
        scoring_partition = config.get("scoring_partition")
        if not scoring_partition:
            scoring_partition = config.get("cpu_partition") if scoring_gpus == 0 else config.get("partition")
        # Scoring waits on every submitted shard so it sees the union of all
        # per-sample results before invoking cchen1's evaluators.
        run_after = shard_expnames if any_shard_submitted else None
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=score_command,
            container=scoring_container,
            partition=scoring_partition,
            num_gpus=scoring_gpus,
            run_after=run_after,
            expname=f"{expname}_score",
            installation_command=config.get("scoring_installation_command"),
            log_dir=f"{eval_results_path}/summarized-results",
            reuse_code=False,
            dry_run=dry_run,
        )

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FDB v3 ChenChen (upstream-end-to-end) eval")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override S2S checkpoint path (alias for --s2s_checkpoint_dir; "
                                        "the orchestrator scripts/run_all_benchmarks.sh forwards --model "
                                        "to every benchmark)")
    parser.add_argument("--s2s_checkpoint_dir", help="Override S2S checkpoint path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--max_examples", type=int, help="Override max_examples")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--generation_only", action="store_true")
    parser.add_argument("--scoring_only", action="store_true")
    parser.add_argument("--scoring_force", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run generation even if model_report.json exists")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to run_fd3_eval.sh -> run_s2s_offline_benchmark.py",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # --model is an alias for --s2s_checkpoint_dir; it wins if both are set since the
    # orchestrator only forwards --model.
    if args.model is not None:
        config["model"] = args.model
    for key in ["cluster", "partition", "s2s_checkpoint_dir", "output_dir", "max_examples"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if args.scoring_force:
        config["scoring_force"] = True
    if args.force:
        config["force"] = True

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)

    extra_passthrough = list(args.extra or [])
    if extra_passthrough and extra_passthrough[0] == "--":
        extra_passthrough = extra_passthrough[1:]
    run_fdb_v3_chen_chen_eval(config, extra_passthrough)


if __name__ == "__main__":
    main()
