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
cchen1's NemotronVoicechatInferenceWrapper, mirrored at NeMo_Voice_Chat_Chen_Chen)
on one 2-GPU node. Scoring just ingests the upstream reports into our metrics.json
schema.

Usage:
    python nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/run_eval.py \
        --config nemo_skills/dataset/fdb/scripts/fdb_v3_chen_chen/fdb_v3_chen_chen_config.yaml
"""

from __future__ import annotations

import argparse
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
    # All upstream artifacts (reports/, logs/, per_sample_results/, model_report.json)
    # land here. Keeping it under eval-results/<benchmark>/ keeps everything for one
    # benchmark in one place.
    return f"{_eval_results_dir(config)}/fd3_run"


def _stage_per_sample_data_dir(config: dict, fd3_run_root: str) -> str:
    """Create a per-run mirror of FDBV3_CHENCHEN/FD3/fdb_v3_data_released under
    fd3_run/per_sample_data/. Each sample dir contains a symlink to the upstream
    input.wav; outputs (output_*.wav, result_*.json, _offline_*) will land here
    at inference time. Keeps all writes inside eval-results/, so rerun-by-deleting
    the eval-results tree fully restores a clean slate."""
    upstream_data = Path(config["fdb_repo_path"]) / "FD3" / "fdb_v3_data_released"
    staged = Path(fd3_run_root) / "per_sample_data"
    staged.mkdir(parents=True, exist_ok=True)
    if not upstream_data.exists():
        print(f"  warning: upstream data dir not found at {upstream_data}; per-sample staging skipped")
        return str(staged)
    n = 0
    for sample_dir in upstream_data.iterdir():
        if not sample_dir.is_dir():
            continue
        dst_sample_dir = staged / sample_dir.name
        dst_sample_dir.mkdir(exist_ok=True)
        for fname in ("input.wav", "metadata.json"):
            src = sample_dir / fname
            dst = dst_sample_dir / fname
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError as e:
                    print(f"  warning: symlink {dst} -> {src} failed: {e}")
        n += 1
    print(f"  staged per-sample data dir: {staged} ({n} samples linked)")
    return str(staged)


def build_generation_command(config: dict, extra_passthrough: list[str]) -> str:
    """Compose the shell command that launches the upstream FD3 orchestrator.

    Exports the env vars upstream expects, prepends the cchen1-NeMo mirror to
    PYTHONPATH so the NemotronVoicechatInferenceWrapper import resolves, then
    execs the orchestrator which brings up vLLM (Qwen3 on GPU 0) + Backend
    Agent + S2S (GPU 1) and runs run_s2s_offline_benchmark + the FD3 evaluators.
    """
    fdb_repo = config["fdb_repo_path"]
    drirf_path = config["nemo_code_path"]
    # Accept both `model` (rewritten by run_all_benchmarks.sh make_patched_config)
    # and `s2s_checkpoint_dir` (the YAML's original key) for backward compat.
    s2s_ckpt = config.get("model") or config.get("s2s_checkpoint_dir")
    if not s2s_ckpt:
        raise SystemExit("Config missing 'model' / 's2s_checkpoint_dir'")
    provider = config.get("provider", "fdb_v3_chen_chen")
    # Match upstream FDBV3_CHENCHEN/FD3/slurm/fd3_audio_eval.sh:35 — RUN_LABEL
    # defaults to basename(checkpoint). Drives `model_name` in model_report.json.
    run_label = config.get("run_label") or Path(s2s_ckpt).name
    run_root = _fd3_run_root(config)

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

    # Optional knobs forwarded to run_fd3_eval.sh -> run_s2s_offline_benchmark.py
    pass_args: list[str] = []
    if config.get("max_examples"):
        pass_args += ["--max_examples", str(config["max_examples"])]
    if config.get("engine_type"):
        pass_args += ["--engine_type", str(config["engine_type"])]
    if config.get("temperature") is not None:
        pass_args += ["--temperature", str(config["temperature"])]
    if config.get("top_p") is not None:
        pass_args += ["--top_p", str(config["top_p"])]
    if config.get("repetition_penalty") is not None:
        pass_args += ["--repetition_penalty", str(config["repetition_penalty"])]
    if config.get("speaker_reference"):
        pass_args += ["--speaker_reference", str(config["speaker_reference"])]
    if config.get("s2s_system_prompt"):
        pass_args += ["--s2s-system-prompt", str(config["s2s_system_prompt"])]
    if config.get("force"):
        pass_args += ["--force"]
    pass_args += list(extra_passthrough)
    pass_args_str = " ".join(shlex.quote(a) for a in pass_args)

    # Stage per-run data dir (symlinks for input.wav). Forces all writes
    # — per-sample result/output WAV, _offline_/wrapper_output, logs, reports —
    # to live under eval-results/.../fd3_run/, NOT in the FDBV3_CHENCHEN tree.
    per_sample_data_dir = _stage_per_sample_data_dir(config, run_root)
    logs_dir = f"{run_root}/logs"
    reports_dir = f"{run_root}/reports"

    # Derived env vars (depend on other YAML fields). These are the only ones
    # the script computes itself; everything else is user-controlled via
    # the `env:` dict in the YAML so we don't bake assumptions about which
    # variables the upstream Backend Agent / orchestrator needs.
    env_exports = [
        f"export S2S_CHECKPOINT_DIR={shlex.quote(s2s_ckpt)}",
        f"export RUN_LABEL={shlex.quote(run_label)}",
        f"export FD3_PROVIDER={shlex.quote(provider)}",
        f"export FD3_RUN_ROOT={shlex.quote(run_root)}",
        f"export FD3_RESULTS_DIR={shlex.quote(per_sample_data_dir)}",
        f"export FD3_LOGS_DIR={shlex.quote(logs_dir)}",
        f"export FD3_REPORTS_DIR={shlex.quote(reports_dir)}",
        f"export PYTHONPATH={shlex.quote(drirf_path)}:${{PYTHONPATH:-}}",
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


def build_score_command(config: dict, force: bool = False) -> str:
    eval_results_dir = _eval_results_dir(config)
    fd3_run_root = _fd3_run_root(config)
    python_exec = (
        config.get("scoring_container_python_exec")
        or config.get("server_container_python_exec")
        or "python"
    )
    cmd_args = [
        f"{python_exec} {SCORING_SCRIPT}",
        f"--eval_results_dir {shlex.quote(eval_results_dir)}",
        f"--fd3_run_root {shlex.quote(fd3_run_root)}",
        f"--provider {shlex.quote(config.get('provider', 'fdb_v3_chen_chen'))}",
    ]
    if force:
        cmd_args.append("--force")
    return " ".join(cmd_args)


def run_fdb_v3_chen_chen_eval(config: dict, extra_passthrough: list[str]) -> None:
    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    print(f"FDB v3 ChenChen: benchmark={BENCHMARK}, output_dir={config['output_dir']}")
    expname = config.get("expname", "fdb_v3_chen_chen")
    eval_results_path = _eval_results_dir(config)
    fd3_run_root = _fd3_run_root(config)
    generation_submitted = False

    if not scoring_only:
        # The orchestrator writes model_report.json at FD3_RUN_ROOT once it completes.
        report_marker = Path(fd3_run_root) / "model_report.json"
        if report_marker.exists() and not config.get("force"):
            print(f"Skipping generation (found {report_marker})")
        else:
            print("Submitting upstream FD3 orchestrator (vLLM + Backend Agent + S2S)...")
            gen_command = build_generation_command(config, extra_passthrough)
            container = config.get("server_container") or config.get("container") or "nemo-skills"
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=gen_command,
                container=container,
                partition=config.get("partition"),
                num_gpus=config.get("num_gpus", 2),
                num_nodes=1,
                expname=expname,
                installation_command=config.get("installation_command"),
                log_dir=f"{eval_results_path}/orchestrator-logs",
                mount_paths=config.get("mount_paths"),
                time_min=config.get("time_min"),
                reuse_code=False,
                dry_run=dry_run,
            )
            generation_submitted = True

    if not generation_only:
        print("Submitting scoring (ingest upstream reports)...")
        score_command = build_score_command(config, force=config.get("scoring_force", False))
        scoring_container = (
            config.get("scoring_container") or config.get("server_container") or "nemo-skills"
        )
        scoring_gpus = config.get("scoring_gpus", 0)
        # GPU partitions reject 0-GPU jobs; fall back to cpu_partition for the
        # scoring step (it's pure JSON ingestion).
        scoring_partition = config.get("scoring_partition")
        if not scoring_partition:
            scoring_partition = config.get("cpu_partition") if scoring_gpus == 0 else config.get("partition")
        run_after = [expname] if generation_submitted else None
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
