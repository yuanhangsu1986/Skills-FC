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

"""
Run BFCL single-turn function-channel evaluation.

Pipeline stages per category:
  1. inference  — starts serve_unified (with function-channel flags) in
                  background on the same GPU node, then runs
                  run_bfcl_fc_inference.py; produces output.jsonl
  2. scoring    — runs run_bfcl_fc_scoring.py; produces metrics.json

Both stages are submitted as Slurm jobs via run_cmd, with stage 2 depending
on stage 1 for each category.

Prerequisites:
  - prepare.py has already been run to produce <data_dir>/<category>/input.jsonl
    and <data_dir>/<category>/audio/*.wav.

Usage:
    python nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
        --config nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config.yaml
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir
from nemo_skills.pipeline.utils.server import get_free_port

from nemo_skills.dataset.bfcl_single_turn_function_channel.constants import SINGLE_TURN_CATEGORIES


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_infer_command(config: dict, category: str) -> str:
    """Build a shell command that starts serve_unified in the background on the
    same node, then runs the inference client, then cleans up.

    The inference client polls for server readiness via --poll_interval /
    --max_poll_attempts, so no additional health-check wrapper is needed.
    """
    model = config["model"]
    server_args = config.get("server_args", "")
    # Pick a random high port per submission. Hardcoding port 8000 caused
    # collisions when SLURM packed multiple bfcl jobs onto the same node:
    # the second job's serve_unified failed to bind, while its client could
    # still see the first job's server on 8000 and silently send requests
    # there. This matches nemo_eval's strategy (pipeline/utils/server.py
    # should_get_random_port + get_free_port) used by every other benchmark.
    port = config.get("server_port") or get_free_port(strategy="random")
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    poll_interval = config.get("poll_interval", 30)
    max_poll_attempts = config.get("max_poll_attempts", 40)
    request_timeout = config.get("request_timeout", 300)

    input_jsonl = f"{data_dir}/{category}/input.jsonl"
    output_jsonl = f"{output_dir}/eval-results/{category}/output.jsonl"

    python_exec = config.get("server_container_python_exec", "python")
    serve_cmd = (
        f"{python_exec} -m nemo_skills.inference.server.serve_unified"
        f" --model {model}"
        f" --port {port}"
        f" {server_args}"
    )

    max_workers = config.get("max_workers", 2)
    max_tokens = config.get("max_tokens", 256)

    infer_cmd = (
        f"{python_exec} nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_bfcl_fc_inference.py"
        f" --server_url http://localhost:{port}"
        f" --input_jsonl {input_jsonl}"
        f" --output_jsonl {output_jsonl}"
        f" --poll_interval {poll_interval}"
        f" --max_poll_attempts {max_poll_attempts}"
        f" --request_timeout {request_timeout}"
        f" --max_workers {max_workers}"
        f" --max_tokens {max_tokens}"
    )

    # MULTI_TURN_FUNCTION_CALLS_ALLOWED: opts the DRIRF wrapper into the
    # multi-turn function-call code path (see _multi_turn_fc_enabled in
    # nemotron_voicechat_inference_wrapper.py). With this set, the
    # wrapper's inference loop does NOT stop at the first EOTC token —
    # required for any benchmark that may need >1 tool-call turn per
    # request (e.g. fdb_v3 agent dispatch). For single-call categories
    # (simple/multiple) it's a no-op: there's still only one EOTC, the
    # loop just runs to total_frames after it. Config-toggleable; default
    # true since BFCL is a tool-calling benchmark.
    multi_turn_fc = bool(config.get("multi_turn_function_calls_allowed", True))
    env_prefix = f"export MULTI_TURN_FUNCTION_CALLS_ALLOWED=true && " if multi_turn_fc else ""

    # Wrap server+client in a compound command { } so that any installation_command
    # prepended by install_packages_wrap (via "&&") runs synchronously before either
    # process starts.  Without the braces, bash operator precedence (&&  before  &)
    # would group the install guard with the server into the background job, leaving
    # the client to run immediately in the foreground before the guard completes.
    # The explicit "cd /nemo_run/code &&" is still needed before infer_cmd because
    # get_cmd prepends its own "cd /nemo_run/code &&" which gets absorbed into the
    # background server side inside the braces.
    # The env-var export sits OUTSIDE the braces so it applies to both the
    # backgrounded server process and the foreground client.
    return (
        f"{env_prefix}"
        f"{{ {serve_cmd} & "
        f"cd /nemo_run/code && {infer_cmd}; "
        f"_BFCL_EXIT=$?; kill %1 2>/dev/null; wait %1 2>/dev/null; exit $_BFCL_EXIT; }}"
    )


def build_score_command(config: dict, category: str, force: bool = False) -> str:
    output_dir = config["output_dir"]
    output_jsonl = f"{output_dir}/eval-results/{category}/output.jsonl"
    python_exec = config.get("scoring_container_python_exec", "python")
    cmd = (
        f"{python_exec} nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_bfcl_fc_scoring.py"
        f" --output_jsonl {output_jsonl}"
        f" --category {category}"
    )
    if force:
        cmd += " --force"
    return cmd


def run_inference_stage(config: dict, category: str, expname: str, dry_run: bool) -> bool:
    """Submit the serve+inference job. Returns True if submitted."""
    output_jsonl = Path(f"{config['output_dir']}/eval-results/{category}/output.jsonl")

    if output_jsonl.exists() and not config.get("scoring_force", False):
        print(f"\n--- Stage 1: Skipping inference (found {output_jsonl}) ---")
        return False

    print("\n--- Stage 1: Running inference (serve + infer) ---")
    log_dir = str(Path(config["output_dir"]) / "eval-results" / category / "summarized-results")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_infer_command(config, category),
        container=config.get("server_container"),
        num_gpus=config.get("server_gpus", 1),
        partition=config.get("partition"),
        expname=expname,
        installation_command=config.get("installation_command"),
        log_dir=log_dir,
        reuse_code=False,
        dry_run=dry_run,
        exclusive=config.get("exclusive"),
    )
    return True


def run_scoring_stage(config: dict, category: str, expname: str, run_after, dry_run: bool):
    """Submit the scoring job."""
    print("\n--- Stage 2: Running scoring ---")
    log_dir = str(Path(config["output_dir"]) / "eval-results" / category / "summarized-results")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_score_command(config, category, force=config.get("scoring_force", False)),
        container=config.get("scoring_container") or "nemo-skills",
        partition=config.get("cpu_partition") or config.get("partition"),
        run_after=run_after,
        expname=f"{expname}_score",
        installation_command=config.get("scoring_installation_command"),
        log_dir=log_dir,
        reuse_code=False,
        dry_run=dry_run,
    )


def run_bfcl_eval(config: dict):
    categories_cfg = config.get("categories", "all")
    if categories_cfg == "all":
        categories = list(SINGLE_TURN_CATEGORIES)
    elif isinstance(categories_cfg, str):
        categories = [c.strip() for c in categories_cfg.split(",")]
    else:
        categories = list(categories_cfg)

    categories = [c for c in categories if c in SINGLE_TURN_CATEGORIES]
    if not categories:
        raise ValueError(f"No valid categories specified. Valid: {SINGLE_TURN_CATEGORIES}")

    inference_only = config.get("inference_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    print(f"Output directory: {config['output_dir']}")

    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print(f"{'=' * 60}")

        expname = f"{config.get('expname', 'bfcl_fc')}_{category}"

        infer_submitted = False
        if not scoring_only:
            infer_submitted = run_inference_stage(config, category, expname, dry_run)
            if inference_only:
                continue

        score_run_after = [expname] if infer_submitted else None
        run_scoring_stage(config, category, expname, score_run_after, dry_run)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run BFCL single-turn function-channel evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--data_dir", help="Override data directory (prepared input.jsonl)")
    parser.add_argument("--categories", help="Override categories (comma-separated)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--inference_only", action="store_true", help="Run stage 1 (inference) only")
    parser.add_argument("--scoring_only", action="store_true", help="Run stage 2 (scoring) only")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run scoring even if output exists")

    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["cluster", "partition", "model", "output_dir", "data_dir", "categories"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.inference_only:
        config["inference_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if args.scoring_force:
        config["scoring_force"] = True

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_bfcl_eval(config)


if __name__ == "__main__":
    main()
