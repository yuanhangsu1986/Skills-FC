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

"""Run FDB v3 (FD3) evaluation: generate responses with nemo-skills, then score
with the vendored FDBV3 evaluators.

Usage:
    python nemo_skills/dataset/fdb/scripts/fdb_v3/run_eval.py \
        --config nemo_skills/dataset/fdb/scripts/fdb_v3/fdb_v3_config_fc_greedy.yaml
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

BENCHMARK = "fdb_v3.tool_call"
SCORING_SCRIPT = "nemo_skills/dataset/fdb/scripts/fdb_v3/run_scoring.py"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict, force: bool = False) -> str:
    eval_results_dir = f"{config['output_dir']}/eval-results/{BENCHMARK}"
    fdb_repo = config["fdb_repo_path"]
    python_exec = (
        config.get("scoring_container_python_exec")
        or config.get("server_container_python_exec")
        or "python"
    )
    cmd_args = [
        f"{python_exec} {SCORING_SCRIPT}",
        f"--eval_results_dir {shlex.quote(str(eval_results_dir))}",
        f"--fdb_repo {shlex.quote(str(fdb_repo))}",
        f"--provider {shlex.quote(config.get('provider', 'fdb_v3'))}",
    ]
    if config.get("use_llm_judge"):
        cmd_args.append("--use_llm_judge")
    if config.get("skip_latency"):
        cmd_args.append("--skip_latency")
    if config.get("skip_asr"):
        cmd_args.append("--skip_asr")
    if force:
        cmd_args.append("--force")
    return " ".join(cmd_args)


def run_fdb_v3_eval(config: dict) -> None:
    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    print(f"FDB v3: benchmark={BENCHMARK}, output_dir={config['output_dir']}")

    base_extra_args = ["++eval_type=null"]
    if config.get("max_samples"):
        base_extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        base_extra_args.append(f"++server.server_type={config['server_server_type']}")
    if config.get("api_key_env_var"):
        base_extra_args.append(f"++server.api_key_env_var={config['api_key_env_var']}")
    if config.get("inference_overrides"):
        base_extra_args.extend(config["inference_overrides"].strip().split())

    extra_args_str = " ".join(base_extra_args)
    expname = config.get("expname", "fdb_v3_tool_call")
    eval_results_path = f"{config['output_dir']}/eval-results/{BENCHMARK}"
    output_jsonl = Path(eval_results_path) / "output.jsonl"
    output_jsonl_done = Path(eval_results_path) / "output.jsonl.done"
    generation_submitted = False

    if not scoring_only:
        num_chunks = int(config.get("num_chunks", 1) or 1)
        eval_dir = Path(eval_results_path)
        if num_chunks > 1:
            chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
        else:
            chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or output_jsonl_done.exists()
        generation_complete = output_jsonl.exists() and chunk_done_ok

        if generation_complete:
            print(f"Skipping generation (found {output_jsonl} and done markers)")
        else:
            print("Running generation...")
            server_gpus = config.get("server_gpus", 1)
            partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
            nemo_eval(
                ctx=wrap_arguments(extra_args_str),
                cluster=config["cluster"],
                output_dir=config["output_dir"],
                benchmarks=BENCHMARK,
                model=config["model"],
                server_type=config.get("server_type", "vllm"),
                server_gpus=server_gpus,
                server_address=config.get("server_address"),
                num_chunks=config.get("num_chunks", 1),
                server_container=config.get("server_container"),
                server_entrypoint=config.get("server_entrypoint"),
                data_dir=config.get("data_dir"),
                server_args=config.get("server_args", ""),
                installation_command=config.get("installation_command"),
                partition=partition,
                expname=expname,
                auto_summarize_results=False,
                reuse_code=False,
                dry_run=dry_run,
            )
            generation_submitted = True

    if not generation_only:
        score_run_after = maybe_merge_before_scoring(
            config,
            eval_results_path,
            expname,
            run_after=[expname] if generation_submitted else None,
            dry_run=dry_run,
        )
        print("Running scoring...")
        score_command = build_score_command(config, force=config.get("scoring_force", False))
        scoring_container = config.get("scoring_container") or config.get("server_container") or "nemo-skills"
        scoring_gpus = config.get("scoring_gpus", 1)
        scoring_partition = config.get("scoring_partition") or config.get("partition")
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=score_command,
            container=scoring_container,
            partition=scoring_partition,
            num_gpus=scoring_gpus,
            run_after=score_run_after,
            expname=f"{expname}_score",
            installation_command=config.get("scoring_installation_command"),
            log_dir=f"{eval_results_path}/summarized-results",
            reuse_code=False,
            dry_run=dry_run,
        )

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full-Duplex-Bench v3 (FD3) evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--generation_only", action="store_true")
    parser.add_argument("--scoring_only", action="store_true")
    parser.add_argument("--scoring_force", action="store_true")
    parser.add_argument("--use_llm_judge", action="store_true", help="Enable FD3 LLM judge for response quality")
    parser.add_argument("--skip_latency", action="store_true", help="Skip audio-derived latency in scoring")
    parser.add_argument(
        "--skip_asr",
        action="store_true",
        help="Skip Parakeet ASR on output audio (response_qual will degrade — debug only)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ["cluster", "partition", "model", "output_dir", "max_samples"]:
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
    if args.use_llm_judge:
        config["use_llm_judge"] = True
    if args.skip_latency:
        config["skip_latency"] = True
    if args.skip_asr:
        config["skip_asr"] = True

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_fdb_v3_eval(config)


if __name__ == "__main__":
    main()
