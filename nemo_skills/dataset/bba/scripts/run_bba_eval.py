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
Run BigBench Audio evaluation: generate responses with nemo-skills, then score with exact match.

Usage:
    python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
        --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import isolate_job_dir

ALL_CATEGORIES = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict, category: str, force: bool = False) -> str:
    eval_results_dir = f"{config['output_dir']}/eval-results/{category}"
    scoring_script = "nemo_skills/dataset/bba/scripts/run_bba_scoring.py"
    cmd_args = [
        f"python {scoring_script}",
        f"--eval_results_dir {eval_results_dir}",
        f"--category {category}",
    ]
    if force:
        cmd_args.append("--force")
    if config.get("judge_model"):
        cmd_args.append(f"--judge_model {config['judge_model']}")
    if config.get("api_type"):
        cmd_args.append(f"--api_type {config['api_type']}")
    return " ".join(cmd_args)


def run_bba_eval(config: dict):
    categories_cfg = config.get("categories", "all")
    if categories_cfg == "all":
        categories = ALL_CATEGORIES
    elif isinstance(categories_cfg, str):
        categories = [c.strip() for c in categories_cfg.split(",")]
    else:
        categories = list(categories_cfg)

    categories = [c for c in categories if c in ALL_CATEGORIES]
    if not categories:
        raise ValueError("No valid categories specified")

    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    print(f"Output directory: {config['output_dir']}")

    base_extra_args = ["++eval_type=null"]
    if config.get("max_samples"):
        base_extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        base_extra_args.append(f"++server.server_type={config['server_server_type']}")
    if config.get("system_message"):
        base_extra_args.append(f"++system_message='{config['system_message']}'")

    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print(f"{'=' * 60}")

        benchmark = category
        expname = f"{config.get('expname', 'bba')}_{category}"
        eval_results_path = f"{config['output_dir']}/eval-results/{category}"
        eval_dir = Path(eval_results_path)
        output_jsonl = eval_dir / "output.jsonl"
        output_jsonl_done = eval_dir / "output.jsonl.done"

        num_chunks = int(config.get("num_chunks", 1) or 1)
        if num_chunks > 1:
            chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
        else:
            chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or output_jsonl_done.exists()
        generation_complete = output_jsonl.exists() and chunk_done_ok
        generation_submitted = False

        if not scoring_only:
            if generation_complete:
                print(f"\n--- Skipping generation (found {output_jsonl} and done markers) ---")
            else:
                print("\n--- Running generation ---")
                server_gpus = config.get("server_gpus", 1)
                partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
                nemo_eval(
                    ctx=wrap_arguments(base_extra_args),
                    cluster=config["cluster"],
                    output_dir=config["output_dir"],
                    benchmarks=benchmark,
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
                    dry_run=dry_run,
                )
                generation_submitted = True

        if not generation_only:
            score_run_after = maybe_merge_before_scoring(
                config, eval_results_path, expname,
                run_after=[expname] if generation_submitted else None,
                dry_run=dry_run,
            )
            print("\n--- Running scoring ---")
            score_command = build_score_command(config, category, force=config.get("scoring_force", False))
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=score_command,
                partition=config.get("cpu_partition") or config.get("partition"),
                run_after=score_run_after,
                expname=f"{expname}_score",
                installation_command=config.get("scoring_installation_command"),
                log_dir=f"{eval_results_path}/summarized-results",
                dry_run=dry_run,
            )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run BigBench Audio evaluation (generate + score)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--categories", help="Override categories (comma-separated)")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--generation_only", action="store_true")
    parser.add_argument("--scoring_only", action="store_true")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run scoring even if metrics.json exists")

    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["cluster", "partition", "model", "output_dir", "categories", "max_samples"]:
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

    isolate_job_dir(config)
    run_bba_eval(config)


if __name__ == "__main__":
    main()
