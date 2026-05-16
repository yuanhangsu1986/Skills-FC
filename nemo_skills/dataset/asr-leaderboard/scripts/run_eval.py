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
Run HF Open ASR Leaderboard evaluation using nemo-skills pipeline.

Evaluates WER on 8 ASR datasets: librispeech_clean, librispeech_other,
voxpopuli, tedlium, gigaspeech, spgispeech, earnings22, ami.

The asr-leaderboard dataset uses METRICS_TYPE="audio" and task_type="ASR_LEADERBOARD",
so nemo-skills handles both generation and WER scoring automatically.

Prepare data first:
    ns prepare_data asr-leaderboard

Usage:
    python run_eval.py --config asr_leaderboard_s2s_incremental_v2_02mar_config.yaml
    python run_eval.py --config asr_leaderboard_s2s_incremental_v2_02mar_config.yaml --dry_run
    python run_eval.py --config asr_leaderboard_s2s_incremental_v2_02mar_config.yaml --generation_only
    python run_eval.py --config asr_leaderboard_s2s_incremental_v2_02mar_config.yaml --scoring_only
"""

import argparse

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import wrap_arguments


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_asr_leaderboard_eval(config: dict):
    """Run HF Open ASR Leaderboard evaluation pipeline."""
    benchmark = config.get("benchmark", "asr-leaderboard")
    expname = config.get("expname", "asr_leaderboard_eval")
    dry_run = config.get("dry_run", False)
    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)

    print(f"{'=' * 60}")
    print("HF Open ASR Leaderboard Evaluation")
    print(f"{'=' * 60}")
    split = config.get("split")

    print(f"Benchmark: {benchmark}")
    print(f"Model: {config['model']}")
    print(f"Output: {config['output_dir']}")
    if split:
        print(f"Split: {split} (single subset)")

    extra_args = []
    if config.get("max_samples"):
        extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        extra_args.append(f"++server.server_type={config['server_server_type']}")
    extra_args_str = " ".join(extra_args)

    eval_kwargs = dict(
        ctx=wrap_arguments(extra_args_str),
        cluster=config["cluster"],
        output_dir=config["output_dir"],
        data_dir=config.get("data_dir"),
        benchmarks=benchmark,
        model=config["model"],
        server_type=config.get("server_type", "vllm"),
        server_gpus=config.get("server_gpus", 1),
        server_nodes=config.get("server_nodes", 1),
        server_args=config.get("server_args", ""),
        server_entrypoint=config.get("server_entrypoint"),
        server_container=config.get("server_container"),
        partition=config.get("partition"),
        num_chunks=config.get("num_chunks", 1),
        installation_command=config.get("installation_command"),
        expname=expname,
        auto_summarize_results=True,
        dry_run=dry_run,
    )
    if config.get("chunk_ids"):
        eval_kwargs["chunk_ids"] = config["chunk_ids"]
    if split:
        eval_kwargs["split"] = split

    if not scoring_only:
        print("\n--- Running generation + scoring ---")
        nemo_eval(**eval_kwargs)

    if scoring_only:
        print("\n--- Running scoring only ---")
        nemo_eval(**eval_kwargs)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="HF Open ASR Leaderboard evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--num_chunks", type=int, help="Override num_chunks")
    parser.add_argument(
        "--split",
        help="Data split to evaluate (default: test = all datasets). "
        "Use a single dataset name (e.g. ami, earnings22) to run only that subset.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--generation_only", action="store_true", help="Only run generation")
    parser.add_argument("--scoring_only", action="store_true", help="Only run scoring")

    args = parser.parse_args()
    config = load_config(args.config)

    override_keys = ["cluster", "partition", "model", "output_dir", "max_samples", "num_chunks", "split"]
    for key in override_keys:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)

    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True

    run_asr_leaderboard_eval(config)


if __name__ == "__main__":
    main()
