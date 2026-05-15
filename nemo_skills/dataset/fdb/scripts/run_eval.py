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
Run Full-Duplex-Bench evaluation: generate responses with nemo-skills, then score with FDB metrics.

Usage:
    python run_eval.py --config fdb_s2s_offline_config.yaml
"""

import argparse
import shlex
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

ALL_SUBTESTS = [
    "pause_candor",
    "pause_synthetic",
    "backchannel",
    "turn_taking",
    "interruption",
]
# v1.5 has different subtasks (overlap-focused)
ALL_SUBTESTS_V1_5 = [
    "background_speech",
    "talking_to_other",
    "backchannel",
    "interruption",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _benchmark_name(subtest: str, fdb_version: str) -> str:
    """Return benchmark name for eval-results path and nemo-skills (e.g. fdb_v1.pause or fdb_v1_5.pause)."""
    if fdb_version == "v1.5":
        return f"fdb_v1_5.{subtest}"
    return f"fdb_v1.{subtest}"


def build_score_command(config: dict, subtest: str, force: bool = False) -> str:
    """Build the scoring command to run via run_cmd.

    Uses run_fdb_scoring.py to create output compatible with nemo-skills:
    - summarized-results/ directory with logs
    - metrics.json with evaluation results
    """
    fdb_version = config.get("fdb_version", "v1.0")
    benchmark = _benchmark_name(subtest, fdb_version)
    eval_results_dir = f"{config['output_dir']}/eval-results/{benchmark}"
    fdb_repo = config["fdb_repo_path"]
    scoring_script = "nemo_skills/dataset/fdb/scripts/run_fdb_scoring.py"

    python_exec = (config.get("scoring_container_python_exec")
                   or config.get("server_container_python_exec")
                   or "python")
    cmd_args = [
        f"{python_exec} {scoring_script}",
        f"--eval_results_dir {eval_results_dir}",
        f"--fdb_repo {fdb_repo}",
        f"--subtest {subtest}",
        f"--fdb_version {fdb_version}",
    ]
    fdb_data_path = config.get("fdb_data_path")
    if fdb_data_path:
        cmd_args.append(f"--fdb_data_path {shlex.quote(str(fdb_data_path))}")
    silero_vad_dir = config.get("silero_vad_dir")
    if silero_vad_dir:
        cmd_args.append(f"--silero_vad_dir {shlex.quote(str(silero_vad_dir))}")
    if force:
        cmd_args.append("--force")

    return " ".join(cmd_args)


def run_fdb_eval(config: dict):
    """Run Full-Duplex-Bench evaluation using direct Python calls."""

    # Parse subtests
    fdb_version = config.get("fdb_version", "v1.0")
    valid_subtests = ALL_SUBTESTS_V1_5 if fdb_version == "v1.5" else ALL_SUBTESTS
    subtests_cfg = config.get("subtests", "all")
    if subtests_cfg == "all":
        subtests = list(valid_subtests)
    elif isinstance(subtests_cfg, str):
        subtests = [s.strip() for s in subtests_cfg.split(",")]
    else:
        subtests = subtests_cfg

    subtests = [s for s in subtests if s in valid_subtests]
    if not subtests:
        raise ValueError("No valid subtests specified")

    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    print(f"Processing {len(subtests)} subtests: {', '.join(subtests)}")
    print(f"Output directory: {config['output_dir']}")

    # Build base extra args for hydra overrides
    # Skip native evaluation for all subtests - FDB scorer handles evaluation
    base_extra_args = ["++eval_type=null"]
    if config.get("max_samples"):
        base_extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        base_extra_args.append(f"++server.server_type={config['server_server_type']}")
    if config.get("api_key_env_var"):
        base_extra_args.append(f"++server.api_key_env_var={config['api_key_env_var']}")
    if config.get("inference_overrides"):
        base_extra_args.extend(config["inference_overrides"].strip().split())

    for subtest in subtests:
        extra_args_str = " ".join(base_extra_args)
        print(f"\n{'=' * 60}")
        print(f"Processing subtest: {subtest} (FDB {fdb_version})")
        print(f"{'=' * 60}")

        benchmark = _benchmark_name(subtest, fdb_version)
        expname = f"{config.get('expname', 'fdb')}_{subtest}"
        eval_results_path = f"{config['output_dir']}/eval-results/{benchmark}"
        output_jsonl = Path(eval_results_path) / "output.jsonl"
        output_jsonl_done = Path(eval_results_path) / "output.jsonl.done"
        generation_submitted = False

        # Generation phase
        if not scoring_only:
            num_chunks = int(config.get("num_chunks", 1) or 1)
            eval_dir = Path(eval_results_path)
            if num_chunks > 1:
                chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
            else:
                chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or output_jsonl_done.exists()
            generation_complete = output_jsonl.exists() and chunk_done_ok

            if generation_complete:
                print(f"\n--- Skipping generation (found {output_jsonl} and done markers) ---")
            else:
                print("\n--- Running generation ---")
                server_gpus = config.get("server_gpus", 1)
                # Use cpu_partition when not self-hosting (external API)
                partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
                nemo_eval(
                    ctx=wrap_arguments(extra_args_str),
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
                    reuse_code=False,
                    dry_run=dry_run,
                )
                generation_submitted = True

        # Scoring phase
        if not generation_only:
            score_run_after = maybe_merge_before_scoring(
                config, eval_results_path, expname,
                run_after=[expname] if generation_submitted else None,
                dry_run=dry_run,
            )
            print("\n--- Running scoring ---")
            score_command = build_score_command(config, subtest, force=config.get("scoring_force", False))
            # FDB scoring runs get_transcript/asr.py which requires NeMo + CUDA; use GPU partition and 1 GPU
            scoring_container = config.get("scoring_container") or config.get("server_container") or "nemo-skills"
            scoring_gpus = config.get("scoring_gpus", 1)  # ASR needs GPU; default 1
            scoring_partition = config.get("scoring_partition") or config.get("partition")  # GPU partition, not cpu_partition
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

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run Full-Duplex-Bench evaluation (generate + score)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    # CLI overrides
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--subtests", help="Override subtests (comma-separated)")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--generation_only", action="store_true", help="Only run generation")
    parser.add_argument("--scoring_only", action="store_true", help="Only run scoring")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run scoring even if metrics.json exists")

    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    for key in ["cluster", "partition", "model", "output_dir", "subtests", "max_samples"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if getattr(args, "scoring_force", False):
        config["scoring_force"] = True

    # Append commit hash to output_dir so each code version gets its own directory.
    # Same commit reuses the same directory (outputs cached); new commit gets a fresh one.
    # Skip when scoring_only so we operate on the existing directory.
    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_fdb_eval(config)


if __name__ == "__main__":
    main()
