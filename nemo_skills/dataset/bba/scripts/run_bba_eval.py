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
Run BigBench Audio evaluation.

Pipeline stages (each independently re-runnable):
  1. generation  — nemo-skills inference, produces output.jsonl
  2. merge       — merge output chunks into output.jsonl (if num_chunks > 1)
  3. asr         — Whisper transcription via vLLM server, produces output_asr.jsonl
  4. scoring     — exact-match + LLM-as-judge, produces metrics.json

Stage control flags:
  --generation_only   run stage 1 only
  --asr_only          run stage 3 only (requires output.jsonl)
  --scoring_only      run stage 4 only (requires output_asr.jsonl)
  (default)           run all stages end-to-end

Usage:
    python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
        --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.transcribe import transcribe_audio
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

ALL_CATEGORIES = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict, category: str, force: bool = False) -> str:
    eval_results_dir = f"{config['output_dir']}/eval-results/{category}"
    python_exec = config.get("scoring_container_python_exec", "python")
    cmd_args = [
        f"{python_exec} nemo_skills/dataset/bba/scripts/run_bba_scoring.py",
        f"--eval_results_dir {eval_results_dir}",
        f"--category {category}",
    ]
    if force:
        cmd_args.append("--force")
    if config.get("judge_model"):
        cmd_args.append(f"--judge_model {config['judge_model']}")
    if config.get("api_type"):
        cmd_args.append(f"--api_type {config['api_type']}")
    if config.get("judge_base_url"):
        cmd_args.append(f"--judge_base_url {config['judge_base_url']}")
    return " ".join(cmd_args)


def run_generation_stage(config: dict, category: str, expname: str, base_extra_args: list, dry_run: bool) -> bool:
    """Stage 1: Submit generation job. Returns True if a job was submitted."""
    eval_dir = Path(f"{config['output_dir']}/eval-results/{category}")
    output_jsonl = eval_dir / "output.jsonl"

    num_chunks = int(config.get("num_chunks", 1) or 1)
    if num_chunks > 1:
        chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
    else:
        chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or (eval_dir / "output.jsonl.done").exists()

    if output_jsonl.exists() and chunk_done_ok:
        print(f"\n--- Stage 1: Skipping generation (found {output_jsonl} and done markers) ---")
        return False

    print("\n--- Stage 1: Running generation ---")
    server_gpus = config.get("server_gpus", 1)
    partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
    nemo_eval(
        ctx=wrap_arguments(base_extra_args),
        cluster=config["cluster"],
        output_dir=config["output_dir"],
        benchmarks=category,
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
    return True


def run_merge_stage(config: dict, eval_results_path: str, expname: str, generation_submitted: bool, dry_run: bool):
    """Stage 2: Merge output chunks if needed. Returns run_after list for downstream stages."""
    return maybe_merge_before_scoring(
        config, eval_results_path, expname,
        run_after=[expname] if generation_submitted else None,
        dry_run=dry_run,
    )


def run_asr_stage(config: dict, category: str, expname: str, eval_results_path: str, run_after, dry_run: bool):
    """Stage 3: Submit Whisper ASR job via vLLM server. Returns run_after list for scoring."""
    output_asr_jsonl = Path(eval_results_path) / "output_asr.jsonl"
    force = config.get("scoring_force", False)

    if output_asr_jsonl.exists() and not force:
        print(f"\n--- Stage 3: Skipping ASR (found {output_asr_jsonl}) ---")
        return run_after

    print("\n--- Stage 3: Running ASR (Whisper via vLLM) ---")
    asr_expname = f"{expname}_asr"
    transcribe_audio(
        cluster=config["cluster"],
        input_jsonl=f"{eval_results_path}/output.jsonl",
        output_jsonl=f"{eval_results_path}/output_asr.jsonl",
        asr_model=config["asr_model"],
        asr_server_args=config.get("asr_server_args", ""),
        data_dir=config.get("data_dir"),
        server_gpus=config.get("asr_server_gpus", 1),
        server_container=config.get("server_container"),
        partition=config.get("partition"),
        expname=asr_expname,
        run_after=run_after,
        installation_command=config.get("asr_installation_command"),
        log_dir=f"{eval_results_path}/summarized-results",
        force=force,
        reuse_code=False,
        dry_run=dry_run,
    )
    return [asr_expname]


def run_scoring_stage(config: dict, category: str, expname: str, eval_results_path: str, run_after, dry_run: bool):
    """Stage 4: Submit scoring job."""
    print("\n--- Stage 4: Running scoring ---")
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_score_command(config, category, force=config.get("scoring_force", False)),
        container=config.get("scoring_container") or "nemo-skills",
        partition=config.get("cpu_partition") or config.get("partition"),
        run_after=run_after,
        expname=f"{expname}_score",
        installation_command=config.get("scoring_installation_command"),
        log_dir=f"{eval_results_path}/summarized-results",
        reuse_code=False,
        dry_run=dry_run,
    )


def build_aggregate_command(config: dict, categories: list, force: bool = False) -> str:
    """Build the stage-5 aggregation command."""
    python_exec = (config.get("scoring_container_python_exec")
                   or config.get("server_container_python_exec")
                   or "python")
    cmd_args = [
        f"{python_exec} nemo_skills/dataset/bba/scripts/run_bba_aggregate.py",
        f"--output_dir {config['output_dir']}",
        f"--categories {' '.join(categories)}",
    ]
    if force:
        cmd_args.append("--force")
    return " ".join(cmd_args)


def run_aggregate_stage(config: dict, categories: list, force: bool = False):
    """Stage 5 (inline): run aggregation directly when --aggregate_only is set."""
    from nemo_skills.dataset.bba.scripts.run_bba_aggregate import aggregate
    aggregate(config["output_dir"], categories, force=force)


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
    asr_only = config.get("asr_only", False)
    scoring_only = config.get("scoring_only", False)
    aggregate_only = config.get("aggregate_only", False)
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
    if config.get("inference_overrides"):
        base_extra_args.extend(config["inference_overrides"].strip().split())

    score_expnames = []
    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print(f"{'=' * 60}")

        expname = f"{config.get('expname', 'bba')}_{category}"
        eval_results_path = f"{config['output_dir']}/eval-results/{category}"

        if aggregate_only:
            continue

        generation_submitted = False
        if not (scoring_only or asr_only):
            generation_submitted = run_generation_stage(config, category, expname, base_extra_args, dry_run)
            if generation_only:
                continue

        merge_run_after = None
        if not (asr_only or scoring_only):
            merge_run_after = run_merge_stage(config, eval_results_path, expname, generation_submitted, dry_run)

        if not scoring_only:
            asr_run_after = run_asr_stage(config, category, expname, eval_results_path, merge_run_after, dry_run)
            if asr_only:
                continue
        else:
            asr_run_after = None

        run_scoring_stage(config, category, expname, eval_results_path, asr_run_after, dry_run)
        score_expnames.append(f"{expname}_score")

    # Stage 5: aggregate across all scored categories
    if not generation_only and not asr_only:
        if aggregate_only:
            # Results already exist on disk — run inline.
            run_aggregate_stage(config, categories, force=config.get("scoring_force", False))
        else:
            # Submit as a Slurm job that depends on all scoring jobs completing.
            agg_expname = f"{config.get('expname', 'bba')}_aggregate"
            agg_log_dir = str(Path(config["output_dir"]) / "eval-results" / "bba_aggregate" / "summarized-results")
            print("\n--- Stage 5: Submitting aggregate job ---")
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=build_aggregate_command(config, categories, force=config.get("scoring_force", False)),
                container=config.get("scoring_container") or config.get("server_container") or "nemo-skills",
                partition=config.get("cpu_partition") or config.get("partition"),
                run_after=score_expnames if score_expnames else None,
                expname=agg_expname,
                installation_command=config.get("scoring_installation_command"),
                log_dir=agg_log_dir,
                reuse_code=False,
                dry_run=dry_run,
            )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run BigBench Audio evaluation (generate + ASR + score)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--categories", help="Override categories (comma-separated)")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--generation_only", action="store_true", help="Run stage 1 (generation) only")
    parser.add_argument("--asr_only", action="store_true", help="Run stage 3 (ASR) only")
    parser.add_argument("--scoring_only", action="store_true", help="Run stage 4 (scoring) only")
    parser.add_argument("--aggregate_only", action="store_true", help="Run stage 5 (aggregation) only")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run ASR and scoring even if outputs exist")

    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["cluster", "partition", "model", "output_dir", "categories", "max_samples"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.asr_only:
        config["asr_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if args.aggregate_only:
        config["aggregate_only"] = True
    if args.scoring_force:
        config["scoring_force"] = True

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_bba_eval(config)


if __name__ == "__main__":
    main()
