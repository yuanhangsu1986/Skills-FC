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
Generate VoiceBench responses using nemo-skills and score with official VoiceBench package.

Usage:
    python generate_from_api_and_score_official.py --config voicebench_eval_config.yaml
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

ALL_SUBTESTS = [
    "advbench",
    "alpacaeval",
    "alpacaeval_full",
    "alpacaeval_speaker",
    "bbh",
    "commoneval",
    "ifeval",
    "mmsu",
    "mtbench",
    "openbookqa",
    "sd_qa",
    "sd_qa_usa",
    "wildvoice",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict, subtest: str, force: bool = False) -> str:
    """Build the scoring command to run via run_cmd.

    Uses run_voicebench_scoring.py to create output compatible with nemo-skills:
    - summarized-results/ directory with logs
    - metrics.json with evaluation results
    """
    try:
        from convert_to_voicebench_format import REQUIRES_GPT_JUDGE, SUBTEST_TO_EVALUATOR
    except ImportError as e:
        raise ImportError(
            "Could not import convert_to_voicebench_format. "
            "Make sure to run this script from the voicebench scripts directory "
            "or add it to PYTHONPATH."
        ) from e

    eval_results_dir = f"{config['output_dir']}/eval-results/voicebench.{subtest}"
    evaluator = SUBTEST_TO_EVALUATOR.get(subtest, "open")
    needs_judge = subtest in REQUIRES_GPT_JUDGE
    voicebench_repo = config["voicebench_repo_path"]
    scoring_script = "nemo_skills/dataset/voicebench/scripts/run_voicebench_scoring.py"

    python_exec = config.get("scoring_container_python_exec", "python")
    decoding_mode = config.get("decoding_mode", "greedy")
    cmd_args = [
        f"{python_exec} {scoring_script}",
        f"--eval_results_dir {eval_results_dir}",
        f"--voicebench_repo {voicebench_repo}",
        f"--subtest {subtest}",
        f"--evaluator {evaluator}",
        f"--decoding_mode {decoding_mode}",
    ]

    if needs_judge:
        cmd_args.append("--needs_judge")
        if config.get("api_type"):
            cmd_args.append(f"--api_type {config['api_type']}")
        if config.get("nvidia_model"):
            cmd_args.append(f"--nvidia_model {config['nvidia_model']}")
    if force:
        cmd_args.append("--force")

    return " ".join(cmd_args)


def build_agent_audio_asr_command(config: dict, subtest: str) -> str:
    """Build the agent-audio ASR + WER/CER command to run via run_cmd."""
    eval_results_dir = f"{config['output_dir']}/eval-results/voicebench.{subtest}"
    asr_script = "nemo_skills/dataset/voicebench/scripts/run_agent_audio_asr_metrics.py"
    asr_model = config.get("agent_audio_asr_model", "nvidia/parakeet-tdt-0.6b-v2")

    env_prefix = ""
    torch_home = config.get("torch_home", "")
    if torch_home:
        env_prefix = f"export TORCH_HOME={torch_home} && "

    python_exec = config.get("server_container_python_exec", "python")
    cmd_args = [
        f"{env_prefix}{python_exec} {asr_script}",
        f"--eval_results_dir {eval_results_dir}",
        f"--subtest {subtest}",
        "--input_jsonl output.jsonl",
        "--output_jsonl output_asr.jsonl",
        f"--asr_model {asr_model}",
    ]
    if config.get("agent_audio_force", False):
        cmd_args.append("--force")

    return " ".join(cmd_args)


def run_voicebench_eval(config: dict):
    """Run VoiceBench evaluation using direct Python calls."""

    # Parse subtests
    subtests_cfg = config.get("subtests", "all")
    if subtests_cfg == "all":
        subtests = ALL_SUBTESTS
    elif isinstance(subtests_cfg, str):
        subtests = [s.strip() for s in subtests_cfg.split(",")]
    else:
        subtests = subtests_cfg

    subtests = [s for s in subtests if s in ALL_SUBTESTS]
    if not subtests:
        raise ValueError("No valid subtests specified")

    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    agent_audio_stage_enabled_cfg = config.get("agent_audio_stage_enabled")
    if agent_audio_stage_enabled_cfg is None:
        # Auto-enable if server_args requests audio generation.
        agent_audio_stage_enabled = "--decode_audio" in (config.get("server_args") or "")
    else:
        agent_audio_stage_enabled = bool(agent_audio_stage_enabled_cfg)

    print(f"Processing {len(subtests)} subtests: {', '.join(subtests)}")
    print(f"Output directory: {config['output_dir']}")

    # Build base extra args for hydra overrides
    # Skip native evaluation for all subtests - VoiceBench scorer handles evaluation
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
        print(f"Processing subtest: {subtest}")
        print(f"{'=' * 60}")

        expname = f"{config.get('expname', 'voicebench')}_{subtest}"
        benchmark = f"voicebench.{subtest}"
        eval_results_path = f"{config['output_dir']}/eval-results/voicebench.{subtest}"
        eval_dir = Path(eval_results_path)
        output_jsonl = eval_dir / "output.jsonl"
        output_jsonl_done = eval_dir / "output.jsonl.done"

        # If output.jsonl and done markers exist, nemo-skills will skip scheduling any generation tasks.
        # In that case, downstream stages must not depend on `expname`.
        num_chunks = int(config.get("num_chunks", 1) or 1)
        if num_chunks > 1:
            chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
        else:
            chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or output_jsonl_done.exists()
        generation_complete = output_jsonl.exists() and chunk_done_ok
        generation_submitted = False

        # Generation phase
        if not scoring_only:
            if generation_complete:
                print(f"\n--- Skipping generation (found {output_jsonl} and done markers) ---")
            else:
                print("\n--- Running generation ---")
                server_gpus = config.get("server_gpus", 1)
                # Use cpu_partition when not self-hosting (external API)
                partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
                gen_exp = nemo_eval(
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
                generation_submitted = gen_exp is not None

        # Gate all scoring stages on merge completing if output.jsonl is missing.
        pre_scoring_run_after = (
            maybe_merge_before_scoring(
                config, eval_results_path, expname,
                run_after=[expname] if generation_submitted else None,
                dry_run=dry_run,
            )
            if not generation_only
            else None
        )

        # Agent-audio ASR + WER/CER phase (optional)
        agent_audio_expname = f"{expname}_agent_audio_asr"
        if not generation_only and agent_audio_stage_enabled:
            print("\n--- Running agent-audio ASR + WER/CER ---")
            asr_command = build_agent_audio_asr_command(config, subtest)
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=asr_command,
                container=config.get("server_container") or "nemo-skills",
                partition=config.get("partition"),
                num_gpus=1,
                run_after=pre_scoring_run_after,
                expname=agent_audio_expname,
                installation_command=config.get("agent_audio_installation_command"),
                log_dir=f"{eval_results_path}/summarized-results",
                reuse_code=False,
                dry_run=dry_run,
            )

        # Scoring phase (Stage 3: generated text, Stage 4: agent ASR transcript)
        if not generation_only:
            scoring_container = config.get("scoring_container") or "nemo-skills"

            # Stage 3: VoiceBench scoring on generated text (output.jsonl)
            print("\n--- Running scoring (generated text) ---")
            score_command_generated = f"{build_score_command(config, subtest, force=config.get('scoring_force', False))} --input_jsonl output.jsonl --metrics_variant generated"
            score_generated_expname = f"{expname}_score_generated"
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=score_command_generated,
                container=scoring_container,
                partition=config.get("cpu_partition") or config.get("partition"),
                run_after=pre_scoring_run_after,
                expname=score_generated_expname,
                installation_command=config.get("scoring_installation_command"),
                log_dir=f"{eval_results_path}/summarized-results",
                reuse_code=False,
                dry_run=dry_run,
            )

            # Stage 4: VoiceBench scoring on agent ASR transcript (output_asr.jsonl)
            if agent_audio_stage_enabled:
                print("\n--- Running scoring (agent ASR) ---")
                score_command_asr = f"{build_score_command(config, subtest, force=config.get('scoring_force', False))} --input_jsonl output_asr.jsonl --metrics_variant asr"
                run_cmd(
                    ctx=wrap_arguments(""),
                    cluster=config["cluster"],
                    command=score_command_asr,
                    container=scoring_container,
                    partition=config.get("cpu_partition") or config.get("partition"),
                    run_after=[agent_audio_expname],
                    expname=f"{expname}_score_asr",
                    installation_command=config.get("scoring_installation_command"),
                    log_dir=f"{eval_results_path}/summarized-results",
                    reuse_code=False,
                    dry_run=dry_run,
                )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="VoiceBench evaluation with official scoring")
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

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_voicebench_eval(config)


if __name__ == "__main__":
    main()
