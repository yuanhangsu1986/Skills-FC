"""
Run S2S demo evaluation: generation + conversation behavior scoring + LLM judge.

Usage:
    python run_s2s_demo_eval.py --config s2s_demo_eval_config.yaml
    python run_s2s_demo_eval.py --config s2s_demo_eval_config.yaml --dry_run
    python run_s2s_demo_eval.py --config s2s_demo_eval_config.yaml --generation_only
    python run_s2s_demo_eval.py --config s2s_demo_eval_config.yaml --scoring_only
    python run_s2s_demo_eval.py --config s2s_demo_eval_config.yaml --llm_judge_only

To enable LLM judge, add to config:
    llm_judge:
      enabled: true
      model: meta/llama-3.1-8b-instruct  # NVIDIA API model name
      server_type: openai
      base_url: https://inference-api.nvidia.com/v1  # default
"""

import argparse

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict, benchmark: str, generate_llm_judge_input: bool = False) -> str:
    """Build the scoring command for eval_conversation_behavior_v2.py."""
    eval_results_dir = f"{config['output_dir']}/eval-results/{benchmark}"
    script_path = "nemo_skills/dataset/s2s_demo/scripts/eval_conversation_behavior_v2.py"

    # Scoring parameters with defaults
    scoring = config.get("scoring", {})

    cmd_args = [
        f"python {script_path}",
        f"--results_dir {eval_results_dir}",
        f"--barge_in_threshold_sec {scoring.get('barge_in_threshold_sec', 1.5)}",
        f"--tt_latency_threshold_sec {scoring.get('tt_latency_threshold_sec', 1.5)}",
        f"--tt_precision_buffer_sec {scoring.get('tt_precision_buffer_sec', 0.5)}",
        f"--tt_recall_buffer_sec {scoring.get('tt_recall_buffer_sec', 0.5)}",
        f"--vad_min_silence_duration_ms {scoring.get('vad_min_silence_duration_ms', 1500)}",
        f"--segment_buffer_sec {scoring.get('segment_buffer_sec', 0.5)}",
        "--output_file metrics.json",
    ]

    if scoring.get("verbose", True):
        cmd_args.append("--verbose")
    if scoring.get("disable_transcription", False):
        cmd_args.append("--disable_transcription")
    if scoring.get("save_per_sample_results", False):
        cmd_args.append("--save_per_sample_results")
    if scoring.get("force_recompute", False):
        cmd_args.append("--force_recompute")
    if generate_llm_judge_input:
        cmd_args.append("--generate_llm_judge_input")

    return " ".join(cmd_args)


def build_aggregate_command(config: dict, benchmark: str, llm_judge_output_dir: str) -> str:
    """Build the aggregation command for aggregate_llm_judge.py."""
    eval_results_dir = f"{config['output_dir']}/eval-results/{benchmark}"
    script_path = "nemo_skills/dataset/s2s_demo/scripts/aggregate_llm_judge.py"

    return (
        f"python {script_path} --results_dir {eval_results_dir} --llm_judge_output {llm_judge_output_dir}/output.jsonl"
    )


def run_s2s_demo_eval(config: dict):
    """Run S2S demo evaluation pipeline."""
    benchmark = config.get("benchmark", "s2s_demo.demo_20251124")
    expname = config.get("expname", "s2s_demo_eval")
    dry_run = config.get("dry_run", False)
    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    llm_judge_only = config.get("llm_judge_only", False)

    eval_results_path = f"{config['output_dir']}/eval-results/{benchmark}"
    llm_judge_output_dir = f"{eval_results_path}/llm_judge"

    print(f"{'=' * 60}")
    print("S2S Demo Evaluation")
    print(f"{'=' * 60}")
    print(f"Benchmark: {benchmark}")
    print(f"Output: {config['output_dir']}")

    # Build extra args for hydra overrides
    extra_args = []
    if config.get("max_samples"):
        extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        extra_args.append(f"++server.server_type={config['server_server_type']}")
    extra_args_str = " ".join(extra_args)

    # Skip to LLM judge if llm_judge_only
    if llm_judge_only:
        scoring_only = True  # Skip generation

    # Generation phase
    if not scoring_only:
        print("\n--- Running generation ---")
        nemo_eval(
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
            expname=expname,
            auto_summarize_results=False,
            dry_run=dry_run,
        )

    # Scoring phase (with LLM judge input generation)
    if not generation_only and not llm_judge_only:
        print("\n--- Running scoring ---")
        run_llm_judge = config.get("llm_judge", {}).get("enabled", False)
        score_command = build_score_command(config, benchmark, generate_llm_judge_input=run_llm_judge)

        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=score_command,
            container=config.get("scoring_container"),
            partition=config.get("scoring_partition") or config.get("partition"),
            num_gpus=config.get("scoring_gpus", 1),  # VAD/ASR needs GPU
            run_after=[expname] if not scoring_only else None,
            expname=f"{expname}_score",
            log_dir=f"{eval_results_path}/scoring-logs",
            dry_run=dry_run,
        )

    # LLM Judge phase
    llm_judge_config = config.get("llm_judge", {})
    if llm_judge_config.get("enabled", False) and not generation_only:
        print("\n--- Running LLM judge ---")
        llm_judge_input = f"{eval_results_path}/llm_judge_input.jsonl"

        # API base URL for LLM judge
        base_url = llm_judge_config.get("base_url", "https://inference-api.nvidia.com/v1")

        from nemo_skills.pipeline.cli import generate

        generate(
            ctx=wrap_arguments("++prompt_format=openai"),
            cluster=config["cluster"],
            output_dir=llm_judge_output_dir,
            input_file=llm_judge_input,
            model=llm_judge_config.get("model", "meta/llama-3.1-8b-instruct"),
            server_address=base_url,  # Pass API URL directly
            server_type=llm_judge_config.get("server_type", "openai"),
            server_gpus=0,  # No GPU needed for API calls
            partition=llm_judge_config.get("partition", "cpu"),  # Use CPU partition for API calls
            run_after=[f"{expname}_score"] if not llm_judge_only else None,
            expname=f"{expname}_llm_judge",
            log_dir=f"{eval_results_path}/llm-judge-logs",
            dry_run=dry_run,
        )

        # Aggregation phase
        print("\n--- Running LLM judge aggregation ---")
        aggregate_command = build_aggregate_command(config, benchmark, llm_judge_output_dir)

        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=aggregate_command,
            container=config.get("scoring_container"),
            partition="cpu",  # Use CPU partition for aggregation
            num_gpus=0,
            run_after=[f"{expname}_llm_judge"],
            expname=f"{expname}_aggregate",
            log_dir=f"{eval_results_path}/aggregate-logs",
            dry_run=dry_run,
        )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="S2S demo evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    # CLI overrides
    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--benchmark", help="Override benchmark")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--num_chunks", type=int, help="Override num_chunks")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--generation_only", action="store_true", help="Only run generation")
    parser.add_argument("--scoring_only", action="store_true", help="Only run scoring")
    parser.add_argument(
        "--llm_judge_only", action="store_true", help="Only run LLM judge (skip generation and scoring)"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    override_keys = ["cluster", "partition", "model", "output_dir", "benchmark", "max_samples", "num_chunks"]
    for key in override_keys:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)

    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if args.llm_judge_only:
        config["llm_judge_only"] = True

    run_s2s_demo_eval(config)


if __name__ == "__main__":
    main()
