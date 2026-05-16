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
Run VoiceBench scoring with nemo-skills compatible output structure.

Creates:
- summarized-results/ directory
- metrics.json with evaluation results
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_scoring(
    eval_results_dir: str,
    voicebench_repo: str,
    subtest: str,
    evaluator: str,
    needs_judge: bool,
    api_type: str = "openai",
    nvidia_model: str = "meta/llama-3.1-70b-instruct",
    force: bool = False,
):
    """Run VoiceBench scoring and save results in nemo-skills format."""
    eval_results_dir = Path(eval_results_dir)
    output_jsonl = eval_results_dir / "output.jsonl"
    converted_jsonl = eval_results_dir / "voicebench_format.jsonl"
    summarized_dir = eval_results_dir / "summarized-results"
    metrics_file = eval_results_dir / "metrics.json"

    # Skip if already scored (unless force is set)
    if metrics_file.exists() and not force:
        print(f"Scoring already done for voicebench.{subtest} (metrics.json exists). Skipping.")
        print("Use --force to re-run scoring.")
        with open(metrics_file) as f:
            existing_metrics = json.load(f)
        print(f"Existing metrics: {json.dumps(existing_metrics, indent=2)}")
        return 0

    # Create summarized-results directory
    summarized_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert format
    print(f"Converting {output_jsonl} to VoiceBench format...")
    convert_script = Path(__file__).parent / "convert_to_voicebench_format.py"
    cmd = f"python {convert_script} --input {output_jsonl} --output {converted_jsonl} --subtest {subtest}"
    subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run GPT judge if needed
    if needs_judge:
        print("Running GPT judge...")
        api_judge_args = f"--src_file {converted_jsonl}"
        if api_type:
            api_judge_args += f" --api_type {api_type}"
        if nvidia_model:
            api_judge_args += f" --nvidia_model {nvidia_model}"
        cmd = f"cd {voicebench_repo} && python api_judge.py {api_judge_args}"
        subprocess.run(cmd, shell=True, check=True)
        result_file = eval_results_dir / "result-voicebench_format.jsonl"
    else:
        result_file = converted_jsonl

    # Step 3: Run evaluate.py and capture metrics
    print(f"Running evaluation with {evaluator} evaluator...")
    cmd = f"cd {voicebench_repo} && python evaluate.py --src_file {result_file} --evaluator {evaluator}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    print(result.stderr, file=sys.stderr)

    # Parse metrics from loguru output
    # Format: 2025-12-07 09:11:15.950 | INFO | __main__:main:18 - {'panda': 30.0, 'gpt': 50.0}
    metrics = {}
    for line in result.stderr.split("\n"):
        if "INFO" in line and "{" in line and "}" in line:
            try:
                match = re.search(r"\{[^}]+\}", line)
                if match:
                    # Use ast.literal_eval for safety
                    import ast

                    metrics = ast.literal_eval(match.group())
            except Exception:
                print(f"Warning: Could not parse metrics from line: {line}", file=sys.stderr)

    # Save metrics.json in nemo-skills format
    nemo_metrics = {f"voicebench.{subtest}": {"greedy": metrics}}
    with open(metrics_file, "w") as f:
        json.dump(nemo_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Also print metrics summary
    print("\n" + "=" * 60)
    print(f"RESULTS for voicebench.{subtest}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run VoiceBench scoring with nemo-skills output format")
    parser.add_argument(
        "--eval_results_dir", required=True, help="Path to eval-results/voicebench.{subtest}/ directory"
    )
    parser.add_argument("--voicebench_repo", required=True, help="Path to VoiceBench repository")
    parser.add_argument("--subtest", required=True, help="Subtest name")
    parser.add_argument("--evaluator", required=True, help="Evaluator type (qa, open, harm, ifeval, mcq, bbh)")
    parser.add_argument("--needs_judge", action="store_true", help="Whether to run GPT judge first")
    parser.add_argument("--api_type", default="openai", choices=["openai", "nvidia"], help="API type for judge")
    parser.add_argument("--nvidia_model", default="meta/llama-3.1-70b-instruct", help="Model for NVIDIA API")
    parser.add_argument("--force", action="store_true", help="Force re-run scoring even if metrics.json exists")

    args = parser.parse_args()

    rc = run_scoring(
        eval_results_dir=args.eval_results_dir,
        voicebench_repo=args.voicebench_repo,
        subtest=args.subtest,
        evaluator=args.evaluator,
        needs_judge=args.needs_judge,
        api_type=args.api_type,
        nvidia_model=args.nvidia_model,
        force=args.force,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
