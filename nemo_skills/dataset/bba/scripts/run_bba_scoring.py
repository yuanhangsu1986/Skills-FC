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
Score BigBench Audio generation output with exact-match accuracy, with LLM-as-judge fallback.

Reads output.jsonl produced by nemo-skills inference, compares each
`generation` against `expected_answer` using case-insensitive exact match first.
On mismatch, falls back to LLM-as-judge for semantic equivalence.

Usage:
    python run_bba_scoring.py --eval_results_dir <path/to/eval-results/category>
                              --category formal_fallacies
                              [--judge_model azure/openai/gpt-4o-mini]
                              [--api_type nvidia]
                              [--api_key_env_var NVIDIA_API_KEY]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def first_word(text: str) -> str:
    """Extract first word, lowercase."""
    words = text.split()
    return words[0].lower().strip(".!?,;:") if words else ""


def exact_match(generation: str, expected: str) -> bool:
    """Compare first word of generation against expected answer (case-insensitive)."""
    return first_word(generation) == expected.lower().strip()


def llm_judge(generation: str, expected: str, client, model: str) -> bool:
    """Ask an LLM if the generation is semantically equivalent to the expected answer."""
    gen = generation
    exp = expected

    prompt = (
        f"Expected answer: {exp}\n"
        f"Model response: {gen}\n\n"
        "Does the model response convey the same answer as the expected answer? "
        "Consider number words (e.g. 'three' = '3'), capitalization, and minor phrasing variations. "
        "Answer only 'yes' or 'no'."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


def score(
    eval_results_dir: str,
    category: str,
    input_jsonl: str = "output.jsonl",
    force: bool = False,
    judge_model: str = None,
    api_type: str = "nvidia",
    api_key_env_var: str = "NV_INFERENCE_KEY",
) -> int:
    eval_results_dir = Path(eval_results_dir)
    output_jsonl = eval_results_dir / input_jsonl
    metrics_file = eval_results_dir / "metrics.json"
    summarized_dir = eval_results_dir / "summarized-results"
    summarized_dir.mkdir(parents=True, exist_ok=True)

    benchmark_key = f"bba.{category}"

    if metrics_file.exists() and not force:
        try:
            existing = json.loads(metrics_file.read_text())
            if benchmark_key in existing:
                print(f"Scoring already done for {benchmark_key}. Skipping (use --force to re-run).")
                return 0
        except Exception:
            pass

    if not output_jsonl.exists():
        print(f"Error: {output_jsonl} not found.", file=sys.stderr)
        return 1

    # Set up LLM judge client if model specified
    client = None
    if judge_model:
        from openai import OpenAI
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            print(f"Error: {api_key_env_var} not set.", file=sys.stderr)
            return 1
        if api_type == "nvidia":
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)

    total = correct = judge_calls = 0
    with open(output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            expected = entry.get("expected_answer", "")
            generation = entry.get("generation", "")

            if exact_match(generation, expected):
                correct += 1
            elif client and first_word(generation):
                judge_calls += 1
                if llm_judge(generation, expected, client, judge_model):
                    correct += 1
            total += 1

    if total == 0:
        print("Error: no entries found in output.jsonl", file=sys.stderr)
        return 1

    if judge_calls:
        print(f"LLM judge called for {judge_calls}/{total} entries.")

    accuracy = round(100.0 * correct / total, 2)
    metrics = {"accuracy": accuracy, "correct": correct, "total": total}

    existing_metrics = {}
    if metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    existing_metrics[benchmark_key] = {"greedy": metrics}
    metrics_file.write_text(json.dumps(existing_metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"RESULTS for {benchmark_key}")
    print("=" * 60)
    print(f"  accuracy : {accuracy}%")
    print(f"  correct  : {correct} / {total}")
    print("=" * 60)
    print(f"Metrics saved to {metrics_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Score BigBench Audio output with exact-match + LLM-as-judge")
    parser.add_argument("--eval_results_dir", required=True, help="Path to eval-results/{category}/ directory")
    parser.add_argument("--category", required=True, choices=["formal_fallacies", "navigate", "object_counting", "web_of_lies"])
    parser.add_argument("--input_jsonl", default="output.jsonl", help="JSONL file to score (default: output.jsonl)")
    parser.add_argument("--force", action="store_true", help="Re-run scoring even if metrics.json exists")
    parser.add_argument("--judge_model", default=None, help="LLM judge model (e.g. azure/openai/gpt-4o-mini). If not set, exact match only.")
    parser.add_argument("--api_type", default="nvidia", choices=["nvidia", "openai"], help="API type for judge")
    parser.add_argument("--api_key_env_var", default="NV_INFERENCE_KEY", help="Env var holding the API key")
    args = parser.parse_args()

    sys.exit(score(
        eval_results_dir=args.eval_results_dir,
        category=args.category,
        input_jsonl=args.input_jsonl,
        force=args.force,
        judge_model=args.judge_model,
        api_type=args.api_type,
        api_key_env_var=args.api_key_env_var,
    ))


if __name__ == "__main__":
    main()
