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
Score BigBench Audio generation output with exact-match accuracy.

Reads output.jsonl produced by nemo-skills inference, compares each
`generation` against `expected_answer` using case-insensitive exact match,
and writes metrics.json.

Usage:
    python run_bba_scoring.py --eval_results_dir <path/to/eval-results/bba.formal_fallacies>
                              --category formal_fallacies
"""

import argparse
import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    """Lowercase, strip whitespace, remove trailing punctuation."""
    return text.lower().strip().rstrip(".!?,;:")


def extract_answer(generation: str, expected: str) -> bool:
    """
    Check if the generation contains the expected answer.

    Tries exact match first, then checks if expected appears as a whole word
    at the start of the response (model may add explanation after the answer).
    """
    gen = normalize(generation)
    exp = normalize(expected)

    if gen == exp:
        return True

    # Accept if generation starts with the expected answer followed by space or punctuation
    pattern = r'^\s*' + re.escape(exp) + r'(\s|[.,;:]|$)'
    if re.match(pattern, gen):
        return True

    return False


def score(eval_results_dir: str, category: str, input_jsonl: str = "output.jsonl", force: bool = False) -> int:
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

    total = correct = 0
    with open(output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            expected = entry.get("expected_answer", "")
            generation = entry.get("generation", "")
            if extract_answer(generation, expected):
                correct += 1
            total += 1

    if total == 0:
        print("Error: no entries found in output.jsonl", file=sys.stderr)
        return 1

    accuracy = round(100.0 * correct / total, 2)
    metrics = {"accuracy": accuracy, "correct": correct, "total": total}

    # Merge with existing metrics.json
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
    parser = argparse.ArgumentParser(description="Score BigBench Audio output with exact-match accuracy")
    parser.add_argument("--eval_results_dir", required=True, help="Path to eval-results/bba.<category>/ directory")
    parser.add_argument("--category", required=True, choices=["formal_fallacies", "navigate", "object_counting", "web_of_lies"])
    parser.add_argument("--input_jsonl", default="output.jsonl", help="JSONL file to score (default: output.jsonl)")
    parser.add_argument("--force", action="store_true", help="Re-run scoring even if metrics.json exists")
    args = parser.parse_args()

    sys.exit(score(
        eval_results_dir=args.eval_results_dir,
        category=args.category,
        input_jsonl=args.input_jsonl,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
