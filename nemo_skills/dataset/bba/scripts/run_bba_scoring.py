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
Score BigBench Audio generation output following the official BBA evaluation methodology.

Reads output_asr.jsonl (produced by the ASR stage), passes each entry to an LLM judge
using the exact official prompt template. Runs the full evaluation 3 independent times
and reports the average accuracy, matching the official "averages across three independent
evaluation runs" methodology.

The question context comes from the question_asr field (Whisper transcript of the input
question audio), which is written by nemo_skills/inference/transcribe_audio.py when
--data_dir is provided.

Usage:
    python run_bba_scoring.py --eval_results_dir <path/to/eval-results/category>
                              --category formal_fallacies
                              --judge_model aws/anthropic/bedrock-claude-sonnet-4-6
                              [--api_type nvidia]
                              [--api_key_env_var NV_INFERENCE_KEY]
"""

import argparse
import json
import sys
from pathlib import Path

# Official BBA judge prompt template (verbatim from the BBA evaluation methodology)
JUDGE_PROMPT = """\
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.
If the CANDIDATE ANSWER contradicts itself, assess the first proposed answer.
If the CANDIDATE ANSWER provides a final answer and working, assess the final answer only.
If the CANDIDATE ANSWER includes irrelevant information, assess only the relevant information.
If the CANDIDATE ANSWER includes a numeric value it is ok if it is spelled e.g. 7 or seven
It is ok if the CANDIDATE ANSWER involves a misspelling of a person's name e.g. Leda or Lida, Autry or Audrie.

The question, for reference only: START QUESTION {question} \n\nEND QUESTION

The OFFICIAL ANSWER:{official_answer}

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply only with CORRECT or INCORRECT."""

NUM_EVAL_RUNS = 3  # Official methodology: average accuracy across 3 independent evaluation runs


def llm_judge(generation: str, expected: str, question: str, client, model: str) -> bool:
    """Call the LLM judge once and return True if CORRECT."""
    prompt = JUDGE_PROMPT.format(
        question=question or "",
        official_answer=expected,
        candidate_answer=generation,
    )
    text = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=1.0,
    ).choices[0].message.content.strip().upper()
    return text == "CORRECT"


def _score_one_run(entries: list, client, judge_model: str, run_idx: int) -> tuple[int, int]:
    """Run one complete evaluation pass. Returns (correct, total)."""
    correct = total = 0
    for entry in entries:
        expected = entry.get("expected_answer", "")
        generation = entry.get("generation", "")
        question = entry.get("question_asr", "")

        is_correct = llm_judge(generation, expected, question, client, judge_model)
        if is_correct:
            correct += 1
        print(f"  [run {run_idx + 1}, entry {total + 1}] judge: {'CORRECT' if is_correct else 'INCORRECT'}", flush=True)
        total += 1

    return correct, total


def score(
    eval_results_dir: str,
    category: str,
    input_jsonl: str = "output_asr.jsonl",
    force: bool = False,
    judge_model: str = None,
    api_type: str = "nvidia",
    judge_base_url: str = None,
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
            if existing.get(benchmark_key):
                print(f"Scoring already done for {benchmark_key}. Skipping (use --force to re-run).")
                return 0
        except Exception:
            pass

    if not output_jsonl.exists():
        print(f"Error: {output_jsonl} not found.", file=sys.stderr)
        return 1

    if not judge_model:
        print("Error: --judge_model is required (official BBA methodology uses LLM judge for all entries).", file=sys.stderr)
        return 1

    if api_type == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
    else:
        import os
        from openai import OpenAI
        api_key = os.environ.get("NV_INFERENCE_KEY") if api_type == "nvidia" else os.environ.get("OPENAI_API_KEY")
        client = OpenAI(base_url=judge_base_url, api_key=api_key)

    with open(output_jsonl) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        print("Error: no entries found in output file.", file=sys.stderr)
        return 1

    print(
        f"Scoring {len(entries)} entries in {output_jsonl} "
        f"with judge {judge_model} ({NUM_EVAL_RUNS} independent runs, averaged)",
        flush=True,
    )

    # Official methodology: run full evaluation NUM_EVAL_RUNS times, average the accuracy scores
    run_accuracies = []
    for run_idx in range(NUM_EVAL_RUNS):
        print(f"\n--- Evaluation run {run_idx + 1}/{NUM_EVAL_RUNS} ---", flush=True)
        correct, total = _score_one_run(entries, client, judge_model, run_idx)
        run_acc = correct / total
        run_accuracies.append(run_acc)
        print(f"  Run {run_idx + 1} accuracy: {round(100.0 * run_acc, 2)}% ({correct}/{total})", flush=True)

    accuracy = round(100.0 * sum(run_accuracies) / NUM_EVAL_RUNS, 2)
    metrics = {
        "accuracy": accuracy,
        "run_accuracies": [round(100.0 * a, 2) for a in run_accuracies],
        "total": len(entries),
    }

    existing_metrics = {}
    if metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    existing_metrics[benchmark_key] = metrics
    metrics_file.write_text(json.dumps(existing_metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"RESULTS for {benchmark_key}")
    print("=" * 60)
    print(f"  accuracy (avg of {NUM_EVAL_RUNS} runs) : {accuracy}%")
    print(f"  per-run accuracies : {metrics['run_accuracies']}")
    print(f"  total entries      : {len(entries)}")
    print("=" * 60)
    print(f"Metrics saved to {metrics_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Score BigBench Audio output using official BBA evaluation methodology")
    parser.add_argument("--eval_results_dir", required=True, help="Path to eval-results/{category}/ directory")
    parser.add_argument("--category", required=True, choices=["formal_fallacies", "navigate", "object_counting", "web_of_lies"])
    parser.add_argument("--input_jsonl", default="output_asr.jsonl", help="JSONL file to score (default: output_asr.jsonl)")
    parser.add_argument("--force", action="store_true", help="Re-run scoring even if metrics.json exists")
    parser.add_argument("--judge_model", required=True, help="LLM judge model (e.g. aws/anthropic/bedrock-claude-sonnet-4-6)")
    parser.add_argument("--api_type", default="nvidia", help="API type for judge")
    parser.add_argument("--judge_base_url", default=None, help="Base URL for the judge API (nvidia/openai only)")
    args = parser.parse_args()

    sys.exit(score(
        eval_results_dir=args.eval_results_dir,
        category=args.category,
        input_jsonl=args.input_jsonl,
        force=args.force,
        judge_model=args.judge_model,
        api_type=args.api_type,
        judge_base_url=args.judge_base_url,
    ))


if __name__ == "__main__":
    main()
