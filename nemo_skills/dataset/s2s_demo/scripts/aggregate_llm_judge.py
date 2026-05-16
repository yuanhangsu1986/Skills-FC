#!/usr/bin/env python3
"""
Aggregate LLM judge results and update metrics.json + output_with_eval.jsonl.

Usage:
    python aggregate_llm_judge.py --results_dir /path/to/eval-results/benchmark
    python aggregate_llm_judge.py --results_dir /path/to/eval-results/benchmark --llm_judge_output /path/to/llm_judge/output.jsonl
"""

import argparse
import json
import os
import re


def extract_rating(text: str) -> float | None:
    """Extract rating from LLM judge response."""
    if not text:
        return None
    match = re.search(r"Rating:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return max(0.0, min(5.0, float(match.group(1))))
    return None


def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM judge results")
    parser.add_argument("--results_dir", required=True, help="Directory with metrics.json and output_with_eval.jsonl")
    parser.add_argument(
        "--llm_judge_output", help="Path to LLM judge output.jsonl (default: results_dir/output.jsonl)"
    )
    args = parser.parse_args()

    llm_judge_file = args.llm_judge_output or os.path.join(args.results_dir, "output.jsonl")
    metrics_file = os.path.join(args.results_dir, "metrics.json")
    eval_file = os.path.join(args.results_dir, "output_with_eval.jsonl")

    # Parse LLM judge output into {item_id: rating}
    judge_ratings = {}
    with open(llm_judge_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            item_id = entry.get("item_id", "")
            rating = extract_rating(entry.get("generation", ""))
            if rating is not None:
                judge_ratings[item_id] = rating

    # Aggregate by subset
    ratings_by_subset = {"full": [], "sounded": []}
    for item_id, rating in judge_ratings.items():
        if item_id.endswith("_full"):
            ratings_by_subset["full"].append(rating)
        elif item_id.endswith("_sounded"):
            ratings_by_subset["sounded"].append(rating)

    # Compute metrics
    llm_judge_metrics = {}
    all_ratings = list(judge_ratings.values())
    if all_ratings:
        avg = sum(all_ratings) / len(all_ratings)
        llm_judge_metrics["overall"] = {
            "avg_rating": round(avg, 3),
            "judge_score": round(avg * 20, 2),
            "count": len(all_ratings),
        }
    for subset, ratings in ratings_by_subset.items():
        if ratings:
            avg = sum(ratings) / len(ratings)
            llm_judge_metrics[subset] = {
                "avg_rating": round(avg, 3),
                "judge_score": round(avg * 20, 2),
                "count": len(ratings),
            }

    # Update metrics.json
    metrics = json.load(open(metrics_file)) if os.path.exists(metrics_file) else {}
    if "dataset_metrics" in metrics:
        metrics["dataset_metrics"]["llm_judge"] = llm_judge_metrics
    else:
        metrics["llm_judge"] = llm_judge_metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {metrics_file}")

    # Update output_with_eval.jsonl with per-sample scores
    if os.path.exists(eval_file):
        updated_entries = []
        with open(eval_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                # Get item_id from original_entry
                original = entry.get("original_entry", entry)
                audio_path = original.get("audio_path", "") or original.get("audio", {}).get("path", "")
                base_id = os.path.basename(audio_path).rsplit(".", 1)[0] if audio_path else ""

                # Add judge scores
                full_rating = judge_ratings.get(f"{base_id}_full")
                sounded_rating = judge_ratings.get(f"{base_id}_sounded")
                entry["llm_judge_scores"] = {
                    "full": {"rating": full_rating, "score": round(full_rating * 20, 2) if full_rating else None},
                    "sounded": {
                        "rating": sounded_rating,
                        "score": round(sounded_rating * 20, 2) if sounded_rating else None,
                    },
                }
                updated_entries.append(entry)

        with open(eval_file, "w") as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Updated {eval_file} with {len(updated_entries)} entries")

    # Print summary
    print("\nLLM Judge Results:")
    for subset, m in llm_judge_metrics.items():
        print(f"  {subset}: {m['avg_rating']:.2f}/5 (score: {m['judge_score']:.1f}/100, n={m['count']})")


if __name__ == "__main__":
    main()
