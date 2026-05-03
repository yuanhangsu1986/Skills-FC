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
Aggregate BBA per-category metrics into a single bba.aggregate entry.

Reads per-category metrics.json files and computes the mean accuracy.
All 4 BBA categories have 250 samples each, so simple average = correct aggregate.

Usage:
    python run_bba_aggregate.py --output_dir <path> [--categories cat1 cat2 ...] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

ALL_CATEGORIES = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]


def aggregate(output_dir: str, categories: list, force: bool = False, decoding_mode: str = "greedy") -> int:
    output_dir = Path(output_dir)
    agg_metrics_file = output_dir / "eval-results" / "bba_aggregate" / "metrics.json"

    if agg_metrics_file.exists() and not force:
        try:
            if decoding_mode in json.loads(agg_metrics_file.read_text()).get("bba.aggregate", {}):
                print(f"Aggregation already done for {decoding_mode}. Skipping (use --force to re-run).")
                return 0
        except Exception:
            pass

    per_category = {}
    for category in categories:
        metrics_file = output_dir / "eval-results" / category / "metrics.json"
        if not metrics_file.exists():
            print(f"Error: missing metrics for category '{category}': {metrics_file}", file=sys.stderr)
            return 1
        try:
            acc = json.loads(metrics_file.read_text()).get(f"bba.{category}", {}).get(decoding_mode, {}).get("accuracy")
        except Exception as e:
            print(f"Error reading metrics for category '{category}': {e}", file=sys.stderr)
            return 1
        if acc is None:
            print(f"Error: no accuracy value found in metrics for category '{category}'", file=sys.stderr)
            return 1
        per_category[category] = acc

    aggregate_accuracy = round(sum(per_category.values()) / len(per_category), 2)
    existing = {}
    if agg_metrics_file.exists():
        try:
            existing = json.loads(agg_metrics_file.read_text())
        except Exception:
            pass
    existing.setdefault("bba.aggregate", {})[decoding_mode] = {
        "accuracy": aggregate_accuracy,
        "num_categories": len(per_category),
        "per_category": per_category,
    }
    agg_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    agg_metrics_file.write_text(json.dumps(existing, indent=2))

    print("\n" + "=" * 60)
    print("BBA AGGREGATE RESULTS")
    print("=" * 60)
    for cat, acc in per_category.items():
        print(f"  {cat}: {acc}%")
    print(f"  AVERAGE: {aggregate_accuracy}%")
    print("=" * 60)
    print(f"Metrics saved to {agg_metrics_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Aggregate BBA per-category metrics")
    parser.add_argument("--output_dir", required=True, help="BBA output directory (eval-results/ lives here)")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES, help="Categories to aggregate")
    parser.add_argument("--force", action="store_true", help="Re-run even if aggregate already exists")
    parser.add_argument("--decoding_mode", default="greedy", choices=["greedy", "sampling"], help="Key under which metrics are stored in metrics.json")
    args = parser.parse_args()
    sys.exit(aggregate(args.output_dir, args.categories, args.force, args.decoding_mode))


if __name__ == "__main__":
    main()
