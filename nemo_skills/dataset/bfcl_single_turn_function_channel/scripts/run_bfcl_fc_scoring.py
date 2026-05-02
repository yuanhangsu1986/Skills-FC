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
BFCL function-channel scoring script.

Reads output.jsonl produced by run_bfcl_fc_inference.py, parses <TOOLCALL>
blocks from each 'generation' field, and scores against the expected_call
and required_fields stored in the same file.

Writes metrics.json under the same directory:
  {
    "bfcl_fc.<category>": {
      "<decoding_mode>": {
        "accuracy": 72.5,
        "num_samples": 400,
        "num_correct": 290
      }
    }
  }

Usage:
    python run_bfcl_fc_scoring.py \
        --output_jsonl /results/simple_python/output.jsonl \
        --category simple_python \
        [--decoding_mode greedy] \
        [--force]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _parse_toolcall(generation: str) -> list:
    """Parse <TOOLCALL>...</TOOLCALL> → [{func_name: args_dict}, ...]."""
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", generation, re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    try:
        calls = json.loads(raw)
        return [
            {tc["name"]: tc.get("arguments", {})}
            for tc in calls
            if isinstance(tc, dict) and "name" in tc
        ]
    except Exception:
        return []


def score(
    output_jsonl: str,
    category: str,
    decoding_mode: str = "greedy",
    force: bool = False,
) -> int:
    output_path = Path(output_jsonl)
    metrics_file = output_path.parent / "metrics.json"
    summarized_dir = output_path.parent / "summarized-results"
    summarized_dir.mkdir(parents=True, exist_ok=True)

    benchmark_key = f"bfcl_fc.{category}"

    if metrics_file.exists() and not force:
        try:
            existing = json.loads(metrics_file.read_text())
            if decoding_mode in existing.get(benchmark_key, {}):
                print(f"Scoring already done for {benchmark_key} ({decoding_mode}). Skipping (use --force to re-run).")
                return 0
        except Exception:
            pass

    if not output_path.exists():
        print(f"Error: {output_path} not found.", file=sys.stderr)
        return 1

    with open(output_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("Error: no entries found in output file.", file=sys.stderr)
        return 1

    print(f"[scoring] Scoring {len(entries)} entries from {output_path}")

    candidates = []
    references = []
    for entry in entries:
        tool_calls = _parse_toolcall(entry.get("generation", ""))
        candidates.append({"tool_response": tool_calls})
        expected_call = entry.get("expected_call", [])
        required_fields = entry.get("required_fields", {})
        # required_fields values are stored as [[param, type], ...] — compatible with List[Tuple]
        references.append((expected_call, required_fields))

    from nemo_skills.dataset.bfcl_single_turn_function_channel.score import _score_one
    num_correct = sum(
        _score_one(c, ref_call, req_fields)
        for c, (ref_call, req_fields) in zip(candidates, references)
    )
    accuracy = round(num_correct * 100.0 / len(entries), 2)
    metrics = {
        "accuracy": accuracy,
        "num_samples": len(entries),
        "num_correct": num_correct,
    }

    existing_metrics = {}
    if metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    existing_metrics.setdefault(benchmark_key, {})[decoding_mode] = metrics
    metrics_file.write_text(json.dumps(existing_metrics, indent=2))

    print("\n" + "=" * 60)
    print(f"RESULTS for {benchmark_key}")
    print("=" * 60)
    print(f"  accuracy    : {accuracy}%")
    print(f"  num_correct : {num_correct} / {len(entries)}")
    print("=" * 60)
    print(f"Metrics saved to {metrics_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Score BFCL function-channel output")
    parser.add_argument("--output_jsonl", required=True, help="Path to output.jsonl from inference")
    parser.add_argument("--category", required=True, help="BFCL category name (e.g. simple_python)")
    parser.add_argument("--decoding_mode", default="greedy", choices=["greedy", "sampling"], help="Key under which metrics are stored in metrics.json")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json exists")
    args = parser.parse_args()

    sys.exit(score(
        output_jsonl=args.output_jsonl,
        category=args.category,
        decoding_mode=args.decoding_mode,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
