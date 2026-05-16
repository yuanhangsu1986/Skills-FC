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

"""FDB v3 ChenChen scoring: ingest upstream FD3 reports into metrics.json.

The upstream orchestrator (run_fd3_audio_eval_job.sh) already runs ASR + the
three FD3 evaluators and writes a consolidated model_report.json. We just
reshape that into the same metrics.json shape fdb_v3 produces, under the key
fdb_v3_chen_chen.tool_call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BENCHMARK_KEY = "fdb_v3_chen_chen.tool_call"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  warning: could not parse {path}: {e}")
        return None


def _build_headline(eval_report: dict | None) -> dict | None:
    if not eval_report:
        return None
    bm = eval_report.get("by_metric") or {}
    return {
        "tool_selection_acc": bm.get("tool_selection_acc"),
        "argument_acc": bm.get("argument_acc"),
        "response_qual": bm.get("response_qual"),
        "turn_take_rate": (eval_report.get("turn_taking") or {}).get("turn_take_rate"),
        "total_scenarios": eval_report.get("total_scenarios"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest upstream FD3 reports into metrics.json")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument(
        "--fd3_run_root",
        type=Path,
        required=True,
        help="The FD3_RUN_ROOT used by run_fd3_audio_eval_job.sh (contains reports/ and model_report.json)",
    )
    parser.add_argument("--provider", type=str, default="fdb_v3_chen_chen")
    parser.add_argument("--force", action="store_true", help="Overwrite metrics.json even if already populated")
    args = parser.parse_args()

    eval_results_dir = args.eval_results_dir.resolve()
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = eval_results_dir / "metrics.json"

    if metrics_file.exists() and not args.force:
        try:
            existing = json.loads(metrics_file.read_text(encoding="utf-8"))
            if existing.get(BENCHMARK_KEY):
                print(f"Scoring already done for {BENCHMARK_KEY}. Skipping (use --force to re-run).")
                return
        except Exception:
            pass

    fd3_run_root = args.fd3_run_root.resolve()
    if not fd3_run_root.exists():
        sys.exit(f"FD3 run root not found: {fd3_run_root}")

    model_report_path = fd3_run_root / "model_report.json"
    reports_dir = fd3_run_root / "reports"

    eval_report = None
    pass_report = None
    latency_report = None
    consolidated = _load_json(model_report_path)

    if consolidated:
        eval_report = consolidated.get("eval")
        pass_report = consolidated.get("pass_rate")
        latency_report = consolidated.get("latency")
    # Fallback to per-file reports if the consolidated summary is missing/partial.
    if eval_report is None:
        eval_report = _load_json(reports_dir / f"{args.provider}_eval.json")
    if pass_report is None:
        pass_report = _load_json(reports_dir / f"{args.provider}_pass_rate.json")
    if latency_report is None:
        latency_report = _load_json(reports_dir / f"{args.provider}_latency.json")

    if eval_report is None and pass_report is None and latency_report is None:
        sys.exit(
            f"FDB v3 ChenChen scoring found no upstream reports under {fd3_run_root}. "
            f"Expected {model_report_path} or {reports_dir}/{args.provider}_*.json"
        )

    metrics: dict[str, Any] = {BENCHMARK_KEY: {}}
    if eval_report is not None:
        metrics[BENCHMARK_KEY]["eval"] = eval_report
    if pass_report is not None:
        metrics[BENCHMARK_KEY]["pass_rate"] = pass_report
    if latency_report is not None:
        metrics[BENCHMARK_KEY]["latency"] = latency_report

    headline = _build_headline(eval_report)
    if headline:
        metrics[BENCHMARK_KEY]["headline"] = headline

    metrics[BENCHMARK_KEY]["provider"] = args.provider
    metrics[BENCHMARK_KEY]["fd3_run_root"] = str(fd3_run_root)
    if consolidated:
        if "checkpoint_dir" in consolidated:
            metrics[BENCHMARK_KEY]["checkpoint_dir"] = consolidated["checkpoint_dir"]
        if "num_result_files" in consolidated:
            metrics[BENCHMARK_KEY]["num_result_files"] = consolidated["num_result_files"]
        if "model_name" in consolidated:
            metrics[BENCHMARK_KEY]["model_name"] = consolidated["model_name"]

    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_file}")
    if headline:
        print(
            "Headline: "
            f"tool_selection_acc={headline.get('tool_selection_acc')} "
            f"argument_acc={headline.get('argument_acc')} "
            f"response_qual={headline.get('response_qual')} "
            f"turn_take_rate={headline.get('turn_take_rate')} "
            f"total_scenarios={headline.get('total_scenarios')}"
        )


if __name__ == "__main__":
    main()
