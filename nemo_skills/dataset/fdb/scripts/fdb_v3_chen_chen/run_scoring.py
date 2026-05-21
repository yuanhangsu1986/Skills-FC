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

"""FDB v3 ChenChen scoring: merge per-shard inference outputs, run cchen1's
evaluators once over the union, and reshape into metrics.json.

Each inference shard writes its own subset of per-sample result_<provider>.json
files under {shard_run_root}/per_sample_data/. This scorer:
  1. Symlink-merges all shards' per_sample_data dirs into a single
     {merged_run_root}/per_sample_data/ — one symlink per sample folder pointing
     at the shard that processed it.
  2. Invokes cchen1's three evaluators (evaluate_tool_calls.py,
     evaluate_pass_rate.py, analyze_tool_latency.py) once over the merged tree.
     This is what makes the headline numbers byte-identical to cchen1's
     single-machine output.
  3. Writes a consolidated model_report.json at {merged_run_root}/, then
     reshapes it into metrics.json under key fdb_v3_chen_chen.tool_call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _merge_shards(shard_run_roots: list[Path], merged_root: Path, provider: str) -> Path:
    """For each per-sample subdir across all shards, create a REAL directory
    at merged/per_sample_data/<sample>/ and symlink result_<provider>.json
    inside.

    Why not symlink at the directory level? pathlib.Path.rglob does NOT follow
    directory symlinks on Python < 3.13 (and cchen1's container ships Python
    3.12). The evaluators walk the merged tree via rglob, so any symlinked
    sample dir is silently skipped. Real dirs + file-level symlinks side-step
    that limitation cleanly — and the evaluators only need the result JSON;
    audio fields (`audio_agent_speech_start`, `transcript`, `actual_tool_calls`)
    are already serialized inside each per-sample JSON.

    Each sample folder belongs to exactly one shard (round-robin assignment),
    so there are no cross-shard collisions to resolve.
    """
    merged_data = merged_root / "per_sample_data"
    # Wipe any prior merged tree so stale entries from an earlier (possibly
    # buggy) merge can't poison the rebuild. merged_data is purely a derived
    # cache — sources of truth are the per-shard dirs.
    if merged_data.exists() or merged_data.is_symlink():
        import shutil
        try:
            if merged_data.is_symlink():
                merged_data.unlink()
            else:
                shutil.rmtree(merged_data)
        except OSError as e:
            print(f"  warning: could not clear stale merged dir {merged_data}: {e}")
    merged_data.mkdir(parents=True, exist_ok=True)
    total = 0
    per_shard = []
    for shard_root in shard_run_roots:
        shard_data = shard_root / "per_sample_data"
        if not shard_data.exists():
            print(f"  warning: shard data dir missing: {shard_data}")
            per_shard.append(0)
            continue
        n = 0
        for sample_dir in sorted(shard_data.iterdir()):
            if not sample_dir.is_dir():
                continue
            src_result = sample_dir / f"result_{provider}.json"
            if not src_result.exists():
                # Inference produced no result file for this sample (e.g.
                # crashed mid-run). Skip; evaluators will treat the sample
                # as missing.
                continue
            dst_sample = merged_data / sample_dir.name
            dst_sample.mkdir(exist_ok=True)
            dst_result = dst_sample / f"result_{provider}.json"
            target = src_result.resolve()
            if dst_result.is_symlink() or dst_result.exists():
                try:
                    if dst_result.resolve() == target:
                        n += 1
                        continue
                except OSError:
                    pass
                try:
                    dst_result.unlink()
                except OSError as e:
                    print(f"  warning: could not replace {dst_result}: {e}")
                    continue
            try:
                dst_result.symlink_to(target)
                n += 1
            except OSError as e:
                print(f"  warning: symlink {dst_result} -> {target} failed: {e}")
        per_shard.append(n)
        total += n
    print(f"  merged {total} sample(s) from {len(shard_run_roots)} shard(s): {per_shard}")
    return merged_data


def _run_evaluator(
    cmd: list[str],
    label: str,
    cwd: Path,
) -> bool:
    print(f"  running {label}: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    except FileNotFoundError as e:
        print(f"  ERROR: {label} executable not found: {e}")
        return False
    if proc.returncode != 0:
        print(f"  ERROR: {label} exited with code {proc.returncode}")
        return False
    return True


def _run_evaluators(
    release_code: Path,
    benchmark_json: Path,
    results_dir: Path,
    reports_dir: Path,
    provider: str,
    use_llm: bool,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable or "python3"
    llm_flag = ["--use-llm"] if use_llm else []

    _run_evaluator(
        [
            py,
            str(release_code / "evaluate_tool_calls.py"),
            "--benchmark", str(benchmark_json),
            "--results-dir", str(results_dir),
            "--provider", provider,
            "--output", str(reports_dir / f"{provider}_eval.json"),
            *llm_flag,
        ],
        "evaluate_tool_calls.py",
        cwd=release_code,
    )
    _run_evaluator(
        [
            py,
            str(release_code / "evaluate_pass_rate.py"),
            "--benchmark", str(benchmark_json),
            "--results-dir", str(results_dir),
            "--provider", provider,
            "--output", str(reports_dir / f"{provider}_pass_rate.json"),
            *llm_flag,
        ],
        "evaluate_pass_rate.py",
        cwd=release_code,
    )
    _run_evaluator(
        [
            py,
            str(release_code / "analyze_tool_latency.py"),
            "--results-dir", str(results_dir),
            "--provider", provider,
            "--output", str(reports_dir / f"{provider}_latency.json"),
        ],
        "analyze_tool_latency.py",
        cwd=release_code,
    )


def _build_consolidated_report(
    merged_root: Path,
    reports_dir: Path,
    results_dir: Path,
    provider: str,
    run_label: str,
    checkpoint_dir: str,
) -> dict:
    summary: dict[str, Any] = {
        "provider": provider,
        "model_name": run_label,
        "checkpoint_dir": checkpoint_dir,
        "run_root": str(merged_root),
    }
    for suffix, key in [("eval", "eval"), ("pass_rate", "pass_rate"), ("latency", "latency")]:
        path = reports_dir / f"{provider}_{suffix}.json"
        if path.exists():
            try:
                summary[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  warning: could not parse {path}: {e}")
    summary["num_result_files"] = len(list(results_dir.rglob(f"result_{provider}.json")))
    return summary


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
    parser = argparse.ArgumentParser(description="Merge shards + run FD3 evaluators + ingest into metrics.json")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument(
        "--fd3_run_root",
        type=Path,
        required=True,
        help="Parent root holding per-shard subtrees (shard_0/, shard_1/, ...) and merged/.",
    )
    parser.add_argument(
        "--merged_run_root",
        type=Path,
        required=True,
        help="Where the unified per_sample_data/ + reports/ go. Evaluators run against this tree.",
    )
    parser.add_argument(
        "--shard_run_roots",
        type=Path,
        nargs="+",
        required=True,
        help="One path per shard. Each must contain per_sample_data/<sample>/result_<provider>.json.",
    )
    parser.add_argument(
        "--fdb_repo_path",
        type=Path,
        required=True,
        help="FDBV3_CHENCHEN root. Used to locate FD3/release_code/{evaluators} and benchmark_data_v2.json.",
    )
    parser.add_argument("--provider", type=str, default="fdb_v3_chen_chen")
    parser.add_argument("--run_label", type=str, default="")
    parser.add_argument("--use-llm", action="store_true", help="Forward to evaluate_tool_calls.py / evaluate_pass_rate.py")
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

    release_code = (args.fdb_repo_path / "FD3" / "release_code").resolve()
    benchmark_json = release_code / "benchmark_data_v2.json"
    if not benchmark_json.exists():
        sys.exit(f"benchmark_data_v2.json not found: {benchmark_json}")

    merged_root = args.merged_run_root.resolve()
    merged_root.mkdir(parents=True, exist_ok=True)
    shard_roots = [p.resolve() for p in args.shard_run_roots]
    print(f"Merging {len(shard_roots)} shard(s) into {merged_root}/per_sample_data/...")
    merged_results = _merge_shards(shard_roots, merged_root, args.provider)

    reports_dir = merged_root / "reports"
    print(f"Running FD3 evaluators against merged tree -> {reports_dir}/...")
    _run_evaluators(
        release_code=release_code,
        benchmark_json=benchmark_json,
        results_dir=merged_results,
        reports_dir=reports_dir,
        provider=args.provider,
        use_llm=args.use_llm,
    )

    # Pull checkpoint_dir off the first available shard's model_report for the
    # summary; if none have one, fall back to whatever the orchestrator wrote
    # to one of the shards.
    checkpoint_dir = ""
    for shard_root in shard_roots:
        per_shard_report = _load_json(shard_root / "model_report.json")
        if per_shard_report and per_shard_report.get("checkpoint_dir"):
            checkpoint_dir = per_shard_report["checkpoint_dir"]
            break

    consolidated = _build_consolidated_report(
        merged_root=merged_root,
        reports_dir=reports_dir,
        results_dir=merged_results,
        provider=args.provider,
        run_label=args.run_label or args.provider,
        checkpoint_dir=checkpoint_dir,
    )
    model_report_path = merged_root / "model_report.json"
    model_report_path.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote consolidated report: {model_report_path}")

    eval_report = consolidated.get("eval")
    pass_report = consolidated.get("pass_rate")
    latency_report = consolidated.get("latency")
    if eval_report is None and pass_report is None and latency_report is None:
        sys.exit(
            f"FDB v3 ChenChen scoring produced no reports. Expected outputs under {reports_dir}/."
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
    metrics[BENCHMARK_KEY]["fd3_run_root"] = str(merged_root)
    metrics[BENCHMARK_KEY]["num_shards"] = len(shard_roots)
    metrics[BENCHMARK_KEY]["num_result_files"] = consolidated.get("num_result_files", 0)
    metrics[BENCHMARK_KEY]["model_name"] = consolidated.get("model_name", args.run_label)
    if checkpoint_dir:
        metrics[BENCHMARK_KEY]["checkpoint_dir"] = checkpoint_dir

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
