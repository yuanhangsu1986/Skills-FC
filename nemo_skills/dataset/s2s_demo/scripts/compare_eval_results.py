#!/usr/bin/env python3
"""
Compare multiple S2S demo and VoiceBench evaluation results and generate a Markdown report.

Usage:
    python compare_eval_results.py \
        --eval_folders \
            "host:/path/to/s2s_demo_eval:Model A" \
            "host:/path/to/voicebench_eval:Model A" \
            "host:/path/to/s2s_demo_eval2:Model B" \
            "host:/path/to/voicebench_eval2:Model B" \
        --output comparison_report.md

The script auto-detects whether each folder contains s2s_demo or voicebench results.
"""

import argparse
import json
import os
import subprocess
from typing import Optional


def load_json_local(path: str) -> Optional[dict]:
    """Load JSON from a local file."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_json_remote(host: str, path: str) -> Optional[dict]:
    """Load JSON from a remote file via SSH."""
    try:
        result = subprocess.run(["ssh", host, f"cat {path}"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def list_dir_remote(host: str, path: str) -> list[str]:
    """List directory contents via SSH."""
    try:
        result = subprocess.run(["ssh", host, f"ls {path}"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        return [x.strip() for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        return []


def list_dir_local(path: str) -> list[str]:
    """List local directory contents."""
    if not os.path.isdir(path):
        return []
    return os.listdir(path)


def count_samples_remote(host: str, path: str) -> int:
    """Count lines in output.jsonl via SSH."""
    try:
        result = subprocess.run(
            ["ssh", host, f"wc -l < {path}/output.jsonl"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def count_samples_local(path: str) -> int:
    """Count lines in local output.jsonl."""
    output_path = os.path.join(path, "output.jsonl")
    if not os.path.exists(output_path):
        return 0
    with open(output_path) as f:
        return sum(1 for _ in f)


def detect_eval_type(host: Optional[str], eval_results_path: str) -> str:
    """Detect if this is s2s_demo or voicebench based on folder names."""
    if host:
        items = list_dir_remote(host, eval_results_path)
    else:
        items = list_dir_local(eval_results_path)

    for item in items:
        if item.startswith("voicebench."):
            return "voicebench"
        if item.startswith("s2s_demo."):
            return "s2s_demo"
    return "unknown"


def load_s2s_demo_metrics(host: Optional[str], folder_path: str) -> dict:
    """Load s2s_demo metrics from folder."""
    eval_results = folder_path + "/eval-results" if not folder_path.endswith("/") else folder_path + "eval-results"

    if host:
        items = list_dir_remote(host, eval_results)
    else:
        items = list_dir_local(eval_results)

    # Find s2s_demo.* subfolder
    demo_folder = None
    for item in items:
        if item.startswith("s2s_demo."):
            demo_folder = item
            break

    if not demo_folder:
        return {}

    metrics_path = f"{eval_results}/{demo_folder}/metrics.json"
    if host:
        metrics = load_json_remote(host, metrics_path)
    else:
        metrics = load_json_local(metrics_path)

    if not metrics:
        return {}

    return extract_s2s_demo_metrics(metrics)


def extract_s2s_demo_metrics(metrics: dict) -> dict:
    """Extract key metrics from s2s_demo metrics dict."""
    dm = metrics.get("dataset_metrics", metrics)
    tt = dm.get("turn_taking", {})
    bi = dm.get("barge_in", {})
    bc = dm.get("backchanneling", {})
    us = dm.get("user_speech", {})
    ags = dm.get("agent_speech", {})
    llm = dm.get("llm_judge", {})

    return {
        "num_samples": dm.get("num_samples_evaluated", 0),
        "tt_latency_ms": tt.get("avg_latency_ms"),
        "tt_precision": tt.get("avg_precision"),
        "tt_recall": tt.get("avg_recall"),
        "tt_f1": tt.get("avg_f1"),
        "bi_success_rate": bi.get("avg_success_rate"),
        "bi_latency_ms": bi.get("avg_latency_ms"),
        "bc_accuracy": bc.get("avg_accuracy"),
        "user_wer": us.get("avg_wer"),
        "oob_ratio": us.get("out_of_bounds_word_ratio"),
        "agent_wer": ags.get("avg_wer"),
        "agent_cer": ags.get("avg_cer"),
        "hallucination_rate": ags.get("hallucination_rate"),
        "llm_judge_overall": llm.get("overall", {}).get("avg_rating"),
        "llm_judge_full": llm.get("full", {}).get("avg_rating"),
        "llm_judge_sounded": llm.get("sounded", {}).get("avg_rating"),
    }


def load_voicebench_metrics(host: Optional[str], folder_path: str) -> dict:
    """Load voicebench metrics from folder. Returns {subtest: {metric: value}}."""
    eval_results = folder_path + "/eval-results" if not folder_path.endswith("/") else folder_path + "eval-results"

    if host:
        items = list_dir_remote(host, eval_results)
    else:
        items = list_dir_local(eval_results)

    result = {"subtests": {}, "total_samples": 0}

    for item in items:
        if not item.startswith("voicebench."):
            continue

        subtest = item.replace("voicebench.", "")
        subtest_path = f"{eval_results}/{item}"
        metrics_path = f"{subtest_path}/metrics.json"

        # Count samples
        if host:
            samples = count_samples_remote(host, subtest_path)
            metrics = load_json_remote(host, metrics_path)
        else:
            samples = count_samples_local(subtest_path)
            metrics = load_json_local(metrics_path)

        result["total_samples"] += samples

        if metrics:
            # Format: {"voicebench.{subtest}": {"greedy": {...}}}
            key = f"voicebench.{subtest}"
            if key in metrics:
                greedy = metrics[key].get("greedy", {})
                if greedy:
                    result["subtests"][subtest] = {"metrics": greedy, "samples": samples}

    return result


def format_value(value, format_spec: str = ".1f", suffix: str = "", na_str: str = "N/A") -> str:
    """Format a value for display."""
    if value is None:
        return na_str
    try:
        return f"{value:{format_spec}}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def generate_s2s_demo_section(models: list[tuple[str, dict]]) -> list[str]:
    """Generate S2S Demo section of the report."""
    if not models:
        return []

    lines = []
    lines.append("## S2S Demo Evaluation\n")

    # Table header
    header = "| Metric | " + " | ".join(name for name, _ in models) + " |"
    separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    rows = [
        ("Samples Evaluated", "num_samples", "d", "", None),
        ("**Turn-Taking**", None, None, None, None),
        ("  Latency (ms) ↓", "tt_latency_ms", ".1f", "", False),
        ("  Precision (%) ↑", "tt_precision", ".1f", "", True),
        ("  Recall (%) ↑", "tt_recall", ".1f", "", True),
        ("  F1 (%) ↑", "tt_f1", ".1f", "", True),
        ("**Barge-In**", None, None, None, None),
        ("  Success Rate (%) ↑", "bi_success_rate", ".1f", "", True),
        ("  Latency (ms) ↓", "bi_latency_ms", ".1f", "", False),
        ("**Backchanneling**", None, None, None, None),
        ("  Accuracy (%) ↑", "bc_accuracy", ".1f", "", True),
        ("**User Speech (ASR)**", None, None, None, None),
        ("  WER (%) ↓", "user_wer", ".1f", "", False),
        ("  OOB Ratio ↓", "oob_ratio", ".3f", "", False),
        ("**Agent Speech (TTS)**", None, None, None, None),
        ("  WER (%) ↓", "agent_wer", ".1f", "", False),
        ("  CER (%) ↓", "agent_cer", ".1f", "", False),
        ("  Hallucination (%) ↓", "hallucination_rate", ".1f", "", False),
        ("**LLM Judge (1-5)**", None, None, None, None),
        ("  Overall Rating ↑", "llm_judge_overall", ".2f", "", True),
        ("  Full Response ↑", "llm_judge_full", ".2f", "", True),
        ("  Sounded Response ↑", "llm_judge_sounded", ".2f", "", True),
    ]

    for display_name, metric_key, fmt, suffix, higher_better in rows:
        if metric_key is None:
            row = f"| {display_name} |" + " |" * len(models)
            lines.append(row)
            continue

        values = []
        raw_values = []
        for _, m in models:
            val = m.get(metric_key)
            raw_values.append(val)
            values.append(format_value(val, fmt, suffix))

        if higher_better is not None:
            valid_vals = [(i, v) for i, v in enumerate(raw_values) if v is not None]
            if len(valid_vals) >= 2:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                values[best_idx] = f"**{values[best_idx]}**"

        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    lines.append("")
    return lines


def generate_voicebench_section(models: list[tuple[str, dict]]) -> list[str]:
    """Generate VoiceBench section of the report."""
    if not models:
        return []

    lines = []
    lines.append("## VoiceBench Evaluation\n")

    # Collect all subtests across all models
    all_subtests = set()
    for _, vb_data in models:
        all_subtests.update(vb_data.get("subtests", {}).keys())
    all_subtests = sorted(all_subtests)

    if not all_subtests:
        lines.append("*No VoiceBench results available.*\n")
        return lines

    # Total samples row
    lines.append("### Summary\n")
    header = "| Model | Total Samples | Subtests |"
    lines.append(header)
    lines.append("|---|---|---|")
    for name, vb_data in models:
        total = vb_data.get("total_samples", 0)
        num_subtests = len(vb_data.get("subtests", {}))
        lines.append(f"| {name} | {total} | {num_subtests} |")
    lines.append("")

    # Per-subtest metrics table
    lines.append("### Per-Subtest Metrics\n")

    # Build a unified table with all metrics
    # First, collect all unique metric names across all subtests
    all_metrics = set()
    for _, vb_data in models:
        for subtest, data in vb_data.get("subtests", {}).items():
            all_metrics.update(data.get("metrics", {}).keys())

    # Metric display info: (metric_key, higher_better)
    metric_info = {
        "acc": True,
        "gpt": True,
        "panda": True,
        "pedant": True,
        "bleu": True,
        "exact_match": True,
        "score": True,
        "fail": False,
        "wer": False,
        "cer": False,
    }

    for subtest in all_subtests:
        lines.append(f"#### {subtest}\n")

        # Check which models have this subtest
        model_data = []
        for name, vb_data in models:
            subtests = vb_data.get("subtests", {})
            if subtest in subtests:
                model_data.append((name, subtests[subtest]))
            else:
                model_data.append((name, None))

        # Collect metrics for this subtest
        subtest_metrics = set()
        for _, data in model_data:
            if data:
                subtest_metrics.update(data.get("metrics", {}).keys())

        if not subtest_metrics:
            lines.append("*No metrics available.*\n")
            continue

        # Build table
        header = "| Metric | " + " | ".join(name for name, _ in model_data) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(model_data) + 1)) + "|")

        # Samples row
        samples_row = "| Samples |"
        for _, data in model_data:
            if data:
                samples_row += f" {data.get('samples', 'N/A')} |"
            else:
                samples_row += " N/A |"
        lines.append(samples_row)

        # Metric rows
        for metric in sorted(subtest_metrics):
            higher_better = metric_info.get(metric.lower(), True)
            arrow = "↑" if higher_better else "↓"

            values = []
            raw_values = []
            for _, data in model_data:
                if data and data.get("metrics"):
                    val = data["metrics"].get(metric)
                else:
                    val = None
                raw_values.append(val)
                values.append(format_value(val, ".2f"))

            # Highlight best
            valid_vals = [(i, v) for i, v in enumerate(raw_values) if v is not None]
            if len(valid_vals) >= 2:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                values[best_idx] = f"**{values[best_idx]}**"

            row = f"| {metric} {arrow} | " + " | ".join(values) + " |"
            lines.append(row)

        lines.append("")

    return lines


def parse_folder_spec(spec: str) -> tuple[Optional[str], str, str]:
    """Parse folder specification: 'path:name', 'host:path:name', 'path', 'host:path'."""
    parts = spec.split(":")

    if len(parts) == 1:
        path = parts[0]
        name = os.path.basename(path.rstrip("/"))
        return None, path, name

    if len(parts) == 2:
        if parts[0].startswith("/") or parts[0].startswith("."):
            return None, parts[0], parts[1]
        else:
            host, path = parts[0], parts[1]
            name = os.path.basename(path.rstrip("/"))
            return host, path, name

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    if len(parts) > 3:
        host = parts[0]
        name = parts[-1]
        path = ":".join(parts[1:-1])
        return host, path, name

    return None, spec, os.path.basename(spec.rstrip("/"))


def generate_report(
    s2s_demo_models: list[tuple[str, dict]], voicebench_models: list[tuple[str, dict]], output_path: str
):
    """Generate the full comparison report."""
    lines = []
    lines.append("# Evaluation Comparison Report\n")

    # Collect unique model names
    all_models = set()
    for name, _ in s2s_demo_models:
        all_models.add(name)
    for name, _ in voicebench_models:
        all_models.add(name)

    lines.append(f"Comparing {len(all_models)} model(s):\n")
    for name in sorted(all_models):
        lines.append(f"- {name}")
    lines.append("")

    # S2S Demo section
    if s2s_demo_models:
        lines.extend(generate_s2s_demo_section(s2s_demo_models))

    # VoiceBench section
    if voicebench_models:
        lines.extend(generate_voicebench_section(voicebench_models))

    # Legend
    lines.append("---")
    lines.append("*↑ = higher is better, ↓ = lower is better, **bold** = best value*")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Compare S2S demo and VoiceBench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare s2s_demo and voicebench results for two models:
    python compare_eval_results.py \\
        --eval_folders \\
            "host:/path/to/model_a/s2s_demo:Model A" \\
            "host:/path/to/model_a/voicebench:Model A" \\
            "host:/path/to/model_b/s2s_demo:Model B" \\
            "host:/path/to/model_b/voicebench:Model B" \\
        --output comparison_report.md

Each folder should contain eval-results/ with either s2s_demo.* or voicebench.* subfolders.
        """,
    )
    parser.add_argument(
        "--eval_folders",
        nargs="+",
        required=True,
        help="Evaluation folders: 'path:name', 'host:path:name', or just 'path' / 'host:path'",
    )
    parser.add_argument("--output", type=str, default="comparison_report.md", help="Output Markdown file")

    args = parser.parse_args()

    s2s_demo_models = []
    voicebench_models = []

    for spec in args.eval_folders:
        host, folder_path, model_name = parse_folder_spec(spec)
        eval_results_path = (
            folder_path + "/eval-results" if not folder_path.endswith("/") else folder_path + "eval-results"
        )

        location_str = f"{host}:{folder_path}" if host else folder_path

        eval_type = detect_eval_type(host, eval_results_path)

        if eval_type == "s2s_demo":
            metrics = load_s2s_demo_metrics(host, folder_path)
            if metrics:
                s2s_demo_models.append((model_name, metrics))
                print(f"Loaded s2s_demo metrics for {model_name} from {location_str}")
            else:
                print(f"Warning: No s2s_demo metrics found in {location_str}")

        elif eval_type == "voicebench":
            metrics = load_voicebench_metrics(host, folder_path)
            if metrics.get("subtests"):
                voicebench_models.append((model_name, metrics))
                n_subtests = len(metrics["subtests"])
                print(f"Loaded voicebench metrics for {model_name} from {location_str} ({n_subtests} subtests)")
            else:
                print(f"Warning: No voicebench metrics found in {location_str}")

        else:
            print(f"Warning: Could not detect eval type for {location_str}")

    if not s2s_demo_models and not voicebench_models:
        print("Error: No valid metrics found.")
        return 1

    generate_report(s2s_demo_models, voicebench_models, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
