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
Scoring wrapper for conv_behav evaluation.

Calls eval_conversation_behavior.py (eval_script_path),
parses its printed output, and writes metrics.json under output_dir.

Metrics written:
  {
    "conv_behav": {
      "tt_latency_ms": float,
      "tt_precision": float,
      "tt_recall": float,
      "tt_f1": float,
      "barge_in_success_rate": float,
      "barge_in_latency_ms": float,
      "bc_accuracy": float,
      "cutoff_rate": float,
      "user_eou_precision": float,   # present only if computed
      "user_eou_recall": float,
      "user_eou_f1": float,
      "user_eou_latency_ms": float,
      "user_wer": float,             # present only if computed
      "num_evaluated": int
    }
  }

Usage:
    python run_scoring.py \
        --output_dir /path/to/output \
        --shar_input_dir /path/to/lhotse_shar \
        --dataset_name team_20251124 \
        --eval_script_path /path/to/NeMo/scripts/speech_eval/eval_conversation_behavior.py \
        [--barge_in_threshold_sec 1.5] \
        [--tt_latency_threshold_sec 1.5] \
        [--tt_precision_buffer_sec 1.0] \
        [--tt_recall_buffer_sec 20.0] \
        [--vad_min_silence_duration_ms 2000] \
        [--force]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _parse_metrics(output: str) -> dict:
    """Parse the Average Metrics block from eval_conversation_behavior.py stdout."""
    output = _strip_ansi(output)

    # Isolate the Average Metrics block (between two ==...== lines after "Average Metrics")
    block_match = re.search(
        r"={20,}\s*Average Metrics.*?={20,}",
        output,
        re.DOTALL,
    )
    if not block_match:
        print("[scoring] Warning: could not find Average Metrics block in output.", file=sys.stderr)
        return {}
    block = block_match.group(0)

    metrics = {}

    # -- Turn-taking (section 1) --
    tt = re.search(r"1\.\s*Turn-taking:(.*?)(?=\d+\.\s|\Z)", block, re.DOTALL)
    if tt:
        t = tt.group(1)
        for key, pat in [
            ("tt_latency_ms",  r"Average latency:\s*([\d.]+)\s*ms"),
            ("tt_precision",   r"Precision:\s*([\d.]+)%"),
            ("tt_recall",      r"Recall:\s*([\d.]+)%"),
            ("tt_f1",          r"F1:\s*([\d.]+)%"),
        ]:
            m = re.search(pat, t)
            if m:
                metrics[key] = float(m.group(1))

    # -- Barge-in (section 2) --
    bi = re.search(r"2\.\s*User barge-in:(.*?)(?=\d+\.\s|\Z)", block, re.DOTALL)
    if bi:
        t = bi.group(1)
        m = re.search(r"Average success rate:\s*([\d.]+)%", t)
        if m:
            metrics["barge_in_success_rate"] = float(m.group(1))
        m = re.search(r"Average latency:\s*([\d.]+)\s*ms", t)
        if m:
            metrics["barge_in_latency_ms"] = float(m.group(1))

    # -- Back-channeling (section 3) --
    bc = re.search(r"3\.\s*Back-channeling:(.*?)(?=\d+\.\s|\Z)", block, re.DOTALL)
    if bc:
        m = re.search(r"Average accuracy:\s*([\d.]+)%", bc.group(1))
        if m:
            metrics["bc_accuracy"] = float(m.group(1))

    # -- Agent cutoff (section 4) --
    co = re.search(r"4\.\s*Agent cutoff detection:(.*?)(?=\d+\.\s|\Z)", block, re.DOTALL)
    if co:
        m = re.search(r"Cutoff rate:\s*([\d.]+)%", co.group(1))
        if m:
            metrics["cutoff_rate"] = float(m.group(1))

    # -- User EOU Detection (optional, section 5 or 6) --
    eou = re.search(r"User EOU Detection:(.*?)(?=\d+\.\s|\Z)", block, re.DOTALL)
    if eou:
        t = eou.group(1)
        for key, pat in [
            ("user_eou_precision",  r"Precision:\s*([\d.]+)%"),
            ("user_eou_recall",     r"Recall:\s*([\d.]+)%"),
            ("user_eou_f1",         r"F1:\s*([\d.]+)%"),
            ("user_eou_latency_ms", r"Average EOU latency:\s*([\d.]+)\s*ms"),
        ]:
            m = re.search(pat, t)
            if m:
                metrics[key] = float(m.group(1))

    # -- User WER (optional) --
    wer = re.search(r"User Speech Recognition.*?Average WER:\s*([\d.]+)%", block, re.DOTALL)
    if wer:
        metrics["user_wer"] = float(wer.group(1))

    # -- Number of audios evaluated --
    m = re.search(r"Number of audios evaluated:\s*(\d+)", block)
    if m:
        metrics["num_evaluated"] = int(m.group(1))

    return metrics


def score(
    output_dir: str,
    shar_input_dir: str,
    dataset_name: str,
    eval_script_path: str,
    barge_in_threshold_sec: float = 1.5,
    tt_latency_threshold_sec: float = 1.5,
    tt_precision_buffer_sec: float = 1.0,
    tt_recall_buffer_sec: float = 20.0,
    vad_min_silence_duration_ms: int = 2000,
    force: bool = False,
    torch_home: str = "",
) -> int:
    output_path = Path(output_dir)
    pred_audio_dir = output_path / "validation_logs" / "pred_wavs"
    jsonl_with_timestamp = output_path / "validation_logs" / "metadatas" / f"{dataset_name}.json"
    metrics_file = output_path / "metrics.json"

    benchmark_key = "conv_behav"

    if metrics_file.exists() and not force:
        try:
            existing = json.loads(metrics_file.read_text())
            if existing.get(benchmark_key):
                print(f"[scoring] Already done for {benchmark_key}. Skipping (use --force to re-run).")
                return 0
        except Exception:
            pass

    if not jsonl_with_timestamp.exists():
        print(f"[scoring] Error: {jsonl_with_timestamp} not found. Run inference first.", file=sys.stderr)
        return 1

    eval_script = Path(eval_script_path)
    if not eval_script.exists():
        print(f"[scoring] Error: {eval_script} not found.", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, str(eval_script),
        "--pred_audio_dir", str(pred_audio_dir),
        "--shar_input_dir", str(shar_input_dir),
        "--jsonl_with_timestamp", str(jsonl_with_timestamp),
        "--validation_set_names", dataset_name,
        "--barge_in_threshold_sec", str(barge_in_threshold_sec),
        "--tt_latency_threshold_sec", str(tt_latency_threshold_sec),
        "--tt_precision_buffer_sec", str(tt_precision_buffer_sec),
        "--tt_recall_buffer_sec", str(tt_recall_buffer_sec),
        "--vad_min_silence_duration_ms", str(vad_min_silence_duration_ms),
        "--end_time", "None",
        "--enable_transcription",
        "--compute_user_eou",
        "--verbose",
    ]

    print(f"[scoring] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    if torch_home:
        env["TORCH_HOME"] = torch_home
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(result.stdout)

    if result.returncode != 0:
        print(f"[scoring] Error: eval script exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    metrics = _parse_metrics(result.stdout)
    if not metrics:
        print("[scoring] Warning: no metrics parsed from output.", file=sys.stderr)
        return 1

    existing_metrics = {}
    if metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    existing_metrics[benchmark_key] = metrics
    metrics_file.write_text(json.dumps(existing_metrics, indent=2))

    print(f"\n[scoring] Metrics saved to {metrics_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Score conv_behav inference output")
    parser.add_argument("--output_dir", required=True, help="Root output dir (contains validation_logs/)")
    parser.add_argument("--shar_input_dir", required=True, help="Lhotse shar directory with user audio")
    parser.add_argument("--dataset_name", required=True, help="Dataset name used during inference")
    parser.add_argument("--eval_script_path", required=True, help="Path to eval_conversation_behavior.py")
    parser.add_argument("--barge_in_threshold_sec", type=float, default=1.5)
    parser.add_argument("--tt_latency_threshold_sec", type=float, default=1.5)
    parser.add_argument("--tt_precision_buffer_sec", type=float, default=1.0)
    parser.add_argument("--tt_recall_buffer_sec", type=float, default=20.0)
    parser.add_argument("--vad_min_silence_duration_ms", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json exists")
    parser.add_argument("--torch_home", default="", help="Override TORCH_HOME for the eval subprocess (used to point torch.hub at a pre-cached model dir on lustre)")
    args = parser.parse_args()

    sys.exit(score(
        output_dir=args.output_dir,
        shar_input_dir=args.shar_input_dir,
        dataset_name=args.dataset_name,
        eval_script_path=args.eval_script_path,
        barge_in_threshold_sec=args.barge_in_threshold_sec,
        tt_latency_threshold_sec=args.tt_latency_threshold_sec,
        tt_precision_buffer_sec=args.tt_precision_buffer_sec,
        tt_recall_buffer_sec=args.tt_recall_buffer_sec,
        vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
        force=args.force,
        torch_home=args.torch_home,
    ))


if __name__ == "__main__":
    main()
