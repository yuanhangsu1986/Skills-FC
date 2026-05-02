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
FDB scoring: prepare (copy audio to fdb_prepared) -> run ASR -> run FDB evaluate -> write metrics.json.
Used by run_eval.py.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ASR_TASK_MAP = {
    "pause": "full",
    "pause_candor": "full",
    "pause_synthetic": "full",
    "backchannel": "full",
    "turn_taking": "full",
    "interruption": "user_interruption",
    "background_speech": "full",
    "talking_to_other": "full",
}
FDB_TASK_MAP = {
    "pause": "pause_handling",
    "pause_candor": "pause_handling",
    "pause_synthetic": "pause_handling",
    "backchannel": "backchannel",
    "turn_taking": "smooth_turn_taking",
    "interruption": "user_interruption",
    "background_speech": "behavior",
    "talking_to_other": "behavior",
}
# v1.5 uses behavior eval for backchannel and interruption (not the v1.0 JSD/TOR-based evals)
FDB_TASK_MAP_V1_5 = {
    **FDB_TASK_MAP,
    "backchannel": "behavior",
    "interruption": "behavior",
}
# v1.5 interruption needs full ASR (not user_interruption which crops to post-interrupt)
ASR_TASK_MAP_V1_5 = {
    **ASR_TASK_MAP,
    "interruption": "full",
}


def _convert_stereo_to_mono(fdb_prepared: Path):
    """Convert stereo output.wav files to mono (model channel) for Silero-VAD compatibility."""
    try:
        import soundfile as sf_mod
    except ImportError:
        print("Warning: soundfile not available, skipping stereo→mono conversion for timing.")
        return
    for sample_dir in sorted(fdb_prepared.iterdir()):
        out_wav = sample_dir / "output.wav"
        if not out_wav.exists():
            continue
        try:
            data, sr = sf_mod.read(str(out_wav))
            if data.ndim == 2 and data.shape[1] >= 2:
                mono = data[:, 1]  # ch1 = model channel
                sf_mod.write(str(out_wav), mono, sr)
        except Exception as e:
            print(f"Warning: could not convert {out_wav} to mono: {e}")


def main():
    parser = argparse.ArgumentParser(description="FDB prepare + ASR + evaluate -> metrics.json")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument("--fdb_repo", type=Path, required=True)
    parser.add_argument("--subtest", required=True, choices=list(ASR_TASK_MAP))
    parser.add_argument("--fdb_data_path", type=Path, default=None, help="FDB dataset root; required for turn_taking (turn_taking.json) and interruption (interrupt.json)")
    parser.add_argument("--fdb_version", default="v1.0", choices=["v1.0", "v1.5"], help="FDB dataset version (metadata paths and metrics key)")
    parser.add_argument("--decoding_mode", default="greedy", choices=["greedy", "sampling"], help="Key under which metrics are stored in metrics.json")
    parser.add_argument("--force", action="store_true", help="Re-run scoring even if metrics.json exists")
    args = parser.parse_args()

    eval_results_dir = args.eval_results_dir.resolve()
    fdb_repo = args.fdb_repo.resolve()
    metrics_file = eval_results_dir / "metrics.json"
    benchmark_key = f"fdb_v1_5.{args.subtest}" if args.fdb_version == "v1.5" else f"fdb_v1.{args.subtest}"

    if metrics_file.exists() and not args.force:
        try:
            existing = json.loads(metrics_file.read_text())
            if decoding_mode in existing.get(benchmark_key, {}):
                print(f"Scoring already done for {benchmark_key} ({decoding_mode}). Skipping (use --force to re-run).")
                sys.exit(0)
        except Exception:
            pass

    if not eval_results_dir.exists() or not fdb_repo.exists():
        print("Error: eval_results_dir or fdb_repo not found.")
        sys.exit(1)
    if not (eval_results_dir / "output.jsonl").exists():
        print("Error: output.jsonl not found.")
        sys.exit(1)

    # turn_taking, interruption, background_speech, talking_to_other need fdb_data_path for metadata / input wavs
    if args.subtest in ("turn_taking", "interruption", "background_speech", "talking_to_other") and (
        args.fdb_data_path is None or not args.fdb_data_path.exists()
    ):
        print(
            f"Error: --fdb_data_path is required for subtest '{args.subtest}' "
            "(FDB dataset root; used to copy task metadata and, for background_speech/talking_to_other, input.wav and clean_input.wav)."
        )
        sys.exit(1)

    asr_map = ASR_TASK_MAP_V1_5 if args.fdb_version == "v1.5" else ASR_TASK_MAP
    asr_task = asr_map[args.subtest]
    task_map = FDB_TASK_MAP_V1_5 if args.fdb_version == "v1.5" else FDB_TASK_MAP
    fdb_task = task_map[args.subtest]
    prep_script = Path(__file__).resolve().parent / "prepare_fdb_eval_dir.py"

    prep_cmd = [
        sys.executable, str(prep_script),
        "--eval_results_dir", str(eval_results_dir),
        "--fdb_repo", str(fdb_repo),
        "--run_asr", "--asr_task", asr_task,
        "--subtest", args.subtest,
        "--fdb_version", args.fdb_version,
    ]
    if args.fdb_data_path is not None:
        prep_cmd.extend(["--fdb_data_path", str(args.fdb_data_path)])
    subprocess.run(prep_cmd, check=True)

    fdb_prepared = eval_results_dir / "fdb_prepared"
    evaluate_script = fdb_repo / "evaluation" / "evaluate.py"
    if not evaluate_script.exists():
        print(f"Error: {evaluate_script} not found")
        sys.exit(1)
    # backchannel eval (v1.0) runs Silero VAD directly on output.wav — needs mono (ch1=model channel).
    # ASR already ran in prep_cmd above, so converting here does not affect output.json transcripts.
    if args.subtest == "backchannel" and args.fdb_version == "v1.0":
        _convert_stereo_to_mono(fdb_prepared)
    # Run from evaluation/ so FDB scripts find ./icc_gt_distribution.json (backchannel) and other relative paths.
    # Pass through env so NVIDIA_API_KEY is available for interruption/behavior tasks (NVIDIA NIM API).
    result = subprocess.run(
        [sys.executable, str(evaluate_script), "--task", fdb_task, "--root_dir", str(fdb_prepared)],
        cwd=str(fdb_repo / "evaluation"), capture_output=True, text=True, env=os.environ.copy(),
    )
    stdout, stderr = result.stdout, result.stderr
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    metrics = {}
    combined = stdout + "\n" + stderr
    # Extract explicitly known FDB metric lines
    # pause -> TOR %; backchannel -> JSD, TOR %, Frequency; turn_taking -> TOR %, latency_ms; interruption -> rating (GPT), TOR %, latency_ms
    explicit_metrics = [
        ("JSD - Mean", "jsd"),
        ("TOR - Mean", "tor"),
        ("Frequency - Mean", "frequency"),
        ("Average take turn", "turn"),
        ("Average latency", "latency"),
        ("Average rating", "rating"),  # GPT/LLM judge score (interruption only)
    ]
    for name, key in explicit_metrics:
        m = re.search(rf"{re.escape(name)}\s*(?:\(s\))?\s*:\s*([0-9.]+)", combined)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass
    # TOR as percentage (0-100)
    if "turn" in metrics:
        metrics["tor_pct"] = round(metrics["turn"] * 100, 2)
    if "tor" in metrics:
        metrics["tor_pct"] = round(metrics["tor"] * 100, 2)
    # Latency in ms for turn_taking and interruption
    if "latency" in metrics:
        metrics["latency_ms"] = round(metrics["latency"] * 1000, 2)
    # Behavior eval (background_speech, talking_to_other): "Ratios (C-axis): {'C_RESPOND': 0.8, 'C_RESUME': 0.2}"
    ratios_match = re.search(r"Ratios \(C-axis\):\s*(.+)", combined)
    if ratios_match:
        raw = ratios_match.group(1).strip().split("\n")[0].strip()
        # Trim to balanced {...} (in case of trailing text)
        if raw.startswith("{"):
            end = raw.rfind("}")
            if end != -1:
                raw = raw[: end + 1]
        try:
            behavior_ratios = ast.literal_eval(raw)
            if isinstance(behavior_ratios, dict):
                metrics["behavior_ratios"] = {k: round(float(v), 4) for k, v in behavior_ratios.items()}
                for k, v in behavior_ratios.items():
                    metrics[f"behavior_{k}"] = round(float(v), 4)
                # Add missing C_* keys as 0 for consistent schema
                for key in ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN"):
                    if key not in metrics["behavior_ratios"]:
                        metrics["behavior_ratios"][key] = 0.0
                        metrics[f"behavior_{key}"] = 0.0
        except (ValueError, SyntaxError):
            pass
    # --- Timing metrics (Stop Latency & Response Latency) for v1.5 ---
    # get_timing.py uses Silero-VAD on input.wav / output.wav to compute overlap and gap intervals.
    # Silero-VAD expects mono; output.wav may be stereo (ch0=user, ch1=model) so convert to mono first.
    if args.fdb_version == "v1.5":
        timing_script = fdb_repo / "evaluation" / "get_timing.py"
        if timing_script.exists():
            _convert_stereo_to_mono(fdb_prepared)
            print(f"Running timing analysis (get_timing.py) on {fdb_prepared} ...")
            timing_result = subprocess.run(
                [sys.executable, str(timing_script), "--root_dir", str(fdb_prepared)],
                cwd=str(fdb_repo / "evaluation"), capture_output=True, text=True,
            )
            print(timing_result.stdout)
            if timing_result.stderr:
                print(timing_result.stderr, file=sys.stderr)

            stop_durations = []
            resp_durations = []
            for sample_dir in sorted(fdb_prepared.iterdir()):
                lat_file = sample_dir / "latency_intervals.json"
                if not lat_file.exists():
                    continue
                with open(lat_file, "r") as lf:
                    lat_data = json.load(lf)
                for s, e in lat_data.get("latency_stop_list", []):
                    stop_durations.append(e - s)
                for s, e in lat_data.get("latency_resp_list", []):
                    resp_durations.append(e - s)

            if stop_durations:
                metrics["stop_latency"] = round(sum(stop_durations) / len(stop_durations), 4)
                metrics["stop_latency_ms"] = round(metrics["stop_latency"] * 1000, 2)
            if resp_durations:
                metrics["response_latency"] = round(sum(resp_durations) / len(resp_durations), 4)
                metrics["response_latency_ms"] = round(metrics["response_latency"] * 1000, 2)
        else:
            print(f"Warning: {timing_script} not found, skipping timing metrics.")

    if not metrics:
        metrics["status"] = "no_metrics_found"

    existing_metrics = {}
    if metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass
    existing_metrics.setdefault(benchmark_key, {})[args.decoding_mode] = metrics
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    print(f"Metrics written to {metrics_file}")



if __name__ == "__main__":
    main()
