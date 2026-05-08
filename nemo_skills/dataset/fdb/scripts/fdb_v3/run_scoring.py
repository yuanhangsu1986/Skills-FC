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

"""FD3 (FDB v3) scoring: reconstruct FD3-shaped per-sample result files from a
nemo-skills `output.jsonl`, then invoke the vendored FDBV3 evaluators
(evaluate_tool_calls.py / evaluate_pass_rate.py / analyze_tool_latency.py) and
merge the three reports into a single `metrics.json` keyed by `fdb_v3.tool_call`.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _alias_judge_api_key():
    """Cluster env sets NV_INFERENCE_KEY, but FD3 evaluators read NVIDIA_API_KEY /
    OPENAI_API_KEY. Mirror the value over so the LLM judge client can authenticate."""
    nv_key = os.environ.get("NV_INFERENCE_KEY")
    if not nv_key:
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = nv_key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = nv_key


_alias_judge_api_key()

DEFAULT_FDB_REPO = Path("/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/NeMo/FDBV3_CHENCHEN")
TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_tool_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("function")
    args = item.get("arguments", item.get("args", {}))
    fn_obj = item.get("function")
    if isinstance(fn_obj, dict):
        name = fn_obj.get("name") or name
        args = fn_obj.get("arguments", args)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw_arguments": args}
    if args is None:
        args = {}
    if not name:
        return None
    return {"function": str(name), "args": args}


def _parse_toolcall_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    seen: set[str] = set()
    calls: list[dict[str, Any]] = []
    payloads = [m.group(1).strip() for m in TOOLCALL_RE.finditer(text)]
    payloads.extend(part.split("</TOOLCALL>", 1)[0].strip() for part in text.split("<TOOLCALL>")[1:])
    for payload in payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            call = _normalize_tool_call(item)
            if call is None:
                continue
            key = json.dumps(call, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            calls.append(call)
    return calls


def _measure_latency(input_wav: Path, output_wav: Path, silence_threshold_db: int = -40) -> dict:
    try:
        from pydub import AudioSegment

        in_audio = AudioSegment.from_file(str(input_wav))
        out_audio = AudioSegment.from_file(str(output_wav))
        first_speech_ms = None
        chunk_ms = 50
        for i in range(0, len(out_audio), chunk_ms):
            if out_audio[i : i + chunk_ms].dBFS > silence_threshold_db:
                first_speech_ms = i
                break
        first_speech_s = first_speech_ms / 1000.0 if first_speech_ms is not None else len(out_audio) / 1000.0
        return {
            "input_duration_s": round(len(in_audio) / 1000.0, 3),
            "output_duration_s": round(len(out_audio) / 1000.0, 3),
            "first_speech_s": round(first_speech_s, 3),
        }
    except Exception as e:
        return {"error": str(e)}


def _detect_speech_bounds(audio_path: Path, silence_threshold_db: int = -40) -> dict:
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(str(audio_path))
        regions = detect_nonsilent(audio, min_silence_len=120, silence_thresh=silence_threshold_db, seek_step=10)
        duration_s = round(len(audio) / 1000.0, 3)
        if not regions:
            return {"duration_s": duration_s, "first_speech_s": None, "last_speech_s": None}
        return {
            "duration_s": duration_s,
            "first_speech_s": round(regions[0][0] / 1000.0, 3),
            "last_speech_s": round(regions[-1][1] / 1000.0, 3),
        }
    except Exception as e:
        return {"duration_s": None, "first_speech_s": None, "last_speech_s": None, "error": str(e)}


def _extract_text_from_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Return (function_channel_text, actual_tool_calls, transcript) from an output.jsonl row."""
    generation = str(row.get("generation") or "")
    serialized = row.get("serialized_output") or []
    transcript = ""
    if isinstance(serialized, list):
        for msg in serialized:
            if not isinstance(msg, dict):
                continue
            audio = msg.get("audio") or {}
            if isinstance(audio, dict) and audio.get("transcript"):
                transcript = audio["transcript"]
                break
            if msg.get("content"):
                transcript = transcript or msg["content"]
    if not transcript:
        audio = row.get("audio") or {}
        if isinstance(audio, dict):
            transcript = audio.get("transcript", "")
    if not transcript:
        transcript = generation

    function_text = generation
    actual_calls = _parse_toolcall_text(generation)
    if not actual_calls and isinstance(serialized, list):
        for msg in serialized:
            if not isinstance(msg, dict):
                continue
            tc = msg.get("tool_calls") or []
            for entry in tc or []:
                fn = entry.get("function") if isinstance(entry, dict) else None
                if isinstance(fn, dict):
                    actual_calls.append(_normalize_tool_call(fn))
        actual_calls = [c for c in actual_calls if c]
    return function_text, actual_calls, transcript


def _reconstruct_fd3_layout(
    output_jsonl: Path,
    fd3_data_root: Path,
    layout_root: Path,
    provider: str,
    skip_latency: bool,
) -> int:
    """Materialize layout_root/{example_id}_{speaker_id}/result_<provider>.json
    plus an output_<provider>.wav copy when the generated audio path is reachable.
    Returns the number of result files written.
    """
    layout_root.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            example_id = row.get("example_id")
            speaker_id = row.get("speaker_id")
            if not (sample_id and example_id and speaker_id):
                continue

            sample_dir = layout_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            fd3_input_dir = fd3_data_root / sample_id
            input_wav = fd3_input_dir / "input.wav"

            function_text, actual_calls, transcript = _extract_text_from_row(row)

            audio_meta = row.get("audio") or {}
            output_wav_src = Path(audio_meta["path"]) if isinstance(audio_meta, dict) and audio_meta.get("path") else None
            output_wav_dst = sample_dir / f"output_{provider}.wav"
            if output_wav_src and output_wav_src.exists():
                if not output_wav_dst.exists() or output_wav_dst.stat().st_size != output_wav_src.stat().st_size:
                    shutil.copy2(output_wav_src, output_wav_dst)

            latency = {}
            user_speech_end_rel = None
            agent_speech_start_rel = None
            perceived_total_latency = None
            if not skip_latency and input_wav.exists() and output_wav_dst.exists():
                latency = _measure_latency(input_wav, output_wav_dst)
                bounds_in = _detect_speech_bounds(input_wav)
                bounds_out = _detect_speech_bounds(output_wav_dst)
                user_speech_end_rel = bounds_in.get("last_speech_s")
                agent_speech_start_rel = bounds_out.get("first_speech_s")
                if user_speech_end_rel is not None and agent_speech_start_rel is not None:
                    perceived_total_latency = round(agent_speech_start_rel - user_speech_end_rel, 3)

            result = {
                "pid": speaker_id,
                "example_id": example_id,
                "category": row.get("domain", "unknown"),
                "title": row.get("title", example_id),
                "provider": provider,
                "evaluated_at": datetime.datetime.now().isoformat(),
                "tool_call_source": "s2s_function_head",
                "backend_used": False,
                "function_channel_text": function_text,
                "function_channel_text_raw": function_text,
                "actual_tool_calls": actual_calls,
                "function_call_count": len(actual_calls),
                "s2s_pred_text": transcript,
                "transcript": transcript,
                "output_transcript_source": "s2s_pred_text_fallback",
                "agent_audio_detected": agent_speech_start_rel is not None,
                "user_speech_end_rel": user_speech_end_rel,
                "audio_agent_speech_start": agent_speech_start_rel,
                "perceived_total_latency": perceived_total_latency,
                "absolute_total_latency": perceived_total_latency,
                "latency": latency,
                "status": "completed",
            }
            (sample_dir / f"result_{provider}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            written += 1
    return written


def _run_evaluator(release_code: Path, script_name: str, args_list: list[str], log_prefix: str) -> dict | None:
    """Run a vendored FD3 evaluator and return its parsed JSON report.

    The upstream `evaluate_tool_calls.py` writes its JSON report before
    printing a summary that can crash on edge cases (e.g. N=0 turn-taken
    samples). We always try to recover the report from disk regardless of
    the script's exit code.
    """
    script = release_code / script_name
    if not script.exists():
        print(f"  warning: {script_name} not found at {script}; skipping {log_prefix}")
        return None
    cmd = [sys.executable, str(script), *args_list]
    print(f"  running {log_prefix}: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(release_code), capture_output=True, text=True)
    print(res.stdout)

    output_idx = args_list.index("--output") if "--output" in args_list else -1
    report_path = Path(args_list[output_idx + 1]) if output_idx >= 0 and output_idx + 1 < len(args_list) else None
    report_written = report_path.exists() if report_path else False

    if res.returncode != 0 and not report_written:
        print(res.stderr, file=sys.stderr)
        print(f"  warning: {log_prefix} exited with code {res.returncode} and produced no report")
    elif res.returncode != 0:
        print(f"  note: {log_prefix} exited with code {res.returncode} but the report was written; continuing")

    if report_written:
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  warning: could not parse {report_path}: {e}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FD3 (FDB v3) scoring entry point")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument(
        "--fdb_repo",
        type=Path,
        default=DEFAULT_FDB_REPO,
        help="Vendored FDBV3_CHENCHEN root (containing FD3/release_code/ and FD3/fdb_v3_data_released/)",
    )
    parser.add_argument("--provider", type=str, default="fdb_v3")
    parser.add_argument("--use_llm_judge", action="store_true", help="Pass --use-llm to the FD3 evaluators")
    parser.add_argument("--skip_latency", action="store_true", help="Skip audio-derived latency computation")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json already populated")
    args = parser.parse_args()

    eval_results_dir = args.eval_results_dir.resolve()
    metrics_file = eval_results_dir / "metrics.json"
    benchmark_key = "fdb_v3.tool_call"

    if metrics_file.exists() and not args.force:
        try:
            existing = json.loads(metrics_file.read_text(encoding="utf-8"))
            if existing.get(benchmark_key):
                print(f"Scoring already done for {benchmark_key}. Skipping (use --force to re-run).")
                return
        except Exception:
            pass

    output_jsonl = eval_results_dir / "output.jsonl"
    if not output_jsonl.exists():
        sys.exit(f"output.jsonl not found at {output_jsonl}")

    release_code = args.fdb_repo / "FD3" / "release_code"
    fd3_data_root = args.fdb_repo / "FD3" / "fdb_v3_data_released"
    benchmark_json = release_code / "benchmark_data_v2.json"
    for path, label in [(release_code, "release_code"), (fd3_data_root, "fdb_v3_data_released"), (benchmark_json, "benchmark_data_v2.json")]:
        if not path.exists():
            sys.exit(f"Required FD3 path missing ({label}): {path}")

    layout_root = eval_results_dir / "fdb_v3_layout"
    print(f"Reconstructing FD3 layout under {layout_root}")
    n_results = _reconstruct_fd3_layout(
        output_jsonl, fd3_data_root, layout_root, args.provider, args.skip_latency
    )
    print(f"Wrote {n_results} result_{args.provider}.json files")

    report_dir = eval_results_dir / "summarized-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_out = report_dir / f"{args.provider}_eval.json"
    pass_out = report_dir / f"{args.provider}_pass_rate.json"
    latency_out = report_dir / f"{args.provider}_latency.json"

    llm_flag = ["--use-llm"] if args.use_llm_judge else []

    eval_report = _run_evaluator(
        release_code,
        "evaluate_tool_calls.py",
        [
            "--benchmark", str(benchmark_json),
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(eval_out),
            *llm_flag,
        ],
        "evaluate_tool_calls",
    )
    pass_report = _run_evaluator(
        release_code,
        "evaluate_pass_rate.py",
        [
            "--benchmark", str(benchmark_json),
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(pass_out),
            *llm_flag,
        ],
        "evaluate_pass_rate",
    )
    latency_report = _run_evaluator(
        release_code,
        "analyze_tool_latency.py",
        [
            "--results-dir", str(layout_root),
            "--provider", args.provider,
            "--output", str(latency_out),
        ],
        "analyze_tool_latency",
    )

    metrics: dict[str, Any] = {benchmark_key: {}}
    if eval_report is not None:
        metrics[benchmark_key]["eval"] = eval_report
    if pass_report is not None:
        metrics[benchmark_key]["pass_rate"] = pass_report
    if latency_report is not None:
        metrics[benchmark_key]["latency"] = latency_report

    if eval_report and "by_metric" in eval_report:
        bm = eval_report["by_metric"]
        metrics[benchmark_key]["headline"] = {
            "tool_selection_acc": bm.get("tool_selection_acc"),
            "argument_acc": bm.get("argument_acc"),
            "response_qual": bm.get("response_qual"),
            "turn_take_rate": eval_report.get("turn_taking", {}).get("turn_take_rate"),
            "total_scenarios": eval_report.get("total_scenarios"),
        }
    metrics[benchmark_key]["num_result_files"] = n_results
    metrics[benchmark_key]["provider"] = args.provider
    metrics[benchmark_key]["fdb_repo"] = str(args.fdb_repo)

    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_file}")


if __name__ == "__main__":
    main()
