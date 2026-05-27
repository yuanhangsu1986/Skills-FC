# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Re-score existing BFCL inference outputs with the AU-Harness BFCLMatchScore
algorithm, byte-for-byte (loaded via importlib from AU-Harness on disk).

Why this exists: our in-tree scorer (`score.py`) is stricter than AU-Harness on
scalar parameter checks. To produce numbers directly comparable to the AU
leaderboard, we re-run AU's actual `bfcl_metric.py` against our inference data.

Inputs (per BFCL category):
    <inference_root>/<category>/output.jsonl     produced by run_bfcl_fc_inference.py

Outputs (per BFCL category):
    <output_root>/<category>/metrics.json        AU-scored result
    <output_root>/summary.json                    aggregated across categories

Usage:
    python run_bfcl_au_scoring.py \
        --inference_root /path/to/.../eval-results \
        --output_root    /path/to/.../bfcl_test \
        [--au_harness    /path/to/au_harness_for_voice_chat/AU-Harness] \
        [--categories simple parallel parallel_multiple multiple irrelevance]
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import List

DEFAULT_AU_HARNESS = "/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/au_harness_for_voice_chat/AU-Harness"
DEFAULT_CATEGORIES = ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"]


def _install_stubs() -> None:
    """Install minimal stubs for AU-Harness framework deps that bfcl_metric.py
    imports at module top but whose contents the scoring math never touches.

    What the scoring code path actually uses from these imports:
      - metrics.metrics.Metrics: only `super().__init__()` is called from
        `BFCLMatchScore.__init__`. We stub with a no-op base class.
      - models.model_response.ModelResponse: only used as a type annotation
        (`Optional[List[ModelResponse]]`) on `__call__`'s kwargs; never
        accessed when we pass model_responses=None.
      - utils.util.smart_round: called once in `__call__` to round the final
        percentage. We stub with `round(v, p)` which matches AU's behavior
        (AU's `smart_round` is just `round` with a default-precision constant).
      - utils.custom_logging.{write_record_log, append_final_score}: called
        only when `task_name` AND `model_name` are both passed to `__call__`.
        We don't pass them, so these are never invoked.

    The `_compute_outputs`, `_compare_tool_call`, `_compare_dicts`,
    `_standardize_value`, `compute_record_level_scores` paths are pure
    Python with no external deps, so byte-identical to AU.
    """

    class _StubMetrics:
        def __init__(self, **kwargs):
            self.name = None

    metrics_pkg = types.ModuleType("metrics")
    metrics_metrics = types.ModuleType("metrics.metrics")
    metrics_metrics.Metrics = _StubMetrics
    sys.modules["metrics"] = metrics_pkg
    sys.modules["metrics.metrics"] = metrics_metrics

    models_pkg = types.ModuleType("models")
    mr = types.ModuleType("models.model_response")
    mr.ModelResponse = object
    sys.modules["models"] = models_pkg
    sys.modules["models.model_response"] = mr

    utils_pkg = types.ModuleType("utils")
    util_mod = types.ModuleType("utils.util")
    util_mod.smart_round = lambda val, prec=2: round(val, prec)
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.util"] = util_mod

    cl_mod = types.ModuleType("utils.custom_logging")
    cl_mod.write_record_log = lambda *a, **k: None
    cl_mod.append_final_score = lambda *a, **k: None
    sys.modules["utils.custom_logging"] = cl_mod


def _load_au_bfcl_metric(au_harness_path: str):
    """Load AU's bfcl_metric.py module file directly via importlib."""
    path = Path(au_harness_path) / "metrics" / "bfcl_metric.py"
    if not path.exists():
        raise FileNotFoundError(f"AU bfcl_metric.py not found at {path}")
    spec = importlib.util.spec_from_file_location("au_bfcl_metric", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- Parsing: match AU's pipeline output exactly ----------

_TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


def parse_tool_calls(generation: str) -> List[dict]:
    """Replicate the (server-parser → postprocessor) output that feeds AU's
    BFCLMatchScore: a list of `{tool_name: arguments_dict}` dicts.

    - Same `<TOOLCALL>...</TOOLCALL>` regex as
      `nemotron_v2_voicechat_toolcall_parser.py` (server side).
    - Same `[` ... `]` fix-up as that parser.
    - `json.loads` the array, then build `{name: arguments}` (no value
      coercion, matching AU's postprocessor which only does `json.loads`
      on the arguments string).
    """
    if not isinstance(generation, str):
        return []
    m = _TOOLCALL_RE.search(generation)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    try:
        calls = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for tc in calls:
        if not isinstance(tc, dict) or "name" not in tc:
            continue
        args = tc.get("arguments", {})
        # AU's postprocessor: arguments may be a JSON string from the server
        # tool-parser. Decode to dict, but accept dicts as-is.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({tc["name"]: args})
    return out


# ---------- Convert required_fields shape ----------

def _normalize_required_fields(rf):
    """Our JSONL stores required_fields as
        {func_name: [[param, type], ...]}
    AU expects List[Tuple[str, str]]. List-of-lists vs list-of-tuples is
    irrelevant for the `for param, param_type in required_params` unpack,
    but normalize explicitly so behavior is unambiguous.
    """
    out = {}
    for k, items in (rf or {}).items():
        out[k] = [(p, t) for p, t in items]
    return out


# ---------- Main scoring loop ----------

def score_category(scorer_cls, category: str, output_jsonl: Path) -> dict:
    candidates = []
    references = []
    with open(output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tool_resp = parse_tool_calls(entry.get("generation", ""))
            candidates.append({"tool_response": tool_resp})
            expected_call = entry.get("expected_call", [])
            req = _normalize_required_fields(entry.get("required_fields", {}))
            references.append((expected_call, req))

    scorer = scorer_cls()
    # Call WITHOUT task_name/model_name so AU's log writers are NOT invoked
    # (we don't need their side outputs).
    result = scorer(candidates, references)
    return {
        "category": category,
        "num_samples": len(candidates),
        "accuracy": result["final"],
        "num_correct": int(round(result["final"] * len(candidates) / 100.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_root", required=True,
                        help="Path to eval-results dir containing <category>/output.jsonl")
    parser.add_argument("--output_root", required=True,
                        help="Path to write <category>/metrics.json and summary.json")
    parser.add_argument("--au_harness", default=DEFAULT_AU_HARNESS,
                        help="Path to AU-Harness root (contains metrics/bfcl_metric.py)")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help="Which BFCL categories to score")
    args = parser.parse_args()

    _install_stubs()
    bfcl_metric = _load_au_bfcl_metric(args.au_harness)
    print(f"[au-scoring] Loaded AU BFCLMatchScore from {args.au_harness}/metrics/bfcl_metric.py")

    inference_root = Path(args.inference_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for category in args.categories:
        oj = inference_root / category / "output.jsonl"
        if not oj.exists():
            print(f"[au-scoring] SKIP {category}: {oj} not found")
            continue
        result = score_category(bfcl_metric.BFCLMatchScore, category, oj)
        out_dir = output_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = out_dir / "metrics.json"
        metrics_file.write_text(json.dumps({f"bfcl_fc.{category}": result}, indent=2))
        print(f"[au-scoring] {category:20s} n={result['num_samples']:4d}  accuracy={result['accuracy']}%")
        summary[category] = result

    if summary:
        total = sum(r["num_samples"] for r in summary.values())
        weighted = sum(r["accuracy"] * r["num_samples"] for r in summary.values()) / total if total else 0.0
        macro = sum(r["accuracy"] for r in summary.values()) / len(summary)
        summary_obj = {
            "per_category": summary,
            "macro_avg_accuracy": round(macro, 2),
            "micro_avg_accuracy": round(weighted, 2),
            "total_samples": total,
            "scoring": "au_harness.bfcl_metric.BFCLMatchScore (loaded directly via importlib)",
        }
        (output_root / "summary.json").write_text(json.dumps(summary_obj, indent=2))
        print()
        print(f"[au-scoring] macro_avg_accuracy = {summary_obj['macro_avg_accuracy']}%")
        print(f"[au-scoring] micro_avg_accuracy = {summary_obj['micro_avg_accuracy']}%")
        print(f"[au-scoring] summary written to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
