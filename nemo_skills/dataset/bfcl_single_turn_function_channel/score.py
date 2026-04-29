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
BFCL function-call match scoring.

Ported from AU-Harness bfcl_metric.py.  Accepts structured tool-call dicts
(as produced by the Nemotron tool-call parser) and checks:
  - function name matches
  - no unexpected parameters
  - required parameters present with correct values

Usage:
    from nemo_skills.dataset.bfcl_single_turn_function_channel.score import bfcl_match_score

    accuracy = bfcl_match_score(candidates, references)
    # candidates: list of {"tool_response": [{func_name: args}, ...]}
    # references: list of (expected_call_list, required_fields_dict)
"""

import re
from typing import Dict, List, Optional, Tuple, Union

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}


def _standardize(val):
    if isinstance(val, str):
        return re.sub(r"[ ,./\-_*^]", "", val).lower().replace("'", '"')
    if isinstance(val, list):
        return [_standardize(v) for v in val]
    if isinstance(val, dict):
        return {k: _standardize(v) for k, v in val.items()}
    return val


def _compare_dicts(tool_dict: dict, ref_dict: dict) -> Tuple[bool, str]:
    for k in tool_dict:
        if k not in ref_dict:
            return False, f"Unexpected key: {k}"
    for k, allowed in ref_dict.items():
        if k not in tool_dict and allowed not in ("", None):
            return False, f"Missing key: {k}"
    for k, v in tool_dict.items():
        tv = _standardize(v)
        rv = _standardize(ref_dict[k])
        if isinstance(rv, list):
            if tv not in [_standardize(x) for x in rv]:
                return False, f"Value mismatch at '{k}': {tv} not in {rv}"
        else:
            if tv != rv:
                return False, f"Value mismatch at '{k}': {tv} != {rv}"
    return True, ""


def _compare_tool_call(
    tool_call: dict,
    ref_call: dict,
    required_fields: Dict[str, List[Tuple[str, str]]],
) -> Tuple[bool, List[str]]:
    if not isinstance(tool_call, dict) or not isinstance(ref_call, dict):
        return False, ["Not a dict"]

    tool_name = list(tool_call.keys())[0]
    ref_name = list(ref_call.keys())[0]

    if re.sub(r"\.", "_", tool_name) != re.sub(r"\.", "_", ref_name):
        return False, [f"Name mismatch: {tool_name} vs {ref_name}"]

    tool_params = tool_call[tool_name]
    ref_params = ref_call[ref_name]

    for k in tool_params:
        if k not in ref_params:
            return False, [f"Unexpected parameter: {k}"]

    req = required_fields.get(tool_name)
    if req is None:
        req = required_fields.get(ref_name)
    if req is None:
        return False, [f"No required-field metadata for '{tool_name}'"]

    errors = []
    for param, param_type in req:
        python_type = PYTHON_TYPE_MAPPING.get(param_type, str)
        if param not in tool_params or param not in ref_params:
            return False, [f"Missing required parameter '{param}'"]

        tv = _standardize(tool_params[param])
        rv = _standardize(ref_params[param])

        if rv in ("", None):
            continue

        # Coerce string values to the expected numeric/bool type so that
        # model outputs like "10" match reference integers like 10.
        if isinstance(tv, str) and python_type in (int, float, bool):
            try:
                tv = tv.lower() in ("true", "1", "yes") if python_type == bool else python_type(tv)
            except (ValueError, TypeError):
                pass

        if python_type == float and isinstance(tv, int):
            tv = float(tv)

        if python_type == dict:
            if isinstance(rv, dict):
                ok, msg = _compare_dicts(tv, rv)
                if not ok:
                    errors.append(msg)
            elif isinstance(rv, list) and all(isinstance(x, dict) for x in rv):
                if not any(_compare_dicts(tv, tmpl)[0] for tmpl in rv):
                    errors.append(f"Dict for '{param}' matched no template")
            else:
                if tv != rv:
                    errors.append(f"Mismatch at '{param}'")
        elif python_type == list:
            if isinstance(rv, list):
                if all(isinstance(x, list) for x in rv):
                    matched = any(
                        len(tv) == len(cand) and all(
                            (_compare_dicts(a, b)[0] if isinstance(b, dict) and isinstance(a, dict) else a == b)
                            for a, b in zip(tv, cand)
                        )
                        for cand in rv
                    )
                    if not matched:
                        errors.append(f"List for '{param}' matched no option")
                else:
                    if tv not in rv:
                        errors.append(f"'{param}' not in allowed list")
            else:
                if tv != rv:
                    errors.append(f"Mismatch at '{param}'")
        else:
            if isinstance(rv, list):
                if tv not in rv:
                    errors.append(f"Mismatch at '{param}': {tv} not in {rv}")
            else:
                if tv != rv:
                    errors.append(f"Mismatch at '{param}': {tv} != {rv}")

    return len(errors) == 0, errors


def _score_one(
    candidate: dict,
    reference_call: List[dict],
    required_fields: Dict[str, List[Tuple[str, str]]],
) -> bool:
    tool_response = candidate.get("tool_response")
    if tool_response is None:
        return False
    if len(tool_response) != len(reference_call):
        return False
    if len(tool_response) == 0:
        return True

    unmatched_cands = list(tool_response)
    for ref in reference_call:
        matched = False
        for j, cand in enumerate(unmatched_cands):
            ok, _ = _compare_tool_call(cand, ref, required_fields)
            if ok:
                unmatched_cands.pop(j)
                matched = True
                break
        if not matched:
            return False
    return True


def bfcl_match_score(
    candidates: List[dict],
    references: List[Tuple[List[dict], Dict[str, List[Tuple[str, str]]]]],
) -> float:
    """Return accuracy in [0, 100]."""
    if not candidates:
        return 0.0
    correct = sum(
        _score_one(c, ref_call, req_fields)
        for c, (ref_call, req_fields) in zip(candidates, references)
    )
    return round(correct * 100.0 / len(candidates), 2)
