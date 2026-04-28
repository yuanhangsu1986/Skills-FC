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
Prepare BFCL single-turn function-channel dataset.

Loads pre-synthesised audio from a HuggingFace BFCL dataset and writes:
  <output_dir>/<category>/input.jsonl   — one sample per line
  <output_dir>/<category>/audio/        — WAV files (one per sample)

Each input.jsonl entry:
  {
    "id":             "simple_0",
    "audio_path":     "/abs/path/to/audio/<id>.wav",
    "system_prompt":  "Here is a list of functions...\n[{...}]",
    "question_text":  "...",          # text of the spoken question (for reference)
    "expected_call":  [{"func": {...}}],
    "required_fields": {"func": [["param", "type"], ...]}
  }

Usage:
    python prepare.py \
        --output_dir /data/bfcl_fc \
        --categories simple parallel \
        [--hf_dataset gorilla-llm/Berkeley-Function-Calling-Leaderboard] \
        [--hf_split test] \
        [--max_samples 100]
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from nemo_skills.dataset.bfcl_single_turn_function_channel.constants import (
    HF_AUDIO_COLUMN,
    HF_DATASET_REPO,
    HF_ID_COLUMN,
    HF_PROMPT_COLUMN,
    HF_SUBSET_MAP,
    HF_TARGET_COLUMN,
    HF_TOOLS_COLUMN,
    SINGLE_TURN_CATEGORIES,
    TOOLS_SYSTEM_PROMPT_PREFIX,
    TOOLS_USER_PROMPT,
)


def _save_wav(array: np.ndarray, sr: int, path: str) -> None:
    import soundfile as sf
    sf.write(path, array, sr)


def _normalise_tool_name(name: str) -> str:
    return re.sub(r"\.", "_", name)


def _build_system_prompt(tools: list) -> str:
    normalised = [{**t, "name": _normalise_tool_name(t["name"])} for t in tools]
    return TOOLS_USER_PROMPT + TOOLS_SYSTEM_PROMPT_PREFIX + json.dumps(normalised, indent=4)


def _build_required_fields(tools: list) -> dict:
    result = {}
    for tool in tools:
        name = _normalise_tool_name(tool["name"])
        params = tool.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})
        result[name] = [(p, props.get(p, {}).get("type", "any")) for p in required]
    return result


def _parse_reference(reference_raw, tools: list) -> list:
    """Normalise reference to [{normalised_func_name: args}, ...]."""
    if isinstance(reference_raw, str):
        reference_raw = json.loads(reference_raw)
    tool_name_map = {t["name"].split(".")[-1]: _normalise_tool_name(t["name"]) for t in tools}
    result = []
    for item in reference_raw:
        func_name = list(item.keys())[0]
        full_name = tool_name_map.get(func_name, _normalise_tool_name(func_name))
        result.append({full_name: item[func_name]})
    return result


def prepare_category(
    dataset,
    category: str,
    output_dir: str,
    audio_column: str,
    tools_column: str,
    target_column: str,
    id_column: str,
    prompt_column: str,
    max_samples: int = None,
) -> int:
    """Write input.jsonl + audio WAVs for one category. Returns sample count."""
    cat_dir = Path(output_dir) / category
    audio_dir = cat_dir / "audio"
    cat_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    jsonl_path = cat_dir / "input.jsonl"
    count = 0

    with open(jsonl_path, "w") as out:
        for i, row in enumerate(dataset):
            if max_samples and count >= max_samples:
                break

            sample_id = row.get(id_column, f"{category}_{i}")

            # Skip multi-turn samples (they contain lists of more than one turn)
            prompt_raw = row.get(prompt_column, "")
            if isinstance(prompt_raw, str):
                try:
                    prompt_raw = ast.literal_eval(prompt_raw)
                except Exception:
                    pass
            if isinstance(prompt_raw, list) and len(prompt_raw) != 1:
                continue
            question_text = prompt_raw[0] if isinstance(prompt_raw, list) else str(prompt_raw)

            # Audio
            audio_data = row.get(audio_column)
            if audio_data is None:
                print(f"[prepare] Warning: no audio for {sample_id}, skipping.")
                continue
            if isinstance(audio_data, list):
                if len(audio_data) != 1:
                    continue
                audio_data = audio_data[0]
            audio_array = np.array(audio_data["array"])
            sr = int(audio_data.get("sampling_rate", 16000))

            wav_path = str((audio_dir / f"{sample_id}.wav").resolve())
            _save_wav(audio_array, sr, wav_path)

            # Tools
            tools_raw = row.get(tools_column, "[]")
            if isinstance(tools_raw, str):
                tools_raw = json.loads(tools_raw)
            tools = tools_raw

            # Reference
            ref_raw = row.get(target_column, "[]")
            expected_call = _parse_reference(ref_raw, tools)
            required_fields = _build_required_fields(tools)
            system_prompt = _build_system_prompt(tools)

            entry = {
                "id": sample_id,
                "audio_path": wav_path,
                "system_prompt": system_prompt,
                "question_text": question_text,
                "expected_call": expected_call,
                "required_fields": required_fields,
            }
            out.write(json.dumps(entry) + "\n")
            count += 1

    print(f"[prepare] {category}: wrote {count} samples → {jsonl_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare BFCL single-turn function-channel dataset")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=SINGLE_TURN_CATEGORIES,
        choices=SINGLE_TURN_CATEGORIES + ["all"],
        help="Categories to prepare (default: all single-turn)",
    )
    parser.add_argument("--hf_dataset", default=HF_DATASET_REPO, help="HuggingFace dataset repo")
    parser.add_argument("--hf_split", default="test", help="Dataset split")
    parser.add_argument("--hf_audio_column", default=HF_AUDIO_COLUMN)
    parser.add_argument("--hf_tools_column", default=HF_TOOLS_COLUMN)
    parser.add_argument("--hf_target_column", default=HF_TARGET_COLUMN)
    parser.add_argument("--hf_id_column", default=HF_ID_COLUMN)
    parser.add_argument("--hf_prompt_column", default=HF_PROMPT_COLUMN)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    categories = SINGLE_TURN_CATEGORIES if "all" in args.categories else args.categories

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. pip install datasets", file=sys.stderr)
        sys.exit(1)

    total = 0
    for cat in categories:
        hf_subset = HF_SUBSET_MAP.get(cat, cat)
        print(f"\n[prepare] Loading {args.hf_dataset} / {hf_subset} (category: {cat}) ...")
        try:
            ds = load_dataset(args.hf_dataset, name=hf_subset, split=args.hf_split, trust_remote_code=True)
        except Exception as e:
            print(f"[prepare] Warning: could not load category '{cat}' (subset '{hf_subset}'): {e}")
            continue

        count = prepare_category(
            dataset=ds,
            category=cat,
            output_dir=args.output_dir,
            audio_column=args.hf_audio_column,
            tools_column=args.hf_tools_column,
            target_column=args.hf_target_column,
            id_column=args.hf_id_column,
            prompt_column=args.hf_prompt_column,
            max_samples=args.max_samples,
        )
        total += count

    print(f"\n[prepare] Done. Total samples: {total}")


if __name__ == "__main__":
    main()
