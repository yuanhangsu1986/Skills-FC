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
Prepare BigBench Audio dataset for NeMo Skills evaluation.

Downloads ArtificialAnalysis/big_bench_audio from HuggingFace, saves audio as WAV,
and creates per-category JSONL files in NeMo Skills format.

Usage:
    python prepare.py --output_dir /lustre/.../data/bba
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]

CATEGORIES = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]


def save_audio(audio_data: dict, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio_data["array"], audio_data["sampling_rate"])


def format_entry(entry: dict, audio_path_relative: str) -> dict:
    """Format a single BBA entry into NeMo Skills JSONL format."""
    audio_info = {"audio": {"path": audio_path_relative}}
    user_message = {"role": "user", "content": "", **audio_info}

    return {
        "id": entry["id"],
        "category": entry["category"],
        "expected_answer": entry["official_answer"],
        "audio_path": audio_path_relative,
        "messages": [user_message],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare BigBench Audio dataset for NeMo Skills")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Root output directory for per-category test.jsonl + data/*.wav. "
            "Must point at a Lustre or other shared dir, NOT inside the repo source tree — "
            "nemo-run stages the source tree to job_dir, so writing data there ships hundreds "
            "of MB of audio with every job submission. Match the `data_dir` field in your eval YAML."
        ),
    )
    args = parser.parse_args()

    try:
        args.output_dir.resolve().relative_to(REPO_ROOT)
        sys.exit(
            f"Refusing to write data into the repo source tree: {args.output_dir}\n"
            f"Pick a path on Lustre (e.g. matching the `data_dir` field of your eval YAML)."
        )
    except ValueError:
        pass  # outside the repo — good.

    output_dir = args.output_dir
    audio_dir = output_dir / "data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ArtificialAnalysis/big_bench_audio from HuggingFace...")
    ds = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")

    # Collect entries per category
    per_category: dict[str, list] = {cat: [] for cat in CATEGORIES}

    for entry in tqdm(ds, desc="Processing samples"):
        category = entry["category"]
        if category not in per_category:
            print(f"Warning: unknown category '{category}', skipping")
            continue

        audio_id = f"bba_{entry['id']}"
        audio_path = audio_dir / f"{audio_id}.wav"
        save_audio(entry["audio"], audio_path)

        audio_path_relative = f"data/{audio_id}.wav"
        formatted = format_entry(entry, audio_path_relative)
        per_category[category].append(formatted)

    # Write per-category JSONL and __init__.py
    for category in CATEGORIES:
        entries = per_category[category]
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = cat_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"Wrote {len(entries)} entries to {jsonl_path}")

    # Per-category __init__.py files are tracked in source under
    # nemo_skills/dataset/bba/<category>/ for benchmark registration; not
    # written here, since this output_dir is intentionally outside the repo.

    print(f"\nDone. Audio saved to {audio_dir}, JSONL files written per category.")
    print("Point data_dir in your eval config to the output_dir.")


if __name__ == "__main__":
    main()
