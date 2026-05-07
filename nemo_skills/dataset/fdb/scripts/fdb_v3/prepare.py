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

"""Convert FD3 `fdb_v3_data_released/{example_id}_{speaker_id}/` flat folders
into the nemo-skills `test.jsonl` format under `dataset/fdb/fdb_v3/tool_call/`.

Each row carries the FD3 metadata fields needed by `run_scoring.py` to
reconstruct FD3-compatible `result_<provider>.json` files after generation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from render_prompt import DEFAULT_FDB_REPO, DEFAULT_TEMPLATE, render_fd3_system_prompt  # noqa: E402

FDB_PKG_DIR = THIS_DIR.parents[1]  # nemo_skills/dataset/fdb
DEFAULT_DATASET_DIR = FDB_PKG_DIR / "fdb_v3" / "tool_call"
FD3_DATA_SUBPATH = Path("FD3") / "fdb_v3_data_released"

_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def _discover_samples(data_root: Path) -> list[tuple[str, str, Path, Path]]:
    samples: list[tuple[str, str, Path, Path]] = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        match = _FOLDER_RE.match(folder.name)
        if not match:
            continue
        example_id = match.group(1)
        speaker_id = match.group(2)
        input_wav = folder / "input.wav"
        metadata = folder / "metadata.json"
        if not input_wav.exists() or not metadata.exists():
            continue
        samples.append((example_id, speaker_id, input_wav, metadata))
    return samples


def _format_entry(
    example_id: str,
    speaker_id: str,
    metadata: dict[str, Any],
    audio_rel_path: str,
    system_prompt: str,
    dataset_name: str,
) -> dict[str, Any]:
    sample_id = f"{example_id}_{speaker_id}"

    audio_info = {"audio": {"path": f"{dataset_name}/{audio_rel_path}"}}
    user_message = {"role": "user", "content": "", **audio_info}
    system_message = {"role": "system", "content": system_prompt}

    return {
        "id": sample_id,
        "example_id": example_id,
        "speaker_id": speaker_id,
        "domain": metadata.get("domain", "unknown"),
        "difficulty": metadata.get("difficulty", "unknown"),
        "title": metadata.get("title", example_id),
        "expected_tool_calls": metadata.get("expected_tool_calls", []),
        "num_expected_calls": metadata.get("num_expected_calls", 0),
        "latency_profile": metadata.get("latency_profile", "normal"),
        "state_rollback_test": metadata.get("state_rollback_test", False),
        "audio_path": audio_rel_path,
        "expected_answer": json.dumps(metadata.get("expected_tool_calls", []), ensure_ascii=False),
        "messages": [system_message, user_message],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FD3 (FDB v3) for nemo-skills")
    parser.add_argument(
        "--fdb_repo",
        type=Path,
        default=DEFAULT_FDB_REPO,
        help="Path to FDBV3_CHENCHEN (vendored FD3 repo)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Output root for fdb_v3/ (defaults to the package dir alongside fdb_v1/, fdb_v1_5/). "
            "Use this to write to a Lustre data dir on the cluster."
        ),
    )
    parser.add_argument(
        "--template_path",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Nemotron Jinja chat template used to render the FD3 system prompt",
    )
    parser.add_argument("--no-audio", action="store_true", help="Skip copying input.wav files")
    args = parser.parse_args()

    data_root = args.fdb_repo / FD3_DATA_SUBPATH
    if not data_root.exists():
        sys.exit(f"FD3 data not found at {data_root} — check --fdb_repo")

    base_dir = args.data_dir if args.data_dir else FDB_PKG_DIR
    out_dir = base_dir / "fdb_v3" / "tool_call"
    audio_dir = out_dir.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "fdb_v3"

    print(f"Rendering FD3 system prompt ({args.template_path.name})...")
    system_prompt = render_fd3_system_prompt(args.fdb_repo, args.template_path)
    print(f"  System prompt: {len(system_prompt)} chars")
    (out_dir.parent / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    samples = _discover_samples(data_root)
    print(f"Discovered {len(samples)} samples in {data_root}")

    test_jsonl = out_dir / "test.jsonl"
    written = 0
    with test_jsonl.open("w", encoding="utf-8") as fout:
        for example_id, speaker_id, input_wav, metadata_path in samples:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  warning: skipping {example_id}_{speaker_id}: bad metadata.json ({e})")
                continue

            audio_rel = f"data/{example_id}_{speaker_id}.wav"
            if not args.no_audio:
                dest_wav = audio_dir / f"{example_id}_{speaker_id}.wav"
                if not dest_wav.exists() or dest_wav.stat().st_size != input_wav.stat().st_size:
                    shutil.copy2(input_wav, dest_wav)

            entry = _format_entry(
                example_id, speaker_id, metadata, audio_rel, system_prompt, dataset_name
            )
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} entries -> {test_jsonl}")
    print(f"Audio dir: {audio_dir}")
    print(f"System prompt: {out_dir.parent / 'system_prompt.txt'}")


if __name__ == "__main__":
    main()
