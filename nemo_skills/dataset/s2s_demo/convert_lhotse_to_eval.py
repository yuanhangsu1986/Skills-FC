#!/usr/bin/env python3
"""
Convert lhotse dataset to evaluation JSONL format.

This script reads a lhotse dataset (cuts.*.jsonl.gz + recording.*.tar) and produces
an evaluation JSONL file similar to voicebench format with:
- Single user turn containing the full recording audio
- System prompt
- No text content (audio only)
"""

import argparse
import gzip
import json
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert lhotse dataset to eval JSONL format")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to lhotse dataset directory containing cuts.*.jsonl.gz and recording.*.tar",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for test.jsonl and extracted audio files",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to use for all samples",
    )
    parser.add_argument(
        "--audio-subdir",
        type=str,
        default="data",
        help="Subdirectory name for audio files in output",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="s2s_demo",
        help="Dataset prefix for audio paths in messages",
    )
    return parser.parse_args()


def load_cuts(input_dir: Path) -> list[dict]:
    """Load all cuts from gzipped JSONL files."""
    cuts = []
    for cuts_file in sorted(input_dir.glob("cuts.*.jsonl.gz")):
        with gzip.open(cuts_file, "rt") as f:
            for line in f:
                if line.strip():
                    cuts.append(json.loads(line))
    return cuts


def extract_audio_files(input_dir: Path, output_audio_dir: Path, recording_ids: set[str]):
    """Extract audio files from tar archives."""
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()

    for tar_file in sorted(input_dir.glob("recording.*.tar")):
        with tarfile.open(tar_file, "r") as tar:
            for member in tar.getmembers():
                # Audio files are stored as {id}.flac
                if member.name.endswith(".flac"):
                    # The recording id is the name without .flac extension
                    # But in tar, files are stored as {recording_id}.flac
                    # where recording_id might already have an extension like .wav or .flac
                    base_name = member.name[:-5]  # Remove .flac
                    if base_name in recording_ids:
                        # Extract to output directory with original name
                        member.name = Path(member.name).name
                        tar.extract(member, output_audio_dir)
                        extracted.add(base_name)
                        print(f"Extracted: {member.name}")

    return extracted


def convert_cut_to_eval_format(cut: dict, audio_subdir: str, dataset_prefix: str, system_prompt: str) -> dict:
    """Convert a single lhotse cut to evaluation format."""
    recording_id = cut["id"]
    audio_filename = f"{recording_id}.flac"
    audio_path = f"{audio_subdir}/{audio_filename}"
    full_audio_path = f"{dataset_prefix}/{audio_path}"

    return {
        "problem": "",
        "audio_path": audio_path,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "", "audio": {"path": full_audio_path}},
        ],
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_audio_dir = output_dir / args.audio_subdir

    print(f"Loading cuts from {input_dir}...")
    cuts = load_cuts(input_dir)
    print(f"Found {len(cuts)} cuts")

    # Get all recording IDs
    recording_ids = {cut["id"] for cut in cuts}

    print(f"Extracting {len(recording_ids)} audio files...")
    extract_audio_files(input_dir, output_audio_dir, recording_ids)

    # Convert to eval format
    print("Converting to eval format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test.jsonl"

    with open(output_file, "w") as f:
        for cut in cuts:
            entry = convert_cut_to_eval_format(cut, args.audio_subdir, args.dataset_prefix, args.system_prompt)
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(cuts)} entries to {output_file}")


if __name__ == "__main__":
    main()
