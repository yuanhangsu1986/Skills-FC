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

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]

# Subtest configurations for Full-Duplex-Bench
# Based on the four main evaluation dimensions
# Pause split into candor and synthetic for separate TOR/latency reporting
SUBTESTS = {
    "pause_candor": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's ability to handle pauses (candor / natural data)",
    },
    "pause_synthetic": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's ability to handle pauses (synthetic data)",
    },
    "backchannel": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's backchanneling behavior (e.g., 'uh-huh', 'yeah')",
    },
    "turn_taking": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's turn-taking capabilities",
    },
    "interruption": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's handling of interruptions",
    },
    # v1.5-only subtasks (overlap-focused)
    "background_speech": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's response with background speech",
    },
    "talking_to_other": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's response when user talks to others",
    },
}

# Template for subtest __init__.py files
INIT_TEMPLATE = """# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

METRICS_TYPE = "{metrics_type}"
GENERATION_ARGS = "++prompt_format=openai"
{eval_args}
"""


def save_audio(audio_data, audio_path, sampling_rate=16000):
    """Save audio data to a WAV file."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio_data, sampling_rate)


def format_entry(entry, subtest_name, config, audio_dir, entry_idx, no_audio=False, dataset_name="fdb_v1"):
    """Format a single entry for nemo-skills with OpenAI messages format.

    Creates three message variants in a single entry:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)
    """
    prompt_text = entry.get("prompt", entry.get("question", ""))

    formatted = {
        "problem": prompt_text,
    }

    # Add expected answer if available
    if config.get("has_reference") and "reference" in entry:
        formatted["expected_answer"] = entry["reference"]
    elif config.get("has_reference") and "answer" in entry:
        formatted["expected_answer"] = entry["answer"]

    # Preserve additional metadata fields
    for field in ["id", "dataset", "sample_id", "category", "duration", "overlap_duration"]:
        if field in entry:
            formatted[field] = entry[field]

    # System message (shared across all variants)
    system_message = {"role": "system", "content": "You are a helpful assistant."}

    # Audio filename: {subtest_name}_{sample_id} so we get pause_candor_18.wav, pause_synthetic_18.wav
    # (not just "pause_18" which would collide when both candor and synthetic write to the same audio_dir)
    sample_id = entry.get("sample_id", str(entry_idx))
    audio_id = f"{subtest_name}_{sample_id}"

    # Handle audio - copy/link files and get audio info
    audio_info = None
    if not no_audio:
        # Check if audio_path is provided (file already exists)
        if "audio_path" in entry:
            import shutil
            from pathlib import Path

            source_audio = Path(entry["audio_path"])
            if source_audio.exists():
                audio_dest = audio_dir / f"{audio_id}.wav"

                # Copy audio file to our data directory
                shutil.copy(source_audio, audio_dest)

                audio_info = {"audio": {"path": f"{dataset_name}/data/{audio_id}.wav"}}
                formatted["audio_path"] = f"data/{audio_id}.wav"

        # Handle direct audio data (for compatibility)
        elif "audio" in entry and entry["audio"] is not None:
            audio_path = audio_dir / f"{audio_id}.wav"

            # Handle different audio data formats
            if isinstance(entry["audio"], dict) and "array" in entry["audio"]:
                save_audio(entry["audio"]["array"], audio_path, entry["audio"].get("sampling_rate", 16000))
            else:
                # If audio is already a numpy array or similar
                save_audio(entry["audio"], audio_path)

            audio_info = {"audio": {"path": f"{dataset_name}/data/{audio_id}.wav"}}
            formatted["audio_path"] = f"data/{audio_id}.wav"

    # Create three message variants:

    # 1. messages: audio only (empty content, with audio)
    user_message_audio = {"role": "user", "content": ""}
    if audio_info:
        user_message_audio.update(audio_info)
    formatted["messages"] = [system_message.copy(), user_message_audio]

    # 2. messages_text_audio: both text and audio
    user_message_text_audio = {"role": "user", "content": prompt_text}
    if audio_info:
        user_message_text_audio.update(audio_info)
    formatted["messages_text_audio"] = [system_message.copy(), user_message_text_audio]

    # 3. messages_text: text only (no audio)
    user_message_text = {"role": "user", "content": prompt_text}
    formatted["messages_text"] = [system_message.copy(), user_message_text]

    return formatted


def create_subtest_init(subtest_dir, config):
    """Create __init__.py for a subtest directory."""
    eval_args_line = f'EVAL_ARGS = "{config["eval_args"]}"' if config["eval_args"] else ""
    content = INIT_TEMPLATE.format(
        metrics_type=config["metrics_type"],
        eval_args=eval_args_line,
    )
    with open(subtest_dir / "__init__.py", "w") as f:
        f.write(content)


def process_subtest(subtest_name, config, data_dir, audio_dir, fdb_data_path, no_audio=False, version="v1.0", dataset_name="fdb_v1"):
    """Process a single subtest and save to JSONL.

    Each entry contains three message variants:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)

    Full-Duplex-Bench structure (v1.0 or v1.5):
    - candor_pause_handling/{ID}/input.wav, pause.json, transcription.json
    - candor_turn_taking/{ID}/input.wav, turn_taking.json, transcription.json
    - icc_backchannel/{ID}/input.wav, transcription.json
    - synthetic_pause_handling/{ID}/input.wav, pause.json, transcription.json
    - synthetic_user_interruption/{ID}/input.wav, context.wav, interrupt.wav, interrupt.json
    """
    subtest_dir = data_dir / subtest_name
    subtest_dir.mkdir(parents=True, exist_ok=True)

    output_file = subtest_dir / "test.jsonl"
    entries = []
    entry_idx = 0

    print(f"Processing {subtest_name}...")
    print(f"  Description: {config['description']}")

    # Map subtests to Full-Duplex-Bench dataset folders (v1.0 vs v1.5 have different folders)
    subtest_mapping_v1_0 = {
        "pause_candor": ["candor_pause_handling"],
        "pause_synthetic": ["synthetic_pause_handling"],
        "backchannel": ["icc_backchannel"],
        "turn_taking": ["candor_turn_taking"],
        "interruption": ["synthetic_user_interruption"],
    }
    # v1.5 Drive folder names: background_speech, talking_to_other, user_interruption, user_backchannel
    subtest_mapping_v1_5 = {
        "background_speech": ["background_speech"],
        "talking_to_other": ["talking_to_other"],
        "backchannel": ["user_backchannel"],
        "interruption": ["user_interruption"],
    }
    subtest_mapping = subtest_mapping_v1_5 if version == "v1.5" else subtest_mapping_v1_0
    dataset_folders = subtest_mapping.get(subtest_name, [])
    if not dataset_folders:
        print(f"  Warning: Unknown subtest {subtest_name}")
        return 0

    # Version folder names (v1.0 / v1_0 and v1.5 / v1_5)
    version_folders = ("v1.0", "v1_0") if version == "v1.0" else ("v1.5", "v1_5")
    # Load test cases from each dataset folder
    test_cases = []
    for folder_name in dataset_folders:
        # Try different possible paths
        possible_paths = [
            fdb_data_path / version_folders[0] / folder_name,
            fdb_data_path / version_folders[1] / folder_name,
            fdb_data_path / folder_name,  # Direct
        ]

        folder_path = None
        for path in possible_paths:
            if path.exists():
                folder_path = path
                break

        if folder_path is None:
            print("  Warning: Dataset folder not found. Tried:")
            for path in possible_paths:
                print(f"    - {path}")
            continue

        # Find all sample directories (numeric IDs)
        sample_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])
        print(f"  Found {len(sample_dirs)} samples in {folder_name}")

        # FDB candor_turn_taking and synthetic_user_interruption use 1-based sample IDs (no 0)
        skip_id_zero = folder_name in ("candor_turn_taking", "synthetic_user_interruption")

        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            if skip_id_zero and sample_id == "0":
                continue
            input_wav = sample_dir / "input.wav"

            if not input_wav.exists():
                print(f"  Warning: input.wav not found in {sample_dir}")
                continue

            # Load transcription if available
            transcription_file = sample_dir / "transcription.json"
            transcription = ""
            if transcription_file.exists():
                with open(transcription_file, "r") as f:
                    trans_data = json.load(f)
                    # Extract text from transcription
                    if isinstance(trans_data, dict):
                        transcription = trans_data.get("text", "")
                    elif isinstance(trans_data, list) and len(trans_data) > 0:
                        # Word-level transcription (FDB uses "text" per segment, some datasets use "word")
                        transcription = " ".join([w.get("text", w.get("word", "")) for w in trans_data])

            # Treat whitespace-only transcription as empty (use fallback so problem/content are meaningful)
            prompt_text = (transcription or "").strip() or "Respond to the user's speech in the audio."
            # Create test case
            test_case = {
                "id": f"{folder_name}_{sample_id}",
                "audio_path": str(input_wav),
                "prompt": prompt_text,
                "dataset": folder_name,
                "sample_id": sample_id,
            }

            test_cases.append(test_case)

    if not test_cases:
        print("  No test cases found. Make sure you downloaded the dataset from:")
        print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
        return 0

    # Process each test case
    for entry in tqdm(test_cases, desc="  Processing entries"):
        formatted = format_entry(
            entry,
            subtest_name,
            config,
            audio_dir,
            entry_idx,
            no_audio=no_audio,
            dataset_name=dataset_name,
        )
        entries.append(formatted)
        entry_idx += 1

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Create __init__.py
    create_subtest_init(subtest_dir, config)

    print(f"  Wrote {len(entries)} entries to {output_file}")
    return len(entries)


# All five zip names per version on Google Drive (must all be downloaded and extracted for full dataset)
EXPECTED_V1_ZIPS = [
    "candor_pause_handling.zip",
    "candor_turn_taking.zip",
    "icc_backchannel.zip",
    "synthetic_pause_handling.zip",
    "synthetic_user_interruption.zip",
]

# v1.5 Drive folder names (after download/extract)
EXPECTED_V1_5_ZIPS = [
    "background_speech",
    "talking_to_other",
    "user_interruption",
    "user_backchannel",
]

# Google Drive: root has v1.0 and v1.5 subfolders
# https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3
FDB_ROOT_FOLDER_ID = "1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3"
FDB_V1_0_FOLDER_ID = "1hxzRk7xgtdr5ZEoctnp0sFK0COv91W3h"  # v1.0 subfolder (direct)


def download_dataset(download_dir: Path, version: str = "v1.0") -> bool:
    """Download the Full-Duplex-Bench v1.0 or v1.5 from Google Drive, then unzip.

    Root folder (v1.0 + v1.5): https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3
    - v1.0: we download the v1.0 subfolder directly into download_dir/v1.0.
    - v1.5: we download the ROOT folder so that download_dir/v1.5 is populated (enter v1.5 in Drive);
      then we extract only the zips under download_dir/v1.5.

    Returns:
        True if download successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("\nError: 'gdown' package not found. Install it with:")
        print("  pip install gdown")
        return False

    import shutil
    import zipfile

    if version == "v1.0":
        folder_id = FDB_V1_0_FOLDER_ID
        version_dir_name = "v1.0"
        output_for_download = download_dir / version_dir_name
        version_dir = output_for_download
        zip_description = "candor_pause_handling, candor_turn_taking, icc_backchannel, synthetic_pause_handling, synthetic_user_interruption"
    else:
        # v1.5: download ROOT folder so we get both v1.0 and v1.5; we only use v1.5
        folder_id = FDB_ROOT_FOLDER_ID
        version_dir_name = "v1.5"
        output_for_download = download_dir  # root download -> download_dir/v1.0 and download_dir/v1.5
        version_dir = download_dir / "v1.5"
        zip_description = "background_speech, talking_to_other, user_interruption, user_backchannel"

    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    print("\n" + "=" * 60)
    print(f"Downloading Full-Duplex-Bench {version} from Google Drive")
    print("=" * 60)
    print(f"Source: {folder_url}")
    if version == "v1.5":
        print("(Downloading root folder; v1.5 zips will be taken from download_dir/v1.5)")
    print(f"Destination: {version_dir}")
    print(f"Zips expected: {zip_description}")
    print("=" * 60 + "\n")

    try:
        output_for_download.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {'root (v1.0 + v1.5)' if version == 'v1.5' else version} dataset...")
        gdown.download_folder(
            url=folder_url,
            output=str(output_for_download),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )

        # For v1.5 we downloaded root -> find v1.5 or v1_5 under download_dir (maybe one level down)
        if version == "v1.5":
            if not version_dir.exists():
                version_dir = download_dir / "v1_5"
            if not version_dir.exists():
                for sub in download_dir.iterdir():
                    if sub.is_dir():
                        for name in ("v1.5", "v1_5"):
                            candidate = sub / name
                            if candidate.exists():
                                version_dir = candidate
                                break
                        if version_dir.exists():
                            break
            if not version_dir.exists():
                print(f"After download, v1.5 folder not found under {download_dir}. Check Drive structure.")
                return False

        # gdown may create a single subfolder with the folder name; collect zips from version_dir and one level down
        zip_files = list(version_dir.glob("*.zip")) or list(version_dir.rglob("*.zip"))
        if not zip_files and any(version_dir.iterdir()):
            sub = next((d for d in version_dir.iterdir() if d.is_dir()), None)
            if sub:
                sub_zips = list(sub.rglob("*.zip"))
                if sub_zips:
                    for z in sub_zips:
                        dest = version_dir / z.name
                        if not dest.exists() or dest.stat().st_size != z.stat().st_size:
                            shutil.move(str(z), str(dest))
                    zip_files = list(version_dir.glob("*.zip"))

        print("\n" + "=" * 60)
        print("Download completed.")
        print("=" * 60 + "\n")

        if not zip_files:
            zip_files = list(version_dir.glob("*.zip")) or list(version_dir.rglob("*.zip"))
        if not zip_files:
            print(f"Warning: No .zip files found under {version_dir_name}. Check Drive permissions or download manually.")
            return True

        # Extract all zip files
        print(f"Extracting {len(zip_files)} ZIP file(s) in {version_dir_name}...")
        for zip_file in tqdm(sorted(zip_files), desc="Extracting"):
            try:
                extract_dir = zip_file.parent
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                zip_file.unlink()
                print(f"  Extracted: {zip_file.name}")
            except Exception as e:
                print(f"  Warning: Failed to extract {zip_file.name}: {e}")

        print(f"\nExtracted {len(zip_files)} ZIP file(s).")

        # Verify expected folders exist (names without .zip for v1.0; explicit list for v1.5)
        if version == "v1.5":
            expected_folders = EXPECTED_V1_5_ZIPS
        else:
            expected_folders = [p.replace(".zip", "") for p in EXPECTED_V1_ZIPS]
        found = [d.name for d in version_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        missing = [f for f in expected_folders if f not in found]
        if missing:
            print(f"Warning: After extraction, missing folder(s): {missing}")
            print("  Re-run prepare (without --fdb_data_path to re-download) or add the missing zip(s) manually to:")
            print(f"  {version_dir}")
        else:
            print(f"All expected {version} dataset folders are present.")

        return True

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nYou can manually download from:")
        print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
        print(f"  Open the {version_dir_name} subfolder, download all 5 zips, and extract them into:")
        print(f"  {version_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Full-Duplex-Bench dataset for nemo-skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download v1.0 to default path (s2s/Full-Duplex-Bench-data) and prepare
  python prepare.py

  # Prepare v1.5 (use existing fdb_data_path that contains v1.5 or v1_5 folder)
  python prepare.py --version v1.5 --fdb_data_path /path/to/Full-Duplex-Bench-data

  # Use existing dataset (no download)
  python prepare.py --fdb_data_path /path/to/Full-Duplex-Bench-data

  # Download to a specific location and prepare
  python prepare.py --fdb_data_path /path/to/download/location
        """,
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1.0", "v1.5"],
        default="v1.0",
        help="Dataset version. v1.0 writes to fdb/fdb_v1/; v1.5 writes to fdb/fdb_v1_5/ (run separately).",
    )
    parser.add_argument(
        "--fdb_data_path",
        type=str,
        default=None,
        help="Path to Full-Duplex-Bench dataset root. If not set, downloads selected version to s2s/Full-Duplex-Bench-data and prepares.",
    )
    parser.add_argument(
        "--subtests",
        nargs="+",
        default=None,
        help="Specific subtests to process (default: all)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip processing audio files (faster, for testing)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "Output root for fdb_v1/ or fdb_v1_5/ (per-subtest test.jsonl + data/*.wav). "
            "Must point at a Lustre or other shared dir, NOT inside the repo source tree — "
            "nemo-run stages the source tree to job_dir, so writing data there ships hundreds "
            "of MB of audio with every job submission. Match the `data_dir` field in your eval YAML."
        ),
    )
    args = parser.parse_args()

    try:
        args.data_dir.resolve().relative_to(REPO_ROOT)
        sys.exit(
            f"Refusing to write data into the repo source tree: {args.data_dir}\n"
            f"Pick a path on Lustre (e.g. matching the `data_dir` field of your eval YAML)."
        )
    except ValueError:
        pass  # outside the repo — good.

    version = args.version
    base_dir = args.data_dir  # fdb package dir (v1 and v1_5 are subgroups under it)
    if version == "v1.0":
        data_dir = base_dir / "fdb_v1"
        dataset_name = "fdb_v1"
    else:
        data_dir = base_dir / "fdb_v1_5"
        dataset_name = "fdb_v1_5"
    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = data_dir / "data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _default_fdb_data = REPO_ROOT.parent / "Full-Duplex-Bench-data"

    if args.fdb_data_path:
        fdb_data_path = Path(args.fdb_data_path)
        if not fdb_data_path.exists():
            print(f"\nError: Full-Duplex-Bench data path not found: {fdb_data_path}")
            print("Run without --fdb_data_path to download, or use:")
            print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
            return
        # If requested version data not present, download and extract first
        if version == "v1.5":
            v1_5_dir = fdb_data_path / "v1.5"
            v1_5_alt = fdb_data_path / "v1_5"
            has_data = any(
                (v1_5_dir / folder).exists() or (v1_5_alt / folder).exists()
                for folder in EXPECTED_V1_5_ZIPS
            )
            if not has_data:
                print(f"v1.5 data not found under {fdb_data_path}. Downloading and extracting v1.5...")
                if not download_dataset(fdb_data_path, version="v1.5"):
                    print("\nDownload failed. Exiting.")
                    return
        elif version == "v1.0":
            v1_0_dir = fdb_data_path / "v1.0"
            v1_0_alt = fdb_data_path / "v1_0"
            expected_v1_0 = [p.replace(".zip", "") for p in EXPECTED_V1_ZIPS]
            has_data = any(
                (v1_0_dir / folder).exists() or (v1_0_alt / folder).exists()
                for folder in expected_v1_0
            )
            if not has_data:
                print(f"v1.0 data not found under {fdb_data_path}. Downloading and extracting v1.0...")
                if not download_dataset(fdb_data_path, version="v1.0"):
                    print("\nDownload failed. Exiting.")
                    return
    else:
        download_path = _default_fdb_data
        print(f"Downloading {version} to {download_path} (overwriting if present)...")
        if not download_dataset(download_path, version=version):
            print("\nDownload failed. Exiting.")
            return
        fdb_data_path = download_path

    if args.subtests:
        subtests_to_process = args.subtests
    elif version == "v1.5":
        subtests_to_process = ["background_speech", "talking_to_other", "backchannel", "interruption"]
    else:
        subtests_to_process = ["pause_candor", "pause_synthetic", "backchannel", "turn_taking", "interruption"]

    print(f"Processing {len(subtests_to_process)} subtests for {version} (dataset_name={dataset_name})...")
    if args.no_audio:
        print("Skipping audio download (--no-audio)")

    total_entries = 0
    for subtest_name in subtests_to_process:
        if subtest_name not in SUBTESTS:
            print(f"Warning: Unknown subtest '{subtest_name}', skipping")
            continue

        config = SUBTESTS[subtest_name]
        count = process_subtest(
            subtest_name, config, data_dir, audio_dir, fdb_data_path,
            no_audio=args.no_audio, version=version, dataset_name=dataset_name,
        )
        total_entries += count

    print(f"\nDone! Processed {total_entries} total entries across {len(subtests_to_process)} subtests.")


if __name__ == "__main__":
    main()
