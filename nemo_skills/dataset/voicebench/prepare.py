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
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]

# Subtest configurations with their split types and evaluation configs
SUBTESTS = {
    # Simple subtests with "test" split
    "bbh": {
        "splits": ["test"],
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "extra_fields": ["id"],  # Required for BBH evaluator to determine task type
    },
    "alpacaeval": {
        "splits": ["test"],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
    },
    "alpacaeval_full": {
        "splits": ["test"],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
    },
    "ifeval": {
        "splits": ["test"],
        "has_reference": False,
        # Use math metric (permissive) - VoiceBench official scoring handles actual ifeval
        "metrics_type": "math",
        "eval_args": "",
        "extra_fields": ["key", "instruction_id_list", "kwargs"],
    },
    "openbookqa": {
        "splits": ["test"],
        "has_reference": True,
        "metrics_type": "multichoice",
        "eval_args": "++eval_type=multichoice",
    },
    "advbench": {
        "splits": ["test"],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
    },
    "commoneval": {
        "splits": ["test"],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
    },
    "wildvoice": {
        "splits": ["test"],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
    },
    # Multi-turn subtest
    "mtbench": {
        "splits": ["test"],
        "has_reference": True,
        "metrics_type": "llm_judge",
        "eval_args": "",
        "multi_turn": True,
        "extra_fields": ["question_id", "category", "turns"],
    },
    # Multi-split subtests (combine all splits into one)
    "mmsu": {
        "splits": [
            "law",
            "engineering",
            "other",
            "biology",
            "business",
            "economics",
            "health",
            "philosophy",
            "psychology",
            "history",
            "chemistry",
            "physics",
        ],
        "has_reference": True,
        "metrics_type": "multichoice",
        "eval_args": "++eval_type=multichoice",
        "extra_fields": ["question_id", "cot_content", "category", "src"],
    },
    "sd_qa": {
        "hf_name": "sd-qa",  # HF uses hyphen
        "splits": ["aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf"],
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
    },
    "alpacaeval_speaker": {
        "splits": [
            "en_AU_Wavenet_A_1.0_0.0_0.0",
            "en_AU_Wavenet_B_1.0_0.0_0.0",
            "en_IN_Wavenet_A_1.0_0.0_0.0",
            "en_IN_Wavenet_B_1.0_0.0_0.0",
            "en_GB_Wavenet_A_1.0_0.0_0.0",
            "en_GB_Wavenet_B_1.0_0.0_0.0",
            "en_US_Wavenet_A_1.0_0.0_0.0",
            "en_US_Wavenet_C_1.0_0.0_0.0",
            "en_US_Wavenet_A_1.5_0.0_0.0",
            "en_US_Wavenet_A_2.0_0.0_0.0",
            "en_US_Wavenet_A_0.5_0.0_0.0",
        ],
        "has_reference": False,
        "metrics_type": "llm_judge",
        "eval_args": "",
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


def save_audio(audio_data, audio_path):
    """Save audio data to a WAV file."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio_data["array"], audio_data["sampling_rate"])


def format_entry(entry, subtest_name, config, audio_dir, entry_idx, split_name=None, no_audio=False):
    """Format a single entry for nemo-skills with OpenAI messages format.

    Creates three message variants in a single entry:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)
    """
    # Get prompt text - MTBench uses 'turns' list instead of 'prompt'
    if config.get("multi_turn") and "turns" in entry:
        prompt_text = entry["turns"][0]  # First turn is the prompt
    else:
        prompt_text = entry["prompt"]

    formatted = {
        "problem": prompt_text,
    }

    # IFEval requires "prompt" field for Google's evaluator
    if subtest_name == "ifeval":
        formatted["prompt"] = prompt_text

    # Add expected answer if available
    if config.get("has_reference") and "reference" in entry:
        formatted["expected_answer"] = entry["reference"]

    # Add extra fields if specified
    for field in config.get("extra_fields", []):
        if field in entry:
            formatted[field] = entry[field]

    # Add subset_for_metrics for multi-split datasets
    if split_name and len(config["splits"]) > 1:
        formatted["subset_for_metrics"] = split_name

    # System message - use MCQ-specific prompt for multiple-choice subtests
    MCQ_SUBTESTS = {"mmsu", "openbookqa", "bbh"}
    if subtest_name in MCQ_SUBTESTS:
        system_content = "Answer the following multiple choice question with an explanation for the answer."
    else:
        system_content = "You are a helpful assistant."
    system_message = {"role": "system", "content": system_content}

    # Text content (already extracted as prompt_text above)
    content = prompt_text

    # Preserve turns for multi-turn datasets
    if config.get("multi_turn") and "turns" in entry:
        formatted["turns"] = entry["turns"]

    # Handle audio - save files and get audio info
    audio_info = None
    if not no_audio:
        if config.get("multi_turn"):
            # MTBench has audio1 and audio2 for two turns
            audios = []
            for i, audio_key in enumerate(["audio1", "audio2"], 1):
                if audio_key in entry and entry[audio_key] is not None:
                    audio_id = f"{subtest_name}_{entry_idx}_turn{i}"
                    audio_path = audio_dir / f"{audio_id}.wav"
                    save_audio(entry[audio_key], audio_path)
                    audios.append({"path": f"voicebench/data/{audio_id}.wav"})
                    formatted[f"audio_path_{i}"] = f"data/{audio_id}.wav"
            if audios:
                audio_info = {"audios": audios}
        else:
            # Single audio
            if "audio" in entry and entry["audio"] is not None:
                audio_id = f"{subtest_name}_{entry_idx}"
                if split_name:
                    audio_id = f"{subtest_name}_{split_name}_{entry_idx}"
                audio_path = audio_dir / f"{audio_id}.wav"
                save_audio(entry["audio"], audio_path)
                audio_info = {"audio": {"path": f"voicebench/data/{audio_id}.wav"}}
                formatted["audio_path"] = f"data/{audio_id}.wav"

    # Create three message variants:

    # 1. messages: audio only (empty content, with audio)
    user_message_audio = {"role": "user", "content": ""}
    if audio_info:
        user_message_audio.update(audio_info)
    formatted["messages"] = [system_message.copy(), user_message_audio]

    # 2. messages_text_audio: both text and audio
    user_message_text_audio = {"role": "user", "content": content}
    if audio_info:
        user_message_text_audio.update(audio_info)
    formatted["messages_text_audio"] = [system_message.copy(), user_message_text_audio]

    # 3. messages_text: text only (no audio)
    user_message_text = {"role": "user", "content": content}
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


def process_subtest(subtest_name, config, data_dir, audio_dir, no_audio=False):
    """Process a single subtest and save to JSONL.

    Each entry contains three message variants:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)
    """
    hf_name = config.get("hf_name", subtest_name)
    subtest_dir = data_dir / subtest_name
    subtest_dir.mkdir(parents=True, exist_ok=True)

    output_file = subtest_dir / "test.jsonl"
    entries = []
    entry_idx = 0

    print(f"Processing {subtest_name}...")

    for split in tqdm(config["splits"], desc="  Loading splits"):
        try:
            dataset = load_dataset("lmms-lab/voicebench", hf_name, split=split, trust_remote_code=True)
            for entry in dataset:
                formatted = format_entry(
                    entry,
                    subtest_name,
                    config,
                    audio_dir,
                    entry_idx,
                    split_name=split if len(config["splits"]) > 1 else None,
                    no_audio=no_audio,
                )
                entries.append(formatted)
                entry_idx += 1
        except Exception as e:
            print(f"  Warning: Failed to load {subtest_name}/{split}: {e}")

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Create __init__.py
    create_subtest_init(subtest_dir, config)

    print(f"  Wrote {len(entries)} entries to {output_file}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Prepare VoiceBench dataset for nemo-skills")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "Output root for the VoiceBench dataset (per-subtest test.jsonl + data/*.wav). "
            "Must point at a Lustre or other shared dir, NOT inside the repo source tree — "
            "nemo-run stages the source tree to job_dir, so writing data there ships hundreds "
            "of MB of audio with every job submission. Match the `data_dir` field in your eval YAML."
        ),
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
        help="Skip downloading and processing audio files",
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

    data_dir = args.data_dir
    audio_dir = data_dir / "data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    subtests_to_process = args.subtests if args.subtests else list(SUBTESTS.keys())

    print(f"Processing {len(subtests_to_process)} subtests...")
    if args.no_audio:
        print("Skipping audio download (--no-audio)")

    total_entries = 0
    for subtest_name in subtests_to_process:
        if subtest_name not in SUBTESTS:
            print(f"Warning: Unknown subtest '{subtest_name}', skipping")
            continue

        config = SUBTESTS[subtest_name]
        count = process_subtest(subtest_name, config, data_dir, audio_dir, no_audio=args.no_audio)
        total_entries += count

    print(f"\nDone! Processed {total_entries} total entries across {len(subtests_to_process)} subtests.")


if __name__ == "__main__":
    main()
