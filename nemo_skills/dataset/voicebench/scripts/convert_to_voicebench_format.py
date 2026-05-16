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

"""Convert nemo-skills output format to VoiceBench format for official scoring."""

import argparse
import json
import re
from pathlib import Path


def clean_special_tokens(text: str) -> str:
    """Remove special timing/frame tokens from S2S model output.

    The S2S model outputs special tokens like:
    - <$X.XX$> - energy/confidence markers
    - <|X.XX|> - timing/duration markers

    These should be stripped for clean text output used in evaluation.
    """
    if not text:
        return text
    # Remove <$X.XX$> patterns (energy/confidence)
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    # Remove <|X.XX|> patterns (timing)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Mapping from VoiceBench subtests to evaluator types
SUBTEST_TO_EVALUATOR = {
    "sd_qa": "qa",
    "sd_qa_usa": "qa",  # Same evaluator as sd_qa
    "alpacaeval": "open",
    "alpacaeval_full": "open",
    "alpacaeval_speaker": "open",
    "commoneval": "open",
    "wildvoice": "open",
    "mtbench": "open",
    "advbench": "harm",
    "ifeval": "ifeval",
    "openbookqa": "mcq",
    "mmsu": "mcq",
    "bbh": "bbh",
}

# Subtests that require GPT judge (api_judge.py) before evaluation
REQUIRES_GPT_JUDGE = {
    "sd_qa",
    "sd_qa_usa",  # Same as sd_qa
    "alpacaeval",
    "alpacaeval_full",
    "alpacaeval_speaker",
    "commoneval",
    "wildvoice",
    "mtbench",
}


def infer_bbh_task_type(prompt: str) -> str:
    """Infer the BBH task type from the prompt content.

    The BBH evaluator uses the task type to determine how to extract answers.
    Task types: navigate, sports_understanding, hyperbaton, web_of_lies
    """
    prompt_lower = prompt.lower()

    # Navigate task - directional instructions
    if any(
        phrase in prompt_lower
        for phrase in [
            "turn right",
            "turn left",
            "turn around",
            "take steps",
            "take 1 step",
            "take 2 steps",
            "return to the starting point",
        ]
    ):
        return "navigate"

    # Sports understanding - plausibility of sports sentences
    if "is the following sentence plausible" in prompt_lower and any(
        sport in prompt_lower
        for sport in [
            "touchdown",
            "home run",
            "corner kick",
            "penalty kick",
            "free throw",
            "slam dunk",
            "field goal",
            "worked a full count",
            "hit a triple",
            "scored a goal",
            "threw a curveball",
        ]
    ):
        return "sports_understanding"

    # Hyperbaton - adjective order
    if "adjective order" in prompt_lower or "which sentence has the correct adjective order" in prompt_lower:
        return "hyperbaton"

    # Web of lies - truth-telling puzzles
    if "tells the truth" in prompt_lower or "does .* tell the truth" in prompt_lower:
        return "web_of_lies"

    # Default fallback - try to infer from Yes/No pattern
    if "yes or no" in prompt_lower:
        if "is the following sentence plausible" in prompt_lower:
            return "sports_understanding"
        return "navigate"

    return "unknown"


def convert_entry(entry: dict, entry_index: int = 0) -> dict:
    """Convert a single nemo-skills entry to VoiceBench format.

    nemo-skills format:
        - problem: the prompt/question
        - prompt: original prompt (for ifeval, preserved from source)
        - generation: model's response
        - expected_answer: reference answer (if available)
        - Additional fields: id, key, instruction_id_list, kwargs, etc.

    VoiceBench format:
        - prompt: the instruction/question
        - response: model's output
        - reference: expected answer (if available)
        - id: identifier (for bbh)
        - Additional fields preserved as-is
    """
    # Prefer 'prompt' if it exists (e.g., for ifeval), otherwise use 'problem'
    prompt_text = entry.get("prompt") or entry.get("problem", "")
    # Clean special tokens from generation (S2S model outputs timing markers)
    generation = clean_special_tokens(entry.get("generation", ""))
    converted = {
        "prompt": prompt_text,
        "response": generation,
        # Default reference to empty string if not present (e.g., for mtbench)
        "reference": entry.get("expected_answer") or "",
    }

    # Preserve additional fields needed by specific evaluators
    # For BBH evaluator - needs 'id' field
    if "id" in entry:
        converted["id"] = entry["id"]
    else:
        # Infer task type for BBH if no id present
        task_type = infer_bbh_task_type(prompt_text)
        if task_type != "unknown":
            converted["id"] = f"{task_type}_{entry_index}"

    # For IFEval evaluator - needs instruction_id_list and kwargs
    if "instruction_id_list" in entry:
        converted["instruction_id_list"] = entry["instruction_id_list"]
    if "kwargs" in entry:
        converted["kwargs"] = entry["kwargs"]
    if "key" in entry:
        converted["key"] = entry["key"]

    # Preserve subset_for_metrics if present
    if "subset_for_metrics" in entry:
        converted["subset_for_metrics"] = entry["subset_for_metrics"]

    return converted


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a nemo-skills JSONL file to VoiceBench format.

    Returns the number of entries converted.
    """
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                entry = json.loads(line)
                converted = convert_entry(entry, entry_index=idx)
                entries.append(converted)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return len(entries)


def get_evaluator_type(subtest: str) -> str:
    """Get the VoiceBench evaluator type for a given subtest."""
    return SUBTEST_TO_EVALUATOR.get(subtest, "open")


def requires_gpt_judge(subtest: str) -> bool:
    """Check if a subtest requires GPT judge scoring before evaluation."""
    return subtest in REQUIRES_GPT_JUDGE


def main():
    parser = argparse.ArgumentParser(description="Convert nemo-skills output to VoiceBench format")
    parser.add_argument("--input", "-i", required=True, help="Path to input nemo-skills JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to output VoiceBench JSONL file")
    parser.add_argument("--subtest", help="Subtest name (for printing evaluator info)")
    args = parser.parse_args()

    count = convert_file(args.input, args.output)
    print(f"Converted {count} entries from {args.input} to {args.output}")

    if args.subtest:
        evaluator = get_evaluator_type(args.subtest)
        needs_judge = requires_gpt_judge(args.subtest)
        print(f"Subtest: {args.subtest}")
        print(f"Evaluator type: {evaluator}")
        print(f"Requires GPT judge: {needs_judge}")


if __name__ == "__main__":
    main()
