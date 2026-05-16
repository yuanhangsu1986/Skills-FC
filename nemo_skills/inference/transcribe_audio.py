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
General-purpose audio transcription client using a vLLM ASR server.

Reads input_jsonl, calls /v1/audio/transcriptions for each entry's audio.path,
and writes output_jsonl with:
  - generation: ASR transcript of the model's audio output
  - generation_text: original model text generation (preserved)
  - question_asr: ASR transcript of input question audio (if --data_dir provided)

This script is meant to run as a client job alongside a vLLM server started by
nemo_skills/pipeline/transcribe.py, which handles server lifecycle and job scheduling.

Usage (standalone, assuming server is already running):
    python nemo_skills/inference/transcribe_audio.py \
        --server_url http://localhost:8000/v1 \
        --asr_model openai/whisper-large-v3 \
        --input_jsonl /path/to/output.jsonl \
        --output_jsonl /path/to/output_asr.jsonl \
        [--data_dir /path/to/audio/data]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files via a vLLM ASR server")
    parser.add_argument("--server_url", required=True, help="Base URL of the vLLM server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--asr_model", required=True, help="ASR model name as served by vLLM")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file (e.g. output.jsonl from generation)")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file to write transcripts to")
    parser.add_argument("--data_dir", default=None, help="Root directory for input question audio files (for question_asr field)")
    parser.add_argument("--force", action="store_true", help="Overwrite output_jsonl if it already exists")
    args = parser.parse_args()

    out_path = Path(args.output_jsonl)
    if out_path.exists() and not args.force:
        print(f"Output already exists at {out_path}. Skipping (use --force to re-run).")
        return

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(base_url=args.server_url, api_key="placeholder")

    def transcribe(audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=args.asr_model,
                file=f,
            )
        return result.text.strip()

    rows = [json.loads(line) for line in in_path.open() if line.strip()]
    print(f"Transcribing {len(rows)} entries from {in_path}...", flush=True)

    out_rows = []
    for i, row in enumerate(rows):
        out_row = dict(row)
        generation_text = row.get("generation", "") or ""
        out_row["generation_text"] = generation_text

        audio_path = None
        if isinstance(row.get("audio"), dict):
            audio_path = row["audio"].get("path")

        if audio_path and Path(audio_path).exists():
            print(f"  [{i+1}/{len(rows)}] output audio: {Path(audio_path).name}", flush=True)
            asr_text = transcribe(audio_path)
            out_row["generation"] = asr_text
            print(f"           transcript: {asr_text!r}", flush=True)
        else:
            print(f"  [{i+1}/{len(rows)}] no output audio found, keeping text generation", flush=True)
            out_row["generation"] = generation_text

        if args.data_dir:
            q_audio_rel = row.get("audio_path", "")
            if q_audio_rel:
                q_audio_full = Path(args.data_dir) / q_audio_rel
                if q_audio_full.exists():
                    out_row["question_asr"] = transcribe(str(q_audio_full))
                else:
                    print(f"  [{i+1}/{len(rows)}] WARNING: question audio not found: {q_audio_full}", file=sys.stderr)

        out_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(out_rows)} entries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
