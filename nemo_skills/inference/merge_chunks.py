#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shlex
import subprocess
import sys


def get_chunked_rs_filename(
    output_dir: str,
    random_seed: int = None,
    chunk_id: int = None,
) -> str:
    """Return a path of the form: {output_dir}/output[-rsSEED][-chunkK].jsonl"""
    if random_seed is not None:
        base_filename = f"output-rs{random_seed}.jsonl"
    else:
        base_filename = "output.jsonl"
    if chunk_id is not None:
        basename, ext = os.path.splitext(base_filename)
        base_filename = f"{basename}_chunk_{chunk_id}{ext}"
    return os.path.join(output_dir, base_filename)


def get_merge_cmd(output_dir: str, num_chunks: int, random_seed: int = None) -> str:
    """Build the shell command to merge chunked generation outputs into a single file."""
    merged_output_file = get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed)
    chunk_files = [
        get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed, chunk_id=i)
        for i in range(num_chunks)
    ]
    return f"python -m nemo_skills.inference.merge_chunks {merged_output_file} {' '.join(chunk_files)}"


def unescape_shell_command(command: str) -> str:
    """Unescape special shell characters so they are correctly interpreted before execution."""
    command = command.strip() if command else ""
    return shlex.split(command)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} output_file input_file1 [input_file2 ...] [-- command_to_run]")
        sys.exit(1)

    # Separate file arguments from optional post-merge command
    if "--" in sys.argv:
        sep_index = sys.argv.index("--")
        output_file = sys.argv[1]
        input_files = sys.argv[2:sep_index]
        post_merge_command = unescape_shell_command(" ".join(sys.argv[sep_index + 1 :]))
    else:
        output_file = sys.argv[1]
        input_files = sys.argv[2:]
        post_merge_command = None

    # Idempotency: if already merged, skip silently.
    if os.path.isfile(output_file) and os.path.isfile(f"{output_file}.done"):
        print(f"Info: {output_file} already merged (.done exists). Skipping.")
        sys.exit(0)

    # Validate that all chunk files and their .done markers exist before merging.
    # Exit with code 1 on any missing file so downstream SLURM jobs fail too.
    for file_ in input_files:
        done_file = f"{file_}.done"
        if not os.path.isfile(done_file):
            print(f"Error: {done_file} not found. Chunk generation incomplete.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(file_):
            print(f"Error: {file_} not found. Chunk generation incomplete.", file=sys.stderr)
            sys.exit(1)

    try:
        with open(output_file, "w") as out_f:
            subprocess.run(["cat"] + input_files, stdout=out_f, check=True)
        print(f"Successfully concatenated {len(input_files)} files to {output_file}")

        open(f"{output_file}.done", "w").close()
        for file_ in input_files:
            os.remove(file_)

        if post_merge_command:
            print(f"Executing post-merge command: {' '.join(post_merge_command)}")
            subprocess.run(" ".join(post_merge_command), shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
