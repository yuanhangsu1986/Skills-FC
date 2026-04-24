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
Download a HuggingFace model to HF_HOME cache if not already present.

Usage:
    python nemo_skills/inference/download_model.py --model openai/whisper-large-v3
"""

import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model to HF_HOME cache")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g. openai/whisper-large-v3)")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"HF_HOME: {hf_home}", flush=True)
    print(f"Checking/downloading {args.model}...", flush=True)

    path = snapshot_download(repo_id=args.model)
    print(f"Model ready at: {path}", flush=True)


if __name__ == "__main__":
    main()
