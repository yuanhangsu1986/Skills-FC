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

# Only single-turn categories — multi-turn requires stateful execution not
# supported by the function-channel S2S inference path.
SINGLE_TURN_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]

# HuggingFace dataset that provides pre-synthesised audio for BFCL questions.
# The dataset must have columns: id, <audio_column>, tools, <target_column>.
# Confirm the exact repo name with the dataset owner before running prepare.py.
HF_DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_AUDIO_COLUMN = "audio"
HF_TOOLS_COLUMN = "tools"
HF_TARGET_COLUMN = "answer"
HF_ID_COLUMN = "id"
HF_PROMPT_COLUMN = "question"

TOOLS_SYSTEM_PROMPT_PREFIX = (
    "Here is a list of functions in JSON format that you can invoke.\n"
)
