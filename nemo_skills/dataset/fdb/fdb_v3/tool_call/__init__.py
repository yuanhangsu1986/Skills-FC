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

# Single-subtest benchmark: per-domain / per-difficulty breakdowns are produced
# by the FD3 evaluators (evaluate_tool_calls.py) and surfaced by run_scoring.py.
METRICS_TYPE = "exact_match"
GENERATION_ARGS = "++prompt_format=openai"
EVAL_ARGS = "++eval_type=null"

# Route through the fdb_v3 task class so FD3_TOOL_SPEC is attached as OpenAI
# `tools` to every request. Required by the chat-template path (server-side
# --chat_template + --no_pre_baked_system_prompt). Defined in
# nemo_skills/dataset/fdb/fdb_v3/task.py.
GENERATION_MODULE = "nemo_skills.dataset.fdb.fdb_v3.task"
