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

# Categories available in ServiceNow-AI/BFCL_v3_audio.
# "simple" covers all language variants (python / java / javascript) in one subset.
# Live categories and multi-turn are not present in the audio dataset.
SINGLE_TURN_CATEGORIES = [
    "simple",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
]

# Map from our category name to the HuggingFace subset name used by
# ServiceNow-AI/BFCL_v3_audio (passed as `name=` in load_dataset).
HF_SUBSET_MAP = {
    "simple":            "BFCL_v3_simple",
    "parallel":          "BFCL_v3_parallel",
    "multiple":          "BFCL_v3_multiple",
    "parallel_multiple": "BFCL_v3_parallel_multiple",
    "irrelevance":       "BFCL_v3_irrelevance",
}

# HuggingFace dataset with pre-synthesised audio for BFCL questions.
# ServiceNow-AI/BFCL_v3_audio contains TTS-synthesised audio alongside the
# original BFCL text fields (question, tools, reference).
HF_DATASET_REPO = "ServiceNow-AI/BFCL_v3_audio"
HF_AUDIO_COLUMN = "audio"
HF_TOOLS_COLUMN = "tools"
HF_TARGET_COLUMN = "reference"
HF_ID_COLUMN = "id"
HF_PROMPT_COLUMN = "question"

TOOLS_USER_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out. "
    "If the given question lacks the parameters required by the function, also point it out.\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of \n\n"
    "```json\n[{\"func_name1\":{\"params_name1\":params_value1, \"params_name2\":params_value2...}}, "
    "{\"func_name2\":{\"params_name1\":params_value1, \"params_name2\":params_value2, \"params_name3\":params_value3...}}]\n``` \n\n"
    "You SHOULD NOT include any other text in the response. "
    "If no relevant function matches then return empty list in json like ```json\n [] \n ```. "
    "Make sure an appropriate json is always there in the response. \n\n"
    "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
    "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
    "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
)
TOOLS_SYSTEM_PROMPT_PREFIX = "Here is a list of functions in JSON format that you can invoke.\n"
