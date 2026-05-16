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

# VoiceBench - A benchmark for evaluating speech language models
# Source: https://huggingface.co/datasets/lmms-lab/voicebench

DATASET_GROUP = "speechlm"
IS_BENCHMARK_GROUP = True

# All VoiceBench subtests that can be run individually
BENCHMARKS = {
    "voicebench.bbh": {},
    "voicebench.alpacaeval": {},
    "voicebench.alpacaeval_full": {},
    "voicebench.alpacaeval_speaker": {},
    "voicebench.ifeval": {},
    "voicebench.openbookqa": {},
    "voicebench.advbench": {},
    "voicebench.commoneval": {},
    "voicebench.wildvoice": {},
    "voicebench.mtbench": {},
    "voicebench.mmsu": {},
    "voicebench.sd_qa": {},
}
