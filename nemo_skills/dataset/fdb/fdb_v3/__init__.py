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

# Full-Duplex-Bench v3 (FD3): direct function-head tool-call evaluation for S2S models.
# Source: /lustre/fsw/portfolios/llmservice/users/cchen1/code/Backend_agent/FD3
# Vendored mirror: /lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/FDBV3_CHENCHEN
# (github.com/yuanhangsu1986/FDBV3, branch chen_chen; NeMo lives there as a submodule
#  tracking github.com/yuanhangsu1986/NeMo_fc.git@fdb_v3_chen_chen)

DATASET_GROUP = "speechlm"
IS_BENCHMARK_GROUP = True

BENCHMARKS = {
    "fdb_v3.tool_call": {},
}
