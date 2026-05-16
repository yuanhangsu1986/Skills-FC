# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Full-Duplex-Bench: one group with v1 and v1.5 as subgroups (same prepare.py and eval files).
# Source: https://github.com/DanielLin94144/Full-Duplex-Bench

DATASET_GROUP = "speechlm"
IS_BENCHMARK_GROUP = True

BENCHMARKS = {
    # v1.0 (pause split for separate TOR/latency)
    "fdb_v1.pause_candor": {},
    "fdb_v1.pause_synthetic": {},
    "fdb_v1.backchannel": {},
    "fdb_v1.turn_taking": {},
    "fdb_v1.interruption": {},
    # v1.5
    "fdb_v1_5.background_speech": {},
    "fdb_v1_5.talking_to_other": {},
    "fdb_v1_5.backchannel": {},
    "fdb_v1_5.interruption": {},
}
