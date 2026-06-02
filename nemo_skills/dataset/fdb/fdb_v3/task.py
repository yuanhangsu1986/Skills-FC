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

"""FDB v3 (FD3) generation task.

Per the dataset-pipelining rule: this module isolates the fdb_v3-specific
conversion (attach `tools` to every generation request) without modifying the
shared generation pipeline. Pattern mirrors nemo_skills/inference/eval/bfcl.py.

prepare.py's output JSONL is NOT modified. The FD3 tool spec is global to the
benchmark, so we load it once at task __init__ from the vendored FDBV3_CHENCHEN
repo (the same source render_prompt.py uses to bake the system prompt) and
attach it to generate_async() on every call.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import asdict, field, is_dataclass
from pathlib import Path
from typing import Any

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_token_count
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


DEFAULT_FDB_REPO = "/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/FDBV3_CHENCHEN"


def _load_fd3_tool_spec(fdb_repo: Path) -> list[dict[str, Any]]:
    benchmark_module_path = fdb_repo / "FD3" / "release_code" / "run_s2s_offline_benchmark.py"
    if not benchmark_module_path.exists():
        raise FileNotFoundError(f"FD3 benchmark module not found: {benchmark_module_path}")
    spec = importlib.util.spec_from_file_location(
        "_fd3_offline_benchmark_for_task", benchmark_module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import FD3 benchmark module: {benchmark_module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_fd3_offline_benchmark_for_task", module)
    spec.loader.exec_module(module)
    return module.FD3_TOOL_SPEC


def _to_openai_tool_format(raw_tools: list[dict]) -> list[dict]:
    out = []
    for tool in raw_tools:
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
            out.append(tool)
        else:
            out.append({"type": "function", "function": tool})
    return out


@nested_dataclass(kw_only=True)
class FDBv3GenerationConfig(GenerateSolutionsConfig):
    """FDB v3 generation config.

    `fdb_repo` points at the vendored FDBV3_CHENCHEN repo from which FD3_TOOL_SPEC
    is loaded. The default matches the path used by prepare.py / render_prompt.py.
    """

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    fdb_repo: str = DEFAULT_FDB_REPO
    # General, optional overrides. Default None -> current fdb_v3 behavior (no
    # regression for fdb_v3 / fdb_v3_chen_chen). The fdb_v3_official variant sets
    # these in its YAML; nothing benchmark-specific is hardcoded here.
    #   tool_spec_path     : load the OpenAI tool spec from this JSON file
    #                        instead of FD3_TOOL_SPEC under fdb_repo.
    #   system_prompt_path : seed cfg.system_message from this file. The base
    #                        openai fill_prompt then applies the standard
    #                        precedence (overrides messages[0] only when
    #                        system_message is truthy).
    tool_spec_path: str | None = None
    system_prompt_path: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_fdb_v3_generation_config", node=FDBv3GenerationConfig)


class FDBv3GenerationTask(GenerationTask):
    def __init__(self, cfg: FDBv3GenerationConfig):
        super().__init__(cfg)
        # Tool spec: from an explicit JSON (official variant) or, by default,
        # FD3_TOOL_SPEC under fdb_repo (current fdb_v3 behavior).
        if self.cfg.tool_spec_path:
            with open(self.cfg.tool_spec_path, "r", encoding="utf-8") as f:
                raw_tools = json.load(f)
            tool_src = self.cfg.tool_spec_path
        else:
            raw_tools = _load_fd3_tool_spec(Path(self.cfg.fdb_repo))
            tool_src = self.cfg.fdb_repo
        self._tools = _to_openai_tool_format(raw_tools)
        # System-prompt precedence (highest -> lowest): (a) cfg.system_message
        # set inline in config; then seeding it from system_prompt_path (also
        # config); (b) the system message baked into test.jsonl by prepare.py;
        # (c) the base GenerateSolutionsConfig.system_message default. (b)/(c)
        # are handled by the base openai fill_prompt, which overrides
        # messages[0] only when system_message is truthy -- so we only seed from
        # the file when no inline system_message was provided.
        if not self.cfg.system_message and self.cfg.system_prompt_path:
            self.cfg.system_message = Path(self.cfg.system_prompt_path).read_text(encoding="utf-8")
        LOG.info(
            "FDBv3GenerationTask: loaded %d tools from %s; system_message=%s",
            len(self._tools),
            tool_src,
            "config/file" if self.cfg.system_message else "test.jsonl-or-default",
        )

    async def process_single_datapoint(self, data_point, all_data):
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        generation_params = {
            **inference_params,
            **self.extra_generate_params,
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            "tools": self._tools,
        }

        if self.cfg.code_execution:
            if (
                self.cfg.override_max_code_executions
                and self.cfg.total_code_executions_in_prompt is not None
            ):
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        result = await self.generate_with_semaphore(**generation_params)

        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(self.hf_tokenizer, generation_params["prompt"])
            result["num_input_tokens"] = num_input_tokens

        return result


GENERATION_TASK_CLASS = FDBv3GenerationTask


@hydra.main(version_base=None, config_name="base_fdb_v3_generation_config")
def fdb_v3_generation(cfg: FDBv3GenerationConfig):
    cfg = FDBv3GenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = FDBv3GenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    FDBv3GenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        fdb_v3_generation()
