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

"""Render the FD3 system prompt by combining the Nemotron tool-calling Jinja
template with `FD3_TOOL_SPEC` from the vendored FDBV3_CHENCHEN repo.

Used by `prepare.py` (to embed the prompt into test.jsonl messages) and
optionally as a standalone CLI to render the prompt to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_FDB_REPO = Path("/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/NeMo/FDBV3_CHENCHEN")
DEFAULT_TEMPLATE = Path(
    "/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/function_calling_share/script/template.jinja"
)


def _load_fd3_tool_spec(fdb_repo: Path) -> list[dict[str, Any]]:
    benchmark_module = fdb_repo / "FD3" / "release_code" / "run_s2s_offline_benchmark.py"
    if not benchmark_module.exists():
        raise FileNotFoundError(f"FD3 benchmark module not found: {benchmark_module}")
    spec = importlib.util.spec_from_file_location("_fd3_offline_benchmark", benchmark_module)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import FD3 benchmark module: {benchmark_module}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_fd3_offline_benchmark", module)
    spec.loader.exec_module(module)
    return module.FD3_TOOL_SPEC


def _tojson_plain(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def render_fd3_system_prompt(
    fdb_repo: Path = DEFAULT_FDB_REPO,
    template_path: Path = DEFAULT_TEMPLATE,
    system_message: str = "",
) -> str:
    from jinja2 import Environment

    tools = _load_fd3_tool_spec(fdb_repo)
    template_text = template_path.read_text(encoding="utf-8")
    env = Environment()
    env.filters["tojson"] = _tojson_plain
    return env.from_string(template_text).render(system_message=system_message, tools=tools)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FD3 S2S system prompt to stdout")
    parser.add_argument("--fdb_repo", type=Path, default=DEFAULT_FDB_REPO)
    parser.add_argument("--template_path", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--system_message", type=str, default="")
    parser.add_argument("--output", type=Path, default=None, help="If set, write to file instead of stdout")
    args = parser.parse_args()

    rendered = render_fd3_system_prompt(args.fdb_repo, args.template_path, args.system_message)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote system prompt ({len(rendered)} chars) -> {args.output}")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
