# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Nemotron v2 tool-call parser for the voicechat unified server.

Extracts ``<TOOLCALL>...</TOOLCALL>`` blocks from model output and converts
them to OpenAI-compatible ``ToolCall`` objects, with optional argument-type
coercion driven by the OpenAI tool schemas in the request.

Logic ported from vtrinh's au_harness_for_voice_chat copy of this file
(byte-equivalent except that this version imports the shared
``ToolParser`` / ``ToolCall`` / ``FunctionCall`` / ``ExtractedToolCallInformation``
from ``recipes.multimodal.server.tool_parser`` rather than redefining them
inline — required so ``_load_tool_parser`` in unified_server.py finds the
class via ``issubclass(..., ToolParser)``).

Usage:
    Pass this file to the unified server via ``--tool_call_parser``:

        --tool_call_parser /path/to/nemotron_v2_voicechat_toolcall_parser.py
"""

import ast
import json
import logging
import math
import re
import uuid
from typing import Optional

from recipes.multimodal.server.tool_parser import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
    ToolParser,
)

logger = logging.getLogger(__name__)

SCHEMA_TYPE_TO_PYTHON = {
    "string": str, "integer": int, "number": float, "float": float,
    "boolean": bool, "array": list, "object": dict,
    # BFCL-idiom aliases (not in vtrinh's upstream): "dict" alongside "object",
    # "tuple" alongside "array", "any" as pass-through. Strict superset — the
    # added keys previously fell back to `str`, and `_coerce_value` is a no-op
    # for non-string values, so adding them cannot regress existing behavior.
    "dict": dict, "tuple": list, "any": object,
}


def _coerce_value(value, expected_type):
    """Coerce a value to the schema-declared Python type.

    Handles two cases:
    - String values output by S2S models (e.g. "5" instead of 5).
    - Non-string type mismatches (e.g. int 18 when float expected).

    This mirrors what BFCL's java_type_converter does for Java and
    what any production tool-calling framework would do before dispatching
    an actual API call.
    """
    if not isinstance(value, str):
        # bool must be checked before int since bool is a subclass of int in Python
        if isinstance(value, bool):
            return value
        if expected_type == float and isinstance(value, int):
            return float(value)   # 18 → 18.0
        if expected_type == int and isinstance(value, float):
            return int(value)     # 18.0 → 18
        return value

    if expected_type == str:
        return value

    if expected_type == int:
        try:
            return int(float(value))   # handles "18" and "18.0"
        except (ValueError, TypeError, OverflowError):
            return value
    elif expected_type == float:
        try:
            result = float(value)
            if math.isinf(result) or math.isnan(result):
                return value   # overflow → inf, or "nan" → NaN
            return result
        except (ValueError, TypeError):
            return value
    elif expected_type == bool:
        low = value.lower()
        if low in ("true", "1"):
            return True
        elif low in ("false", "0"):
            return False
        return value
    elif expected_type == list:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
    elif expected_type == dict:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return value


def _build_param_type_map(tools):
    """Build {func_name: {param_name: python_type}} from OpenAI-format tool schemas."""
    mapping = {}
    if not tools:
        return mapping
    for tool in tools:
        func = tool.get("function", tool) if "function" in tool else tool
        name = func.get("name", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        param_types = {}
        for pname, pschema in props.items():
            ptype_str = pschema.get("type", "string")
            param_types[pname] = SCHEMA_TYPE_TO_PYTHON.get(ptype_str, str)
        mapping[name] = param_types
        mapping[re.sub(r"\.", "_", name)] = param_types
    return mapping


def _coerce_arguments(arguments, func_name, param_type_map):
    """Apply type coercion to all arguments of a tool call."""
    if not isinstance(arguments, dict):
        return arguments
    type_info = param_type_map.get(func_name) or param_type_map.get(
        re.sub(r"\.", "_", func_name), {}
    )
    if not type_info:
        return arguments
    coerced = {}
    for k, v in arguments.items():
        expected = type_info.get(k)
        coerced[k] = _coerce_value(v, expected) if expected else v
    return coerced


class NemotronV2VoiceChatToolParser(ToolParser):
    """Parse ``<TOOLCALL>[...]</TOOLCALL>`` blocks from Nemotron v2 output."""

    def __init__(self):
        self.tool_call_start_token: str = "<TOOLCALL>"
        self.tool_call_end_token: str = "</TOOLCALL>"
        self.tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[list] = None,
    ) -> ExtractedToolCallInformation:

        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        else:
            param_type_map = _build_param_type_map(tools)

            try:
                str_tool_calls = self.tool_call_regex.findall(model_output)[0].strip()
                if not str_tool_calls.startswith("["):
                    str_tool_calls = "[" + str_tool_calls
                if not str_tool_calls.endswith("]"):
                    str_tool_calls = str_tool_calls + "]"
                json_tool_calls = json.loads(str_tool_calls)
                tool_calls = []
                for tool_call in json_tool_calls:
                    try:
                        func_name = tool_call["name"]
                        arguments = tool_call["arguments"]
                        if isinstance(arguments, dict) and param_type_map:
                            arguments = _coerce_arguments(arguments, func_name, param_type_map)

                        tool_calls.append(ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=json.dumps(arguments, ensure_ascii=False)
                                    if isinstance(arguments, dict) else arguments,
                            ),
                            id=f"call_{uuid.uuid4().hex[:8]}",
                        ))
                    except Exception:
                        continue

                content = model_output[:model_output.rfind(self.tool_call_start_token)]

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception(
                    "Error in extracting tool call from response. Response: %s",
                    model_output,
                )
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
