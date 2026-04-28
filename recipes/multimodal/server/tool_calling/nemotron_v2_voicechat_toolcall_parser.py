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
Nemotron v2 tool-call parser.

Extracts <TOOLCALL>...</TOOLCALL> blocks from model output and converts them
to OpenAI-compatible ToolCall objects.

Pass this file to the unified server via --tool_call_parser:

    --tool_call_parser /path/to/nemotron_v2_voicechat_toolcall_parser.py
"""

import json
import logging
import re
import uuid

from recipes.multimodal.server.tool_parser import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
    ToolParser,
)

logger = logging.getLogger(__name__)


class NemotronV2VoiceChatToolParser(ToolParser):
    """Parse <TOOLCALL>[...]</TOOLCALL> blocks from Nemotron v2 function channel output."""

    def __init__(self):
        self.tool_call_start_token = "<TOOLCALL>"
        self.tool_call_end_token = "</TOOLCALL>"
        self.tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def extract_tool_calls(self, model_output: str) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            str_tool_calls = self.tool_call_regex.findall(model_output)[0].strip()
            if not str_tool_calls.startswith("["):
                str_tool_calls = "[" + str_tool_calls
            if not str_tool_calls.endswith("]"):
                str_tool_calls = str_tool_calls + "]"

            json_tool_calls = json.loads(str_tool_calls)
            tool_calls = []
            for tc in json_tool_calls:
                try:
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=tc["name"],
                                arguments=(
                                    json.dumps(tc["arguments"], ensure_ascii=False)
                                    if isinstance(tc["arguments"], dict)
                                    else tc["arguments"]
                                ),
                            ),
                            id=f"call_{uuid.uuid4().hex[:8]}",
                        )
                    )
                except Exception:
                    continue

            content = model_output[: model_output.rfind(self.tool_call_start_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from: %s", model_output[:200])
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
