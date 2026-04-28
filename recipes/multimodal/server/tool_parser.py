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
Tool-call parser base class and supporting data types.

Concrete parsers (e.g. NemotronV2VoiceChatToolParser) subclass ToolParser
and are loaded dynamically at server startup via --tool_call_parser <path>.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON-encoded string


@dataclass
class ToolCall:
    type: str  # always "function"
    function: FunctionCall
    id: str = ""  # unique call id, e.g. "call_abc123"


@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: List[ToolCall] = field(default_factory=list)
    content: Optional[str] = None  # text before the first tool-call block


class ToolParser(ABC):
    """Abstract base for tool-call parsers.

    Subclass and implement extract_tool_calls().  No tokenizer or streaming
    required — regex/string parsing only.
    """

    @abstractmethod
    def extract_tool_calls(self, model_output: str) -> ExtractedToolCallInformation:
        ...
