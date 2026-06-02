"""Official FDB-v3 mock-API executor adapter for the DRIRF generalized dispatch.

DRIRF's `_mock_api_dispatch` is implementation-agnostic: it parses <TOOLCALL>,
extracts (name, args), and calls one configurable entry point with the contract

    _execute(name: str, args: dict) -> str   # JSON string of the tool result

This module provides that entry point backed by the OFFICIAL upstream tool
implementations (vendored mock_apis.py). Point the dispatch at it with
    FDB_MOCK_API_IMPL=/nemo_run/code/nemo_skills/dataset/fdb/scripts/fdb_v3_official/mock_api/fd3_mock_execute.py
(the default FDB_MOCK_API_FUNC is `_execute`).

We call the official tool functions directly (no LatencyInjector): the response
CONTENT is the official benchmark's; live API-latency simulation does not apply
to offline in-stream dispatch (our latency is audio-derived).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mock_apis import MockAPIRegistry  # noqa: E402


def _execute(name: str, args: dict) -> str:
    func = MockAPIRegistry.FUNCTIONS.get(str(name))
    if func is None:
        return json.dumps({"status": "error", "message": f"Unknown function: {name}"})
    try:
        result = func(**(args or {}))
    except Exception as e:  # surface arg-shape errors as a tool error, like a real API
        result = {"status": "error", "message": str(e)}
    return json.dumps(result)
