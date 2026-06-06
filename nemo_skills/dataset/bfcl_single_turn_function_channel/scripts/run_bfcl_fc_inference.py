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

"""
BFCL single-turn function-channel inference client.

Reads prepared input.jsonl, sends each sample to a running serve_unified
server (with --tool_call_parser + --use_function_channel_for_tool_calls +
--decode_function_channel), and writes output.jsonl.

Per-sample system prompts (tool definitions) are sent as the 'system' role
message and tools are passed in OpenAI format so the server activates the
function channel for each request.

The 'generation' field in output.jsonl contains the raw function-channel text
(e.g. '<TOOLCALL>[...]</TOOLCALL>'), which run_bfcl_fc_scoring.py parses.

Usage:
    python run_bfcl_fc_inference.py \
        --server_url http://localhost:8000 \
        --input_jsonl /data/bfcl_fc/simple_python/input.jsonl \
        --output_jsonl /results/simple_python/output.jsonl \
        [--poll_interval 30] [--max_poll_attempts 40] [--max_workers 2]
"""

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _is_openai_schema(tools: list) -> bool:
    """Return True if tools are already in OpenAI JSON Schema format.
    Detects this by checking whether the first tool's parameters.type == 'object'
    (the converted form); raw BFCL data uses 'dict' here.
    """
    if not tools:
        return True
    first = tools[0]
    func = first.get("function", first)
    return func.get("parameters", {}).get("type") == "object"


def _convert_to_openai_tools(tools: list) -> list:
    """Convert raw BFCL tool definitions to OpenAI JSON Schema format.
    Mirrors prepare.py._convert_to_openai_tools() — used as a fallback for
    input.jsonl files prepared before this conversion was added to prepare.py.
    input.jsonl stores tools wrapped as {"type": "function", "function": {...}};
    _prepare_convert expects unwrapped dicts, so we unwrap first.
    """
    from nemo_skills.dataset.bfcl_single_turn_function_channel.prepare import (
        _convert_to_openai_tools as _prepare_convert,
    )
    unwrapped = [t.get("function", t) for t in tools]
    return _prepare_convert(unwrapped)


def _tool_calls_to_generation(tool_calls: list) -> str:
    """Convert OpenAI-format tool_calls to <TOOLCALL>...</TOOLCALL> generation string."""
    calls = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        calls.append({"name": name, "arguments": args})
    return f"<TOOLCALL>{json.dumps(calls)}</TOOLCALL>"


def _poll_server(server_url: str, interval: int, max_attempts: int) -> bool:
    import urllib.request
    health_url = server_url.rstrip("/") + "/health"
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "healthy":
                    print(f"[inference] Server ready (attempt {attempt})")
                    return True
        except Exception:
            pass
        print(f"[inference] Waiting for server... attempt {attempt}/{max_attempts}")
        time.sleep(interval)
    return False


def _send_request(
    server_url: str,
    audio_bytes: bytes,
    system_prompt: str,
    tools: list,
    timeout: int = 300,
    max_tokens: int = 512,
) -> dict:
    import urllib.request
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ],
            },
        ],
        "tools": tools,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _process_sample(server_url: str, sample: dict, timeout: int, max_tokens: int) -> tuple[dict, bool]:
    """Process a single sample; returns (out_entry, failed)."""
    audio_path = sample["audio_path"]
    system_prompt = sample["system_prompt"]
    tools = sample.get("tools", [])
    if not _is_openai_schema(tools):
        tools = _convert_to_openai_tools(tools)

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        print(f"[inference] WARNING: could not read audio {audio_path}: {e}")
        generation = ""
        failed = True
    else:
        try:
            response = _send_request(server_url, audio_bytes, system_prompt, tools, timeout, max_tokens)
            message = response["choices"][0]["message"]
            raw_tool_calls = message.get("tool_calls") or []
            finish_reason = response["choices"][0].get("finish_reason")
            print(f"[inference] {sample['id']} | finish_reason: {finish_reason} | "
                  f"content: {message.get('content', '')!r} | tool_calls: {raw_tool_calls}")
            if raw_tool_calls:
                generation = _tool_calls_to_generation(raw_tool_calls)
            else:
                generation = message.get("content", "")
            failed = False
        except Exception as e:
            print(f"[inference] WARNING: request failed for {sample['id']}: {e}")
            generation = ""
            failed = True

    out_entry = {
        "id": sample["id"],
        "generation": generation,
        "expected_call": sample["expected_call"],
        "required_fields": sample["required_fields"],
        "question_text": sample.get("question_text", ""),
    }
    return out_entry, failed


def main():
    parser = argparse.ArgumentParser(description="BFCL function-channel inference client")
    parser.add_argument("--server_url", required=True, help="Base URL of the serve_unified server")
    parser.add_argument("--input_jsonl", required=True, help="Prepared input.jsonl from prepare.py")
    parser.add_argument("--output_jsonl", required=True, help="Where to write output.jsonl")
    parser.add_argument("--poll_interval", type=int, default=30, help="Seconds between server polls")
    parser.add_argument("--max_poll_attempts", type=int, default=40, help="Max server poll attempts")
    parser.add_argument("--request_timeout", type=int, default=300, help="Per-request HTTP timeout (s)")
    parser.add_argument("--max_workers", type=int, default=2, help="Concurrent requests (should match server batch_size)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per request; caps generation to prevent runaway inference on GPU(s)")
    parser.add_argument("--num_chunks", type=int, default=1, help="Split input into N chunks for parallel SLURM jobs")
    parser.add_argument("--chunk_id", type=int, default=0, help="Which chunk to process (0-indexed)")
    args = parser.parse_args()

    if not _poll_server(args.server_url, args.poll_interval, args.max_poll_attempts):
        print("[inference] ERROR: Server did not become ready.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]

    # Chunk selection: evenly split across num_chunks, assign this job's slice
    if args.num_chunks > 1:
        chunk_size = (len(all_samples) + args.num_chunks - 1) // args.num_chunks
        start = args.chunk_id * chunk_size
        samples = all_samples[start: start + chunk_size]
        # Write to output_chunk_{chunk_id}.jsonl instead of output.jsonl
        output_path = output_path.parent / f"output_chunk_{args.chunk_id}.jsonl"
        print(f"[inference] Chunk {args.chunk_id}/{args.num_chunks}: samples {start}-{start+len(samples)-1} ({len(samples)} total)")
    else:
        samples = all_samples
    print(f"[inference] Processing {len(samples)} samples from {input_path}")

    # Submit all requests upfront; the executor caps concurrency at max_workers.
    # Iterating futures in submission order gives ordered, streaming writes while
    # keeping the server's batch continuously filled.
    n_failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(_process_sample, args.server_url, sample, args.request_timeout, args.max_tokens)
            for sample in samples
        ]
        with open(output_path, "w") as out:
            for i, future in enumerate(futures):
                out_entry, failed = future.result()
                if failed:
                    n_failed += 1
                out.write(json.dumps(out_entry) + "\n")
                out.flush()
                if (i + 1) % 10 == 0:
                    print(f"[inference] {i + 1}/{len(samples)} done")

    print(f"[inference] Wrote {len(samples)} entries to {output_path} ({n_failed} failures)")
    # Write done marker for chunk jobs so merge/scoring can detect completion
    if args.num_chunks > 1:
        done_path = output_path.with_suffix(".jsonl.done") if output_path.suffix != ".done" else output_path.parent / (output_path.name + ".done")
        done_path = output_path.parent / (output_path.name + ".done")
        done_path.write_text("ok\n")
        print(f"[inference] Wrote done marker: {done_path}")
    if n_failed > 0:
        failure_rate = n_failed / len(samples)
        if failure_rate > 0.1:
            print(f"[inference] ERROR: {n_failed}/{len(samples)} samples failed ({failure_rate:.0%}), aborting.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
