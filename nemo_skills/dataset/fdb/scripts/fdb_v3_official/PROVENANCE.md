# Vendored: Full-Duplex-Bench v3 (official) — inference assets + scoring

`fdb_v3_official` makes the eval faithful to the public benchmark
(github.com/DanielLin94144/Full-Duplex-Bench, `v3/`, commit
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, vendored 2026-06-02) on every
surface the model and scorer touch. It reuses our S2S inference *mechanism* and
the `fdb_v3` prepared data (same audio), but injects the official prompt/tools at
generation time and uses the official scoring + mock tool responses.

## Vendored material

| Path | Upstream source | Notes |
|---|---|---|
| `assets/fd3_tool_spec_official.json` | `v3/lk_agent_tool.py` `@function_tool` defs | OpenAI-format tool spec built from the 12 tools: decorator `description`, signature params, docstring `Args:` for per-param text, defaults → `required` |
| `assets/system_prompt_official.txt` | `v3/lk_agent_tool.py` `VoiceAgent(instructions=...)` | verbatim |
| `mock_api/mock_apis.py` | `v3/mock_apis.py` | 12 tool impls verbatim; only the `latency_injector` import made optional |
| `mock_api/fd3_mock_execute.py` | (ours) | `_execute(name,args)->json_str` adapter over the official `MockAPIRegistry.FUNCTIONS` |
| `release_code/{evaluate_*.py,benchmark_data_v2.json}` | `v3/` | scoring — see `release_code/PROVENANCE.md` (judge routed to NV gateway, gpt-4o) |

## Faithfulness boundaries (can't be matched, by construction)

- **Tool serialization / chat template.** The official harness runs each provider
  through LiveKit, which serializes tools into that provider's native format —
  there is no canonical wire format. Our S2S model ingests tools only via its
  nemotron chat template (`nano_v2_voicechat_toolcalling_chat_template.jinja`).
  We match the tool *set, descriptions, param schemas, and system instructions*;
  the template wrapping is necessarily ours.
- **API-latency injection.** Upstream's `latency_injector` simulates live API
  delays during the conversation. Our dispatch injects the tool RESPONSE into the
  S2S stream and derives latency from audio, so latency injection is not
  replicated. The tool-response *content* is the official benchmark's.

## How specifics stay out of shared code

Shared code (`fdb_v3/task.py`, DRIRF `_mock_api_dispatch`, `run_eval.py`) was made
general + config-driven with defaults preserving current behavior (no regression
for `fdb_v3` / `fdb_v3_chen_chen`). All official-specific values live in
`fdb_v3_official_*.yaml`: `tool_spec_path`, `system_prompt_path`, `mock_api_impl`,
`release_code_dir`, `benchmark_key`, and the data root.
