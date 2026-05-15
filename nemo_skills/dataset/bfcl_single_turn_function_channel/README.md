# BFCL Single-Turn Function Channel

Speech language model evaluation on function-calling tasks from the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).
The model receives a spoken question alongside tool definitions and must emit the correct
function call via the model's dedicated function channel.

- **Source**: `ServiceNow-AI/BFCL_v3_audio` (HuggingFace) — TTS-synthesised audio on top of the
  official [BFCL v3 text benchmark](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **Size**: 1,240 samples across 5 categories (split: `test`)
- **Scoring**: exact match — function name + required parameter values (no LLM judge)

## Categories

Five categories are available in `ServiceNow-AI/BFCL_v3_audio`
(HF subset name in parentheses):

| Category | HF subset | Description |
|---|---|---|
| `simple` | `BFCL_v3_simple` | Single call — covers Python, Java, JavaScript variants |
| `parallel` | `BFCL_v3_parallel` | Multiple simultaneous calls |
| `multiple` | `BFCL_v3_multiple` | One call chosen from multiple tools |
| `parallel_multiple` | `BFCL_v3_parallel_multiple` | Parallel calls across multiple tools |
| `irrelevance` | `BFCL_v3_irrelevance` | No matching tool — model should not call anything |

Live categories and per-language simple variants (simple_python / simple_java /
simple_javascript) are not present in the audio dataset.

## Step 1 — Prepare the dataset

Run **once** on the cluster to download audio from HuggingFace and write per-category JSONL:

```bash
python nemo_skills/dataset/bfcl_single_turn_function_channel/prepare.py \
    --output_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/bfcl_fc_data
```

Prepare a subset of categories:

```bash
python nemo_skills/dataset/bfcl_single_turn_function_channel/prepare.py \
    --output_dir /lustre/.../bfcl_fc_data \
    --categories simple_python parallel multiple
```

Limit samples per category (useful for smoke-testing):

```bash
python nemo_skills/dataset/bfcl_single_turn_function_channel/prepare.py \
    --output_dir /lustre/.../bfcl_fc_data \
    --max_samples 50
```

Output layout:

```
<output_dir>/
├── simple_python/
│   ├── input.jsonl
│   └── audio/
│       ├── simple_python_0.wav
│       └── ...
├── parallel/
│   ├── input.jsonl
│   └── audio/
└── ...
```

Each `input.jsonl` line contains `id`, `audio_path`, `system_prompt` (tool definitions),
`question_text`, `expected_call`, and `required_fields`.

Set `data_dir` in the config to the `output_dir` used above.

## Step 2 — Edit the config

Copy and edit either:

- `scripts/bfcl_fc_config_s2s_incremental_v2_greedy.yaml` — temperature=0, deterministic

Mandatory fields to update:

```yaml
model:      /path/to/your/checkpoint
data_dir:   /path/to/bfcl_fc_data    # output_dir from Step 1
output_dir: /path/to/eval/results
```

The `--tool_call_parser`, `--decode_function_channel`, and
`--use_function_channel_for_tool_calls` flags in `server_args` activate the
function channel path. Do not remove them.

## Step 3 — Launch the evaluation

### Standard foreground run

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_s2s_incremental_v2_greedy.yaml
```

### Background run with nohup (recommended for long submissions)

```bash
nohup python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_s2s_incremental_v2_greedy.yaml \
    > bfcl_greedy.log 2>&1 &

echo "PID: $!"
tail -f bfcl_greedy.log
```

`-u` forces unbuffered Python output so lines appear in the log immediately rather
than being held in a buffer until the process exits.

### Uncommitted code

`run_eval.py` calls `isolate_job_dir`, which creates a unique per-submission
directory using a timestamp and UUID — independent of git state. This means
**every submission re-uploads the current working directory**, including any
uncommitted edits. There is no need to commit before launching.

### Stage control

Run inference only (submit Slurm jobs, do not score yet):

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml --inference_only
```

Run scoring only on existing `output.jsonl` (no server needed):

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml --scoring_only
```

Force re-run scoring even if `metrics.json` already exists:

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml --scoring_only --scoring_force
```

Run a single category:

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml --categories simple_python
```

Dry run (validate config and print job commands without submitting):

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml --dry_run
```

### CLI overrides

Any top-level config key can be overridden from the command line:

```bash
python -u ... --config bfcl_fc_config_s2s_incremental_v2_greedy.yaml \
    --model /path/to/other/checkpoint \
    --output_dir /tmp/quick_test
```

## How inference works

Unlike VB/FDB/BBA, BFCL requires a **different system prompt per sample** (the tool
definitions change for every question). The standard nemo-skills batch evaluator uses a
single global system message and cannot support this, so BFCL uses a standalone HTTP
client (`run_bfcl_fc_inference.py`) instead.

Each Slurm inference job:
1. Starts `serve_unified` in the background on `localhost:<port>` with
   `--decode_function_channel --tool_call_parser ... --use_function_channel_for_tool_calls`
2. Runs the inference client, which polls until the server is healthy, then sends one
   request per sample with the per-sample system prompt (tool definitions)
3. Writes `output.jsonl` with `generation` = raw `<TOOLCALL>[...]</TOOLCALL>` text
4. Kills the server and exits

### Server port allocation

`run_eval.py` picks the server port via `get_free_port(strategy="random")` from
`nemo_skills.pipeline.utils.server` — the same mechanism `nemo_eval` uses for every
other benchmark (BBA, FDB, VoiceBench, ...). Each of the 5 category submissions gets
an independent `random.randint(1024, 65535)` port, resolved at submit time.

You can override this with `--server_port <int>` (or `server_port:` in the YAML) for
local debugging, but **do not hardcode a port in production runs**: an earlier version
of this script pinned port 8000, which caused silent failures when SLURM packed
multiple bfcl jobs onto the same node — the second job's `serve_unified` failed to
bind, while its client could still poll `/health` on 8000 and unwittingly send
requests to the first job's server (on the wrong GPU).

**Residual race (small, unmitigated):** `get_free_port(strategy="random")` does
`random.randint(1024, 65535)` *on the submission host* — it does not verify that
the chosen port is free on the compute node where the job eventually runs. So if
two bfcl jobs (or one bfcl job and an unrelated process on the same node) happen
to draw the same random port AND land on the same node, the second `serve_unified`
will fail to bind. This is the same compromise the rest of nemo_skills makes;
collision probability across N jobs is ≈ N²/(2·64k), e.g. ~0.003 for N=20.
If we ever see this hit in practice, the fix is to switch to a runtime port
pick on the compute node (e.g. `python -c "import socket; s=socket.socket();
s.bind(('',0)); print(s.getsockname()[1])"` inside the bash command) so the OS
hands us an ephemeral port that is guaranteed free at bind time.

## Output

Inference writes `output.jsonl` to `<output_dir>/eval-results/<category>/`.

Scoring writes `metrics.json` to the same directory:

```json
{
  "bfcl_fc.simple_python":   {"greedy": {"accuracy": 74.5, "num_samples": 200, "num_correct": 149}},
  "bfcl_fc.parallel":        {"greedy": {"accuracy": 61.0, "num_samples": 200, "num_correct": 122}},
  "bfcl_fc.irrelevance":     {"greedy": {"accuracy": 88.0, "num_samples": 200, "num_correct": 176}}
}
```

Scoring is deterministic exact-match — no LLM judge is involved.
