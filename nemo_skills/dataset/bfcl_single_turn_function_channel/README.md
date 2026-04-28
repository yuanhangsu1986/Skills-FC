# BFCL Single-Turn Function Channel

Speech language model evaluation on function-calling tasks from the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).
The model receives a spoken question alongside tool definitions and must emit the correct
function call via the model's dedicated function channel.

- **Source**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (HuggingFace, pre-synthesised audio)
- **Size**: ~2,600 samples across 13 single-turn categories
- **Scoring**: exact match — function name + required parameter values (no LLM judge)

## Categories

| Category | Description |
|---|---|
| `simple_python` | Single call, Python types |
| `simple_java` | Single call, Java types |
| `simple_javascript` | Single call, JavaScript types |
| `parallel` | Multiple simultaneous calls |
| `multiple` | One call chosen from multiple tools |
| `parallel_multiple` | Parallel calls across multiple tools |
| `irrelevance` | No matching tool — model should not call anything |
| `live_simple` | Real-world single call |
| `live_multiple` | Real-world multi-tool |
| `live_parallel` | Real-world parallel |
| `live_parallel_multiple` | Real-world parallel + multi-tool |
| `live_irrelevance` | Real-world irrelevance |
| `live_relevance` | Real-world relevance detection |

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

- `scripts/bfcl_fc_config_greedy.yaml` — temperature=0, deterministic
- `scripts/bfcl_fc_config_sampling.yaml` — temperature=0.8, stochastic

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
    --config nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_greedy.yaml
```

### Background run with nohup (recommended for long submissions)

```bash
nohup python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_greedy.yaml \
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
    --config bfcl_fc_config_greedy.yaml --inference_only
```

Run scoring only on existing `output.jsonl` (no server needed):

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_greedy.yaml --scoring_only
```

Force re-run scoring even if `metrics.json` already exists:

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_greedy.yaml --scoring_only --scoring_force
```

Run a single category:

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_greedy.yaml --categories simple_python
```

Dry run (validate config and print job commands without submitting):

```bash
python -u nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_eval.py \
    --config bfcl_fc_config_greedy.yaml --dry_run
```

### CLI overrides

Any top-level config key can be overridden from the command line:

```bash
python -u ... --config bfcl_fc_config_greedy.yaml \
    --model /path/to/other/checkpoint \
    --output_dir /tmp/quick_test
```

## How inference works

Unlike VB/FDB/BBA, BFCL requires a **different system prompt per sample** (the tool
definitions change for every question). The standard nemo-skills batch evaluator uses a
single global system message and cannot support this, so BFCL uses a standalone HTTP
client (`run_bfcl_fc_inference.py`) instead.

Each Slurm inference job:
1. Starts `serve_unified` in the background on `localhost:<server_port>` with
   `--decode_function_channel --tool_call_parser ... --use_function_channel_for_tool_calls`
2. Runs the inference client, which polls until the server is healthy, then sends one
   request per sample with the per-sample system prompt (tool definitions)
3. Writes `output.jsonl` with `generation` = raw `<TOOLCALL>[...]</TOOLCALL>` text
4. Kills the server and exits

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
