---
description: Generate a single-page HTML scorecard aggregating results from all S2S FC eval benchmarks. Usage: /make-scorecard [output_dir=PATH] [eval_mode=greedy|sampling|greedy+sampling]
---

You are going to generate a comprehensive single-page HTML scorecard aggregating metrics from all S2S FC eval benchmarks.

## Arguments

Parse `$ARGUMENTS` as `key=value` tokens (whitespace-delimited).

| Key | Meaning |
|---|---|
| `output_dir` | The `--output_dir` value passed to `run_all_benchmarks.sh`. If provided, results are read from `{output_dir}/{mode}_{commit_hash}/`. If omitted, each benchmark's output dir is read from its config YAML. |
| `eval_mode` | `greedy`, `sampling`, or `greedy+sampling` (default: `greedy+sampling`). Determines which modes to include in the scorecard. |

**Default (no args):** reads output dirs from config YAMLs + current git commit hash, includes both greedy and sampling.

---

## Step 1 — Resolve output directories

Run the following steps using available tools:

### 1a. Get the git commit hash
Run: `git rev-parse --short HEAD`
Store as `COMMIT`.

### 1b. Determine eval modes
From `eval_mode` arg: `greedy+sampling` → modes = `["greedy", "sampling"]`, `greedy` → `["greedy"]`, `sampling` → `["sampling"]`.
Default: `["greedy", "sampling"]`.

### 1c. Resolve output dir for each benchmark × mode

**If `output_dir` arg is provided:**
For each mode in modes, the base output dir for every benchmark is:
`{output_dir}/{mode}_{COMMIT}`

**If `output_dir` arg is NOT provided:**
Read `output_dir:` from each benchmark's config YAML, then append `_{COMMIT}`.

Config YAML locations (relative to repo root):
| Benchmark | Greedy config | Sampling config |
|---|---|---|
| `vb_nonmcq` | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc_greedy.yaml` | `...fc_sampling.yaml` |
| `vb_mcq` | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_mcq_config_fc_greedy.yaml` | `...mcq_config_fc_sampling.yaml` |
| `fdb` | `nemo_skills/dataset/fdb/scripts/fdb_s2s_incremental_v2_02mar_config_fc_greedy.yaml` | `...fc_sampling.yaml` |
| `bba` | `nemo_skills/dataset/bba/scripts/bba_config_fc_greedy.yaml` | `...bba_config_fc_sampling.yaml` |
| `bfcl` | `nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_greedy.yaml` | `...bfcl_fc_config_sampling.yaml` |
| `conv_behav` | `nemo_skills/dataset/conv_behav/scripts/conv_behav_config_greedy.yaml` | `...conv_behav_config_sampling.yaml` |

Extract the `output_dir:` field from each YAML (first non-comment line matching `^output_dir:`), then the resolved path is `{yaml_output_dir}_{COMMIT}`.

### 1d. Read all metrics files

For each benchmark × mode, read the relevant metrics files using the Read tool.
If a file does not exist, set values to `None` for that benchmark+mode.

Metrics file locations (relative to the resolved output dir):

**VoiceBench non-MCQ**: `eval-results/voicebench.{subtest}/metrics.json` for each subtest in `[sd_qa, alpacaeval_full, alpacaeval, ifeval, advbench, commoneval, wildvoice, alpacaeval_speaker]`
Key: `voicebench.{subtest}.greedy` or `.sampling` → `gpt` (%), `panda` (%), optionally `gpt_asr`, `panda_asr`

**VoiceBench MCQ**: same pattern, subtests `[bbh, openbookqa, mmsu]`

**FDB**: `eval-results/fdb_v1.{dim}/metrics.json` for dims `[backchannel, interruption, pause_candor, pause_synthetic, turn_taking]`

**BBA**: `eval-results/bba_aggregate/metrics.json`
Key: `bba.aggregate.greedy` or `.sampling` → `accuracy`, `per_category`

**BFCL**: `eval-results/{cat}/metrics.json` for cats `[simple, parallel, multiple, parallel_multiple, irrelevance]`
Key: `bfcl_fc.{cat}.greedy` or `.sampling` → `accuracy`, `num_samples`, `num_correct`

**conv_behav**: `eval-results/metrics.json`
Key: `conv_behav.greedy` or `.sampling` → `tt_f1`, `barge_in_success_rate`, `bc_accuracy`, `cutoff_rate`, optionally `tt_precision`, `tt_recall`, `tt_latency_ms`, `barge_in_latency_ms`, `user_eou_f1`, `num_evaluated`

---

## Step 2 — Write the generator script

Write `/tmp/make_scorecard.py` with all metric values embedded as Python literals — no file I/O except writing the HTML. The script receives one dict per benchmark keyed by mode.

Structure:

```python
import json
from pathlib import Path
from datetime import date

# Set to {output_dir}/scorecard.html if output_dir arg was provided,
# otherwise {repo_root}/asset/scorecard.html (asset/ relative to CWD).
OUTPUT_PATH = Path("...")
TODAY = date.today().isoformat()
MODES = ["greedy", "sampling"]   # only modes that were actually run

# ── Embedded metric data ──────────────────────────────────────────────────────
# Each benchmark dict: mode → {subtest/category → metrics_dict or None}
# Set entire mode key to None if that mode was not run for the benchmark.

VB_NONMCQ = {
    "greedy":   {"sd_qa": None, "alpacaeval_full": None, "alpacaeval": None,
                 "ifeval": None, "advbench": None, "commoneval": None,
                 "wildvoice": None, "alpacaeval_speaker": None},
    "sampling": None,   # None if not run
}
VB_MCQ = {
    "greedy":   {"bbh": None, "openbookqa": None, "mmsu": None},
    "sampling": None,
}
FDB = {
    "greedy":   {"backchannel": None, "interruption": None, "pause_candor": None,
                 "pause_synthetic": None, "turn_taking": None},
    "sampling": None,
}
BBA = {
    "greedy":   {"accuracy": None, "per_category": None},
    "sampling": None,
}
BFCL = {
    "greedy":   {"simple": None, "parallel": None, "multiple": None,
                 "parallel_multiple": None, "irrelevance": None},
    "sampling": None,
}
CONV_BEHAV = {
    "greedy":   {"tt_f1": None, "barge_in_success_rate": None, "bc_accuracy": None,
                 "cutoff_rate": None, "tt_precision": None, "tt_recall": None,
                 "tt_latency_ms": None, "barge_in_latency_ms": None,
                 "user_eou_f1": None, "num_evaluated": None},
    "sampling": None,
}

# ── Per-benchmark report links (file:// paths to report.html if they exist) ──
REPORTS = {
    "vb_nonmcq": {"greedy": None, "sampling": None},
    "vb_mcq":    {"greedy": None, "sampling": None},
    "fdb":       {"greedy": None, "sampling": None},
    "bba":       {"greedy": None, "sampling": None},
    "bfcl":      {"greedy": None, "sampling": None},
    "conv_behav":{"greedy": None, "sampling": None},
}
```

After the literals, compute derived values and render HTML as described in Step 3.

---

## Step 3 — HTML scorecard specification

Single self-contained file. Dark GitHub theme (same CSS as all other analysis skills). Chart.js 4.4.0 from CDN.

```css
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
        --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950;
        --ye:#d29922; --re:#f85149; }
.scorecard-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:16px 0; }
.bench-card { background:var(--bg2); border:1px solid var(--bd); border-radius:10px; padding:20px; }
.bench-card h3 { font-size:1em; font-weight:600; margin-bottom:14px; color:var(--ac); }
.metric-row { display:flex; justify-content:space-between; align-items:center;
              padding:5px 0; border-bottom:1px solid var(--bg3); font-size:0.88em; }
.metric-row:last-child { border-bottom:none; }
.metric-val { font-weight:700; font-size:1.05em; }
.not-run { color:var(--tx2); font-style:italic; font-size:0.85em; }
.mode-tab { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.8em;
            font-weight:600; margin-right:6px; }
.mode-greedy  { background:#1f3a5f; color:#79c0ff; }
.mode-sampling{ background:#3d2b00; color:#f0b429; }
@media(max-width:900px) { .scorecard-grid { grid-template-columns:1fr; } }
```

### Page structure

**Header**: "S2S FC Eval Scorecard", date, subtitle "Aggregated benchmark results."
Mode badges (`.mode-greedy` / `.mode-sampling`) showing which modes are included.

**Top summary row** (6 tiles, flex-wrap): one tile per benchmark.
- Each tile shows the **greedy** headline number (primary) and **sampling** headline in smaller text if available.
- Color coding on the greedy headline:
  - VB non-MCQ / MCQ GPT avg: green ≥60%, yellow ≥40%, red <40%
  - BBA accuracy: green ≥70%, yellow ≥50%, red <50%
  - BFCL avg: green ≥70%, yellow ≥40%, red <40%
  - conv_behav TT-F1: green ≥75%, yellow ≥55%, red <55%
  - FDB: "See details"
- "Not run" tile (grey) if no data for either mode.

**Radar chart** (center, 600px): axes for VB-nonMCQ, VB-MCQ, BBA, BFCL, conv_behav, FDB-composite.
If both modes available, draw two overlapping polygons (greedy = blue, sampling = orange).
Show only if ≥3 axes have data.

**Scorecard grid** (3 cols × 2 rows):

Each card shows **greedy** metrics as primary rows. If sampling is also available, add a second set of rows with a `.mode-sampling` badge prefix, or show a side-by-side delta column (greedy vs sampling).

Cards: VB non-MCQ, VB MCQ, FDB, BBA, BFCL, conv_behav — always all 6, "Not run" if missing.

FDB card shows TOR per dimension (↑ higher better for turn_taking/interruption; ↓ lower better for pause/backchannel).
All other cards show per-subtest/category accuracy rows sorted descending.

**Comparison bar chart**: grouped bars per benchmark (greedy bar + sampling bar side by side) vs target line.

**Conclusions section**: two columns — "Strongest Areas" and "Areas for Improvement" based on greedy results.

---

## Step 4 — Run and report

Run `python3 /tmp/make_scorecard.py`.

Report:
1. Path to `scorecard.html` (`{output_dir}/scorecard.html` if `output_dir` arg was given, otherwise `asset/scorecard.html`)
2. Summary table: benchmark × mode → headline number
3. Which benchmarks/modes had missing results
