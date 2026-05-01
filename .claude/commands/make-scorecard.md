---
description: Generate a single-page HTML scorecard aggregating results from all S2S FC eval benchmarks. Usage: /make-scorecard [output_dir=PATH] [eval_mode=greedy|sampling|greedy+sampling] [name=FILENAME]
---

You are going to generate a comprehensive single-page HTML scorecard aggregating metrics from all S2S FC eval benchmarks.

## Arguments

Parse `$ARGUMENTS` as `key=value` tokens (whitespace-delimited).

| Key | Meaning |
|---|---|
| `output_dir` | The `--output_dir` value passed to `run_all_benchmarks.sh`. If provided, results are read from `{output_dir}/{mode}_{commit_hash}/`. If omitted, each benchmark's output dir is read from its config YAML. |
| `eval_mode` | `greedy`, `sampling`, or `greedy+sampling` (default: `greedy+sampling`). Determines which modes to include in the scorecard. |
| `name` | Output HTML filename (default: `scorecard`). Extension `.html` is appended automatically if omitted. Written to `asset/{name}.html`. |

**Default (no args):** reads output dirs from config YAMLs + current git commit hash, includes both greedy and sampling, writes `asset/scorecard.html`.

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

**BBA**: `eval-results/{category}/metrics.json` for categories `[formal_fallacies, navigate, object_counting, web_of_lies]`
Key: `bba.{category}.greedy` or `.sampling` → `accuracy`, `total`

**BFCL**: `eval-results/{cat}/metrics.json` for cats `[simple, parallel, multiple, parallel_multiple, irrelevance]`
Key: `bfcl_fc.{cat}.greedy` or `.sampling` → `accuracy`, `num_samples`, `num_correct`

**conv_behav**: `eval-results/metrics.json`
Key: `conv_behav.greedy` or `.sampling` → `tt_f1`, `barge_in_success_rate`, `bc_accuracy`, `cutoff_rate`, optionally `tt_precision`, `tt_recall`, `tt_latency_ms`, `barge_in_latency_ms`, `user_eou_f1`, `num_evaluated`

---

## Step 2 — Write the generator script

Write `/tmp/make_scorecard.py`. It embeds all metric values as Python literals, dynamically loads audio examples from disk, and writes the final HTML. No file I/O in the script except writing the output HTML.

### Script top-level constants

```python
import json, re as _re
from pathlib import Path
from datetime import date

NAME       = "scorecard"          # from name= arg (strip .html if present)
COMMIT     = "abc1234"            # from git rev-parse --short HEAD
OUTPUT_PATH = Path(f"asset/{NAME}.html")   # relative to repo root / CWD
TODAY      = date.today().isoformat()
MODES      = ["greedy", "sampling"]        # only modes actually run
SERVE_ROOT = Path("/lustre")               # HTTP server root for audio URLs
```

### Embedded metric data

```python
# Each benchmark: mode → {subtest/category → metrics_dict or None}
# Set entire mode key to None if that mode was not run.

VB_NONMCQ = {
    "greedy":   {"sd_qa": None, "alpacaeval_full": None, "alpacaeval": None,
                 "ifeval": None, "advbench": None, "commoneval": None,
                 "wildvoice": None, "alpacaeval_speaker": None},
    "sampling": None,
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
    "greedy":   {"formal_fallacies": None, "navigate": None,
                 "object_counting": None, "web_of_lies": None},
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

REPORTS = {k: {"greedy": None, "sampling": None}
           for k in ["vb_nonmcq","vb_mcq","fdb","bba","bfcl","conv_behav"]}
```

### Audio helper functions

```python
TOKEN_RE = _re.compile(r'<[^>]+>')

def clean_gen(gen):
    return TOKEN_RE.sub('', gen).strip()

def audio_url(abs_path):
    """Return root-relative URL (served from SERVE_ROOT) or None if file missing."""
    p = Path(abs_path)
    if not p.exists():
        return None
    try:
        return "/" + str(p.relative_to(SERVE_ROOT))
    except ValueError:
        return None

def audio_tag(abs_path):
    url = audio_url(abs_path)
    if url:
        return f'<audio controls src="{url}" class="aplayer"></audio>'
    return '<em style="color:var(--tx2);font-size:0.82em">audio not found</em>'

def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_audio_path(record):
    a = record.get("audio", "")
    return a.get("path", "") if isinstance(a, dict) else str(a or "")
```

### Audio example loading

Load examples dynamically from the same resolved output dirs used for metrics.
Use greedy mode for all benchmarks except conv_behav (fall back to sampling if greedy not run).

**VoiceBench CommonEval** (`vb_nonmcq_greedy_dir/eval-results/voicebench.commoneval/`):
- Load `output.jsonl` and `result-voicebench_format.jsonl` (joined by index)
- `avg_score = mean(float(x) for x in score_list)`
- Select 3 examples that each have a non-empty audio path and non-empty response text:
  - **good**: first with `avg_score ≥ 4.2`
  - **medium**: first with `2.7 ≤ avg_score ≤ 3.3` (not already selected)
  - **bad**: first with `avg_score ≤ 1.5`

**FDB turn_taking** (`fdb_greedy_dir/eval-results/fdb_v1.turn_taking/output.jsonl`):
- `tor = bool(re.search(r'<\|[\d.]+\|>', generation))`
- Select 2 `tor=True` (took turn, good ✓) + 2 `tor=False` (missed, bad ✗)

**FDB pause_candor** (`fdb_greedy_dir/eval-results/fdb_v1.pause_candor/output.jsonl`):
- `tor=False` = good (stayed silent ✓), `tor=True` = bad (interrupted ✗)
- Select 2 good + 2 bad

**BBA navigate** (`bba_greedy_dir/eval-results/navigate/output.jsonl`):
- `correct = expected_answer.lower() in clean_gen(generation).lower()[:40]`
- Select 2 correct + 2 wrong (each must have audio)

**conv_behav** (sampling dir, `eval-results/validation_logs/agent/*.wav`):
- List all `.wav` files, take first 3

### Example card builders

```python
def score_badge(score):
    c = "var(--gr)" if score >= 4 else ("var(--ye)" if score >= 3 else "var(--re)")
    return f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">GPT {score:.1f}/5</span>'

def outcome_badge(label, good):
    c = "var(--gr)" if good else "var(--re)"
    return f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{label}</span>'

def ex_card(header_html, left_html, right_html):
    return f'''<div class="ex-card">
  <div class="ex-header">{header_html}</div>
  <div class="ex-body">
    <div class="ex-left">{left_html}</div>
    <div class="ex-right">{right_html}</div>
  </div>
</div>'''
```

Card layouts:
- **VB card**: header = benchmark name + `score_badge`; left = question in `.bubble.bubble-q`; right = response (truncated 200 chars) in `.bubble.bubble-a` + `audio_tag`
- **FDB card**: header = name + `outcome_badge` + sample id `<code>`; left = problem text or `"Input audio →"`; right = generation or `"(model stayed silent)"` + `audio_tag`
- **BBA card**: header = name + `outcome_badge` + `expected: {val}`; left = category; right = generation + `audio_tag`
- **conv_behav card**: header = "conv_behav Agent Audio" + filename `<code>`; left = "Full-duplex conversation session recording."; right = `audio_tag`

---

## Step 3 — HTML scorecard specification

Single self-contained file. Dark GitHub theme. Chart.js 4.4.0 from CDN.

```css
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
        --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950;
        --ye:#d29922; --re:#f85149; }
/* Metric scorecard */
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
.mode-greedy   { background:#1f3a5f; color:#79c0ff; }
.mode-sampling { background:#3d2b00; color:#f0b429; }
/* Audio example cards */
.aplayer { width:100%; height:34px; margin-top:6px; border-radius:4px; accent-color:var(--ac); }
.ex-card { background:var(--bg2); border:1px solid var(--bd); border-radius:10px;
           margin-bottom:12px; overflow:hidden; }
.ex-header { background:var(--bg3); padding:10px 16px; font-size:0.88em;
             border-bottom:1px solid var(--bd); display:flex; align-items:center;
             gap:8px; flex-wrap:wrap; }
.ex-body { display:grid; grid-template-columns:1fr 1fr; }
.ex-left  { padding:14px 16px; border-right:1px solid var(--bd); }
.ex-right { padding:14px 16px; }
.bubble { border-radius:6px; padding:10px 12px; font-size:0.87em;
          margin-bottom:8px; line-height:1.5; }
.bubble-q { background:#0d1f3a; border-left:3px solid var(--ac); }
.bubble-a { background:#0d2a1a; border-left:3px solid var(--gr); }
.badge { display:inline-block; padding:2px 8px; border-radius:4px;
         font-size:0.8em; font-weight:600; }
@media(max-width:900px) {
  .scorecard-grid { grid-template-columns:1fr; }
  .ex-body { grid-template-columns:1fr; }
  .ex-left { border-right:none; border-bottom:1px solid var(--bd); }
}
```

### Page structure

**Header**: title `"S2S FC Eval Scorecard — {NAME}"`, date, commit, mode badges.

**Top summary row** (6 tiles, flex-wrap): one tile per benchmark.
- Each tile shows the **greedy** headline number (primary) and **sampling** headline in smaller text if available.
- Color coding on the greedy headline:
  - VB non-MCQ / MCQ: green ≥60%, yellow ≥40%, red <40%
  - BBA accuracy: green ≥70%, yellow ≥50%, red <50%
  - BFCL avg: green ≥70%, yellow ≥40%, red <40%
  - conv_behav TT-F1: green ≥75%, yellow ≥55%, red <55%
  - FDB composite: green ≥60%, yellow ≥40%, red <40%
- "Not run" tile (grey) if no data for either mode.

**Radar chart** (center, 600px): axes for VB-nonMCQ, VB-MCQ, BBA, BFCL, conv_behav, FDB-composite.
If both modes available, draw two overlapping polygons (greedy = blue, sampling = orange).
Show only if ≥3 axes have data.

**Scorecard grid** (3 cols × 2 rows):

Each card shows **greedy** metrics as primary rows, with sampling side-by-side where available.
Cards: VB non-MCQ, VB MCQ, FDB, BBA, BFCL, conv_behav — always all 6, "Not run" if missing.
FDB card shows TOR per dimension (↑ for turn_taking/interruption; ↓ for pause/backchannel).

**Comparison bar chart**: grouped bars (greedy=blue, sampling=orange) per benchmark, Y axis 0–100.

**Conclusions section**: two columns — "Strongest Areas" and "Areas for Improvement" based on greedy headline vs thresholds.

**Audio Examples section** (always included, placed after Conclusions):

```html
<h2>Audio Examples</h2>
<p style="color:var(--tx2);font-size:0.88em;margin-bottom:16px">
  Representative model outputs. Audio requires the HTTP server —
  run <code>bash scripts/serve_scorecard.sh --name {NAME}</code> and follow the printed instructions.
</p>

<div class="ex-section">
  <h3 style="color:var(--ac)">VoiceBench CommonEval — Response Quality</h3>
  <!-- 3 VB cards: good / medium / bad -->
</div>

<div class="ex-section">
  <h3 style="color:var(--ac)">FDB — Turn-Taking &amp; Pause Handling</h3>
  <!-- 2 turn_taking (✓/✗) + 2 pause_candor (✓/✗) cards -->
</div>

<div class="ex-section">
  <h3 style="color:var(--ac)">BBA Navigate — Correct vs Wrong</h3>
  <!-- 2 correct + 2 wrong cards -->
</div>

<div class="ex-section">
  <h3 style="color:var(--ac)">conv_behav — Agent Session Audio (sampling)</h3>
  <!-- 3 agent wav cards -->
</div>
```

If a benchmark has no available audio (output dir missing or no wav files found), show `<p class="not-run">No examples available.</p>` for that section.

---

## Step 4 — Headline derivations

- **VB nonMCQ**: normalize each subtest to 0–100 (`sd_qa`/`ifeval`/`advbench` already %; GPT-judge subtests on 1–5 scale → ×20); average across available subtests.
- **VB MCQ**: average of `acc` across available subtests.
- **FDB composite**: average of per-dim normalized TOR — `turn_taking`/`interruption` contribute as-is (↑ better); `backchannel`/`pause_candor`/`pause_synthetic` contribute as `1 − TOR` (↓ better). Multiply by 100.
- **BBA**: average `accuracy` across available categories.
- **BFCL weighted**: `sum(num_correct) / sum(num_samples) × 100`.
- **conv_behav**: `tt_f1` (use sampling if greedy not run).

---

## Step 5 — Run and report

Run `python3 /tmp/make_scorecard.py`.

Report:
1. Path to `asset/{name}.html`
2. To view with audio: `bash scripts/serve_scorecard.sh --name {NAME}`, then follow the printed SSH tunnel + browser URL instructions
3. Summary table: benchmark × mode → headline number
4. Which benchmarks/modes had missing results
