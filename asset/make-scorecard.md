---
description: Generate a single-page HTML scorecard aggregating results from all S2S FC eval benchmarks. Usage: /make-scorecard [output_dir=PATH] [name=FILENAME]
---

You are going to generate a comprehensive single-page HTML scorecard aggregating metrics from all S2S FC eval benchmarks.

## Arguments

Parse `$ARGUMENTS` as `key=value` tokens (whitespace-delimited).

| Key | Meaning |
|---|---|
| `output_dir` | The `--output_dir` value passed to `run_all_benchmarks.sh`. If provided, results are read from `{output_dir}/{name}_{commit_hash}/`. If omitted, each benchmark's output dir is read from its config YAML. |
| `name` | Output HTML filename (default: `scorecard`). Extension `.html` is appended automatically if omitted. Written to `asset/{name}.html`. |

**Default (no args):** reads output dirs from config YAMLs + current git commit hash, writes `asset/scorecard.html`.

---

## Step 1 — Resolve output directories

Run the following steps using available tools:

### 1a. Get the git commit hash
Run: `git rev-parse --short HEAD`
Store as `COMMIT`.

### 1b. Resolve output dir for each benchmark

**If `output_dir` arg is provided:**
For each benchmark, the output dir is:
`{output_dir}/{benchmark}_{COMMIT}`

**If `output_dir` arg is NOT provided:**
Read `output_dir:` from each benchmark's config YAML, then append `_{COMMIT}`.

Default config YAML locations (relative to repo root):
| Benchmark | Config YAML |
|---|---|
| `vb_nonmcq` | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_config_fc_s2s_incremental_v2_greedy.yaml` |
| `vb_mcq` | `nemo_skills/dataset/voicebench/scripts/vb_matched_demo_v2_02mar_mcq_config_fc_s2s_incremental_v2_greedy.yaml` |
| `fdb` | `nemo_skills/dataset/fdb/scripts/fdb_s2s_incremental_v2_02mar_config_fc_greedy.yaml` |
| `bba` | `nemo_skills/dataset/bba/scripts/bba_config_fc_s2s_incremental_v2_greedy.yaml` |
| `bfcl` | `nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/bfcl_fc_config_s2s_incremental_v2_greedy.yaml` |
| `conv_behav` | `nemo_skills/dataset/conv_behav/scripts/conv_behav_config_greedy.yaml` |

Extract the `output_dir:` field from each YAML (first non-comment line matching `^output_dir:`), then the resolved path is `{yaml_output_dir}_{COMMIT}`.

### 1c. Read all metrics files

For each benchmark, read the relevant metrics files using the Read tool.
If a file does not exist, set values to `None` for that benchmark.

The current `metrics.json` schema is **flat** — no decoding-mode middle key:

```json
{ "fdb_v1.pause_candor": { "turn": 0.71, "tor_pct": 70.83 } }
```

Metrics file locations (relative to the resolved output dir):

**VoiceBench non-MCQ**: `eval-results/voicebench.{subtest}/metrics.json` for each subtest in `[sd_qa, alpacaeval_full, alpacaeval, ifeval, advbench, commoneval, wildvoice, alpacaeval_speaker]`
Key: `voicebench.{subtest}` → `gpt` (%), `panda` (%), optionally `gpt_asr`, `panda_asr`

**VoiceBench MCQ**: same pattern, subtests `[bbh, openbookqa, mmsu]`
Key: `voicebench.{subtest}` → `acc` (%), `acc_asr`

**FDB**: `eval-results/fdb_v1.{dim}/metrics.json` for dims `[backchannel, interruption, pause_candor, pause_synthetic, turn_taking]`
Key: `fdb_v1.{dim}` → `tor`, `tor_pct`, optionally `latency_ms`, `rating`

**BBA**: `eval-results/{category}/metrics.json` for categories `[formal_fallacies, navigate, object_counting, web_of_lies]`
Key: `bba.{category}` → `accuracy`, `total`

**BFCL**: `eval-results/{cat}/metrics.json` for cats `[simple, parallel, multiple, parallel_multiple, irrelevance]`
Key: `bfcl_fc.{cat}` → `accuracy`, `num_samples`, `num_correct`

**conv_behav**: `eval-results/metrics.json`
Key: `conv_behav` → `tt_f1`, `barge_in_success_rate`, `bc_accuracy`, `cutoff_rate`, optionally `tt_precision`, `tt_recall`, `tt_latency_ms`, `barge_in_latency_ms`, `user_eou_f1`, `num_evaluated`

---

## Step 2 — Write the generator script

Write `/tmp/make_scorecard.py`. It embeds all metric values as Python literals, dynamically loads audio examples from disk, writes the final HTML, and writes a sidecar JSON file.

### Script top-level constants

```python
import json, re as _re
from pathlib import Path
from datetime import date

NAME        = "scorecard"          # from name= arg (strip .html if present)
COMMIT      = "abc1234"            # from git rev-parse --short HEAD
OUTPUT_PATH = Path(f"asset/{NAME}.html")
JSON_PATH   = Path(f"asset/{NAME}.json")
TODAY       = date.today().isoformat()
SERVE_ROOT  = Path("/lustre")
```

### Embedded metric data

Flat layout — one dict per benchmark, keyed by subtest/category/metric. Set the whole benchmark dict to `None` if not run.

```python
VB_NONMCQ = {"sd_qa": None, "alpacaeval_full": None, "alpacaeval": None,
             "ifeval": None, "advbench": None, "commoneval": None,
             "wildvoice": None, "alpacaeval_speaker": None}
VB_MCQ    = {"bbh": None, "openbookqa": None, "mmsu": None}
FDB       = {"backchannel": None, "interruption": None, "pause_candor": None,
             "pause_synthetic": None, "turn_taking": None}
BBA       = {"formal_fallacies": None, "navigate": None,
             "object_counting": None, "web_of_lies": None}
BFCL      = {"simple": None, "parallel": None, "multiple": None,
             "parallel_multiple": None, "irrelevance": None}
CONV_BEHAV = {"tt_f1": None, "barge_in_success_rate": None, "bc_accuracy": None,
              "cutoff_rate": None, "tt_precision": None, "tt_recall": None,
              "tt_latency_ms": None, "barge_in_latency_ms": None,
              "user_eou_f1": None, "num_evaluated": None}

REPORTS = {k: None for k in ["vb_nonmcq","vb_mcq","fdb","bba","bfcl","conv_behav"]}
```

Replace each `None` with the metrics dict (or leave `None` if the file is missing).

### Metric filtering and formatting helpers

```python
_ASR_SUFFIXES = ("_asr",)
_ASR_PREFIXES = ("agent_",)
_REDUNDANT = {"tor", "turn", "latency"}

def non_asr_items(m):
    if not m: return {}
    return {k: v for k, v in m.items()
            if not any(k.endswith(s) for s in _ASR_SUFFIXES)
            and not any(k.startswith(p) for p in _ASR_PREFIXES)
            and k not in _REDUNDANT}

_FRAC_KEYS  = {"refusal_rate","final","strict-prompt","strict-instruction",
               "loose-prompt","loose-instruction","fail"}
_SCORE5_KEYS = {"gpt","rating"}
_PCT_KEYS   = {"panda","acc","accuracy","tor_pct","tt_f1","tt_precision",
               "tt_recall","barge_in_success_rate","bc_accuracy","cutoff_rate"}
_MS_KEYS    = {"latency_ms","tt_latency_ms","barge_in_latency_ms"}
_COUNT_KEYS = {"total","num_samples","num_correct","num_evaluated"}

def smart_fmt(key, val):
    if val is None: return "—"
    if key in _MS_KEYS:     return f"{val:.0f}ms"
    if key in _COUNT_KEYS:  return str(int(val))
    if key in _SCORE5_KEYS: return f"{val:.2f}/5"
    if key in _PCT_KEYS:    return f"{val:.1f}%"
    if key in _FRAC_KEYS:   return f"{val*100:.1f}%"
    if isinstance(val, float): return f"{val:.3f}"
    return str(val)

def sub_metric_rows(label, m):
    """Return HTML rows for all non-ASR metrics of a subtest/category."""
    items = non_asr_items(m)
    rows = [f'<div class="sub-hdr">{label}</div>']
    for key, val in items.items():
        rows.append(metric_row(f"  {key}", val,
                               fmt_fn=lambda x, k=key: smart_fmt(k, x)))
    return "".join(rows)
```

`metric_row(label, val, fmt_fn)` should emit a `.metric-row` div with the label on the left and `fmt_fn(val)` on the right (use `.not-run` styling when `val is None`).

### Audio helper functions

```python
TOKEN_RE = _re.compile(r'<[^>]+>')

def clean_gen(gen):
    return TOKEN_RE.sub('', gen).strip()

def audio_url(abs_path):
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

**VoiceBench CommonEval** (`vb_nonmcq_dir/eval-results/voicebench.commoneval/`):
- Load `output.jsonl` and `result-voicebench_format.jsonl` (joined by index)
- `avg_score = mean(float(x) for x in score_list)`
- Select 3 examples with non-empty audio + response:
  - **good**: first with `avg_score ≥ 4.2`
  - **medium**: first with `2.7 ≤ avg_score ≤ 3.3`
  - **bad**: first with `avg_score ≤ 1.5`

**FDB turn_taking** (`fdb_dir/eval-results/fdb_v1.turn_taking/output.jsonl`):
- `tor = bool(re.search(r'<\|[\d.]+\|>', generation))`
- 2 `tor=True` (took turn ✓) + 2 `tor=False` (missed ✗)

**FDB pause_candor** (`fdb_dir/eval-results/fdb_v1.pause_candor/output.jsonl`):
- `tor=False` = good (silent ✓), `tor=True` = bad (interrupted ✗)
- 2 good + 2 bad

**BBA navigate** (`bba_dir/eval-results/navigate/output.jsonl`):
- `correct = expected_answer.lower() in clean_gen(generation).lower()[:40]`
- 2 correct + 2 wrong

**conv_behav** (`conv_behav_dir/eval-results/validation_logs/agent/*.wav`):
- Take first 3 wav files

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

### Scorecard grid card builder

For VB nonMCQ, VB MCQ, FDB, BBA, and BFCL cards, use `sub_metric_rows` to show **all** non-ASR metrics for each subtest/category. Each call emits a `.sub-hdr` divider followed by one `.metric-row` per key, formatted with `smart_fmt`.

```python
# VB nonMCQ
_vb_rows = []
for sub in ["sd_qa","alpacaeval_full","alpacaeval","ifeval","advbench","commoneval","wildvoice","alpacaeval_speaker"]:
    _vb_rows.append(sub_metric_rows(sub, (VB_NONMCQ or {}).get(sub)))
vb_card_html = bench_card("VB non-MCQ", "".join(_vb_rows), _hl["vb_nonmcq"], "vb_nonmcq")

# VB MCQ
_mcq_rows = []
for sub in ["bbh","openbookqa","mmsu"]:
    _mcq_rows.append(sub_metric_rows(sub, (VB_MCQ or {}).get(sub)))
mcq_card_html = bench_card("VB MCQ", "".join(_mcq_rows), _hl["vb_mcq"], "vb_mcq")

# FDB
_fdb_rows = []
for dim in ["turn_taking","interruption","backchannel","pause_candor","pause_synthetic"]:
    _fdb_rows.append(sub_metric_rows(dim, (FDB or {}).get(dim)))
fdb_card_html = bench_card("FDB", "".join(_fdb_rows), _hl["fdb"], "fdb")

# BBA
_bba_rows = []
for cat in ["formal_fallacies","navigate","object_counting","web_of_lies"]:
    _bba_rows.append(sub_metric_rows(cat, (BBA or {}).get(cat)))
bba_card_html = bench_card("BBA", "".join(_bba_rows), _hl["bba"], "bba")

# BFCL
_bfcl_rows = []
for cat in ["simple","parallel","multiple","parallel_multiple","irrelevance"]:
    _bfcl_rows.append(sub_metric_rows(cat, (BFCL or {}).get(cat)))
bfcl_card_html = bench_card("BFCL", "".join(_bfcl_rows), _hl["bfcl"], "bfcl")
```

`bench_card(title, rows_html, headline, anchor)` should render a `.bench-card` with the title, headline number (color-coded per Step 4), and the metric rows. conv_behav uses plain `metric_row` calls (single flat dict, no subtest grouping).

### Step 2g — Sidecar JSON output

At the end of the script body (after writing the HTML), also write `JSON_PATH`. Build sidecar variables alongside the HTML so they can be reused without recomputation.

```python
# Headlines: same per-benchmark values as the summary tiles (Step 4).
# Float or None.
_hl = {
    "vb_nonmcq":  vb_headline(VB_NONMCQ),
    "vb_mcq":     vb_mcq_headline(VB_MCQ),
    "fdb":        fdb_headline(FDB),
    "bba":        bba_headline(BBA),
    "bfcl":       bfcl_headline(BFCL),
    "conv_behav": (CONV_BEHAV or {}).get("tt_f1"),
}

# Detailed: per-subtest display values (same floats as Detailed Scores section).
_det = {
    "vb_nonmcq": {k: (VB_NONMCQ[k]["gpt"] if (VB_NONMCQ or {}).get(k) else None)
                  for k in (VB_NONMCQ or {})},
    "vb_mcq":    {k: (VB_MCQ[k]["acc"]  if (VB_MCQ or {}).get(k) else None)
                  for k in (VB_MCQ or {})},
    "fdb":       {k: (round((FDB[k].get("tor") or 0)*100, 2) if (FDB or {}).get(k) else None)
                  for k in (FDB or {})},
    "bba":       {k: ((BBA[k].get("accuracy")) if (BBA or {}).get(k) else None)
                  for k in (BBA or {})},
    "bfcl":      {k: ((BFCL[k].get("accuracy")) if (BFCL or {}).get(k) else None)
                  for k in (BFCL or {})},
    "conv_behav": dict(CONV_BEHAV or {}) or None,
}

# audio_examples: built during audio loading; keep a stable "key" per entry
# for cross-ckpt matching in compare_ckpts.py.
_audio = {
    "vb_commoneval":    [],   # {"key", "quality", "score", "question", "response", "audio_src"}
    "fdb_turn_taking":  [],   # {"key", "sample_id", "outcome", "generation", "audio_src"}
    "fdb_pause_candor": [],   # {"key", "sample_id", "outcome", "context", "generation", "audio_src"}
    "bba_navigate":     [],   # {"key", "correct", "expected", "response", "audio_src"}
    "conv_behav":       [],   # {"key", "filename", "audio_src"}
}

with open(JSON_PATH, "w") as _f:
    json.dump({
        "name": NAME, "commit": COMMIT, "date": TODAY,
        "headlines": _hl, "detailed": _det, "audio_examples": _audio,
    }, _f, indent=2)
print(f"Sidecar: {JSON_PATH}")
```

---

## Step 3 — HTML scorecard specification

Single self-contained file. Dark GitHub theme. Chart.js 4.4.0 from CDN.

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
.sub-hdr { font-size:0.78em; font-weight:700; text-transform:uppercase;
           letter-spacing:.05em; color:var(--tx2); padding:8px 0 2px; }
@media(max-width:900px) {
  .scorecard-grid { grid-template-columns:1fr; }
  .ex-body { grid-template-columns:1fr; }
  .ex-left { border-right:none; border-bottom:1px solid var(--bd); }
}
```

### Page structure

**Header**: title `"S2S FC Eval Scorecard — {NAME}"`, date, commit.

**Top summary row** (6 tiles, flex-wrap): one tile per benchmark. Each tile shows the benchmark headline number, color-coded:
- VB non-MCQ / MCQ: green ≥60%, yellow ≥40%, red <40%
- BBA accuracy: green ≥70%, yellow ≥50%, red <50%
- BFCL avg: green ≥70%, yellow ≥40%, red <40%
- conv_behav TT-F1: green ≥75%, yellow ≥55%, red <55%
- FDB composite: green ≥60%, yellow ≥40%, red <40%
- "Not run" tile (grey) if no data.

**Radar chart** (center, 600px): axes for VB-nonMCQ, VB-MCQ, BBA, BFCL, conv_behav, FDB-composite. Single polygon. Show only if ≥3 axes have data.

**Scorecard grid** (3 cols × 2 rows): VB non-MCQ, VB MCQ, FDB, BBA, BFCL, conv_behav — always all 6, "Not run" if missing. FDB card shows TOR per dimension (↑ for turn_taking/interruption; ↓ for pause/backchannel).

**Comparison bar chart**: bars per benchmark, Y axis 0–100.

**Conclusions section**: two columns — "Strongest Areas" and "Areas for Improvement" based on headline vs thresholds.

**Audio Examples section** (always included, after Conclusions):

```html
<h2>Audio Examples</h2>
<p style="color:var(--tx2);font-size:0.88em;margin-bottom:16px">
  Representative model outputs. Audio requires the HTTP server —
  run <code>bash scripts/serve_scorecard.sh --name {NAME}</code> and follow the printed instructions.
</p>

<div class="ex-section">
  <h3 style="color:var(--ac)">VoiceBench CommonEval — Response Quality</h3>
  <!-- 3 cards: good / medium / bad -->
</div>
<div class="ex-section">
  <h3 style="color:var(--ac)">FDB — Turn-Taking &amp; Pause Handling</h3>
  <!-- 2 turn_taking + 2 pause_candor -->
</div>
<div class="ex-section">
  <h3 style="color:var(--ac)">BBA Navigate — Correct vs Wrong</h3>
  <!-- 2 correct + 2 wrong -->
</div>
<div class="ex-section">
  <h3 style="color:var(--ac)">conv_behav — Agent Session Audio</h3>
  <!-- 3 agent wav cards -->
</div>
```

If a benchmark has no available audio, show `<p class="not-run">No examples available.</p>`.

---

## Step 4 — Headline derivations

- **VB nonMCQ**: normalize each subtest to 0–100 (`sd_qa`/`ifeval`/`advbench` already %; GPT-judge subtests on 1–5 scale → ×20); average across available subtests.
- **VB MCQ**: average `acc` across available subtests:
  ```python
  def vb_mcq_headline(data):
      if not data: return None
      vals = [v.get("acc") for v in data.values() if v and v.get("acc") is not None]
      return round(sum(vals)/len(vals), 2) if vals else None
  ```
- **FDB composite**: per-dim normalized TOR — `turn_taking`/`interruption` contribute as-is (↑ better); `backchannel`/`pause_candor`/`pause_synthetic` contribute as `100 − tor_pct` (↓ better). Average across dims.
- **BBA**: average `accuracy` across available categories.
- **BFCL weighted**: `sum(num_correct) / sum(num_samples) × 100`.
- **conv_behav**: `tt_f1`.

---

## Step 5 — Run and report

Run `python3 /tmp/make_scorecard.py`.

Report:
1. Path to `asset/{name}.html`
2. Path to `asset/{name}.json` (sidecar)
3. To view with audio: `bash scripts/serve_scorecard.sh --name {NAME}`
4. Summary table: benchmark → headline number
5. Which benchmarks had missing results

---

## Step 6 — Conditionally emit compare_ckpts.py

After Step 5, count JSON files in `asset/`:

```bash
ls asset/*.json 2>/dev/null | wc -l
```

If the count is **≥ 2**, write `scripts/compare_ckpts.py` with the following content. If only 1 JSON exists, skip this step.

```python
#!/usr/bin/env python3
"""
Compare S2S FC eval results across multiple checkpoints.

Usage:
  python3 scripts/compare_ckpts.py [JSON ...] [--output PATH]

Positional args: sidecar .json paths (default: asset/*.json, sorted by name)
--output / -o : output HTML path (default: asset/comparison.html)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

COLORS = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149",
    "#a5d6ff", "#7ee787", "#ffa657", "#ff7b72",
    "#79c0ff", "#56d364", "#e3b341", "#ffa198",
]

BENCHMARKS = ["vb_nonmcq", "vb_mcq", "fdb", "bba", "bfcl", "conv_behav"]
BENCH_LABELS = {
    "vb_nonmcq":  "VB nonMCQ",
    "vb_mcq":     "VB MCQ",
    "fdb":        "FDB",
    "bba":        "BBA",
    "bfcl":       "BFCL",
    "conv_behav": "conv_behav",
}
BENCH_THRESHOLDS = {
    "vb_nonmcq":  (60, 40),
    "vb_mcq":     (60, 40),
    "fdb":        (60, 40),
    "bba":        (70, 50),
    "bfcl":       (70, 40),
    "conv_behav": (75, 55),
}
BENCH_SUBTESTS = {
    "vb_nonmcq":  ["sd_qa", "alpacaeval", "alpacaeval_full", "commoneval",
                   "wildvoice", "alpacaeval_speaker", "ifeval", "advbench"],
    "vb_mcq":     ["bbh", "openbookqa", "mmsu"],
    "fdb":        ["backchannel", "interruption", "pause_candor", "pause_synthetic", "turn_taking"],
    "bba":        ["formal_fallacies", "navigate", "object_counting", "web_of_lies"],
    "bfcl":       ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"],
    "conv_behav": ["tt_f1", "tt_precision", "tt_recall", "barge_in_success_rate",
                   "bc_accuracy", "cutoff_rate", "tt_latency_ms", "barge_in_latency_ms"],
}
RAW_METRICS = {"tt_latency_ms", "barge_in_latency_ms", "num_evaluated"}

AUDIO_CATS = [
    ("vb_commoneval",    "VoiceBench CommonEval — Response Quality"),
    ("fdb_turn_taking",  "FDB — Turn-Taking"),
    ("fdb_pause_candor", "FDB — Pause-Candor"),
    ("bba_navigate",     "BBA Navigate — Correct vs Wrong"),
    ("conv_behav",       "conv_behav — Agent Sessions"),
]


def color_val(bench, val):
    if val is None:
        return "var(--tx2)"
    hi, lo = BENCH_THRESHOLDS[bench]
    return "var(--gr)" if val >= hi else ("var(--ye)" if val >= lo else "var(--re)")


def headline(ckpt, bench):
    return (ckpt.get("headlines") or {}).get(bench)


def fmt_val(bench, sub, v):
    if v is None:
        return "—"
    if bench == "conv_behav" and sub in RAW_METRICS:
        return f"{v:.0f}"
    return f"{v:.1f}%"


def _js(obj):
    return json.dumps(obj)


def summary_table(ckpts):
    hdrs = "".join(
        f'<th style="color:{COLORS[i % len(COLORS)]};white-space:nowrap">'
        f'{ck["name"]}<br><small style="font-weight:400;color:var(--tx2)">'
        f'{ck.get("commit", "")}</small></th>'
        for i, ck in enumerate(ckpts)
    )
    rows = ""
    for bench in BENCHMARKS:
        cells = ""
        for ck in ckpts:
            v = headline(ck, bench)
            col = color_val(bench, v)
            txt = f"{v:.1f}%" if v is not None else "—"
            cells += f'<td style="text-align:center;color:{col};font-weight:700">{txt}</td>'
        rows += f"<tr><td><strong>{BENCH_LABELS[bench]}</strong></td>{cells}</tr>"
    return (
        f'<div style="overflow-x:auto"><table class="cmp-table">'
        f'<thead><tr><th>Benchmark</th>{hdrs}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def radar_data(ckpts):
    datasets = []
    for i, ck in enumerate(ckpts):
        vals = [round(headline(ck, b), 1) if headline(ck, b) is not None else 0
                for b in BENCHMARKS]
        c = COLORS[i % len(COLORS)]
        datasets.append({
            "label": ck["name"],
            "data": vals,
            "backgroundColor": c + "33",
            "borderColor": c,
            "borderWidth": 2,
            "pointBackgroundColor": c,
        })
    return _js({"labels": [BENCH_LABELS[b] for b in BENCHMARKS], "datasets": datasets})


def bar_charts(ckpts):
    html_parts, js_parts = [], []
    for bench in BENCHMARKS:
        cid = f"bar_{bench}"
        labels = [ck["name"] for ck in ckpts]
        vals, bg, bd = [], [], []
        for i, ck in enumerate(ckpts):
            c = COLORS[i % len(COLORS)]
            v = headline(ck, bench)
            vals.append(round(v, 1) if v is not None else None)
            bg.append(c + "cc"); bd.append(c)
        chart_data = {
            "labels": labels,
            "datasets": [{"label": BENCH_LABELS[bench], "data": vals,
                          "backgroundColor": bg, "borderColor": bd, "borderWidth": 1}],
        }
        html_parts.append(
            f'<div class="chart-wrap">'
            f'<h3 style="color:var(--ac);margin-bottom:8px">{BENCH_LABELS[bench]}</h3>'
            f'<canvas id="{cid}" height="100"></canvas></div>'
        )
        js_parts.append(
            f"new Chart(document.getElementById('{cid}'),{{"
            f"type:'bar',data:{_js(chart_data)},"
            f"options:{{responsive:true,"
            f"plugins:{{legend:{{labels:{{color:'#e6edf3'}}}}}},"
            f"scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},"
            f"y:{{min:0,max:100,ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}}}})"
        )
    bar_grid = f'<div class="bar-grid">{"".join(html_parts)}</div>'
    return bar_grid, "\n".join(js_parts)


def detailed_tables(ckpts):
    parts = ["<h2>Detailed Metrics</h2>"]
    for bench in BENCHMARKS:
        subtests = BENCH_SUBTESTS.get(bench, [])
        col_hdrs = "".join(
            f'<th style="color:{COLORS[i % len(COLORS)]};text-align:center">{ck["name"]}</th>'
            for i, ck in enumerate(ckpts)
        )
        rows = ""
        for sub in subtests:
            cells = ""
            for ck in ckpts:
                det = (ck.get("detailed") or {}).get(bench) or {}
                v = det.get(sub) if isinstance(det, dict) else None
                if v is None:
                    cells += '<td style="color:var(--tx2);text-align:center">—</td>'
                else:
                    cells += f'<td style="text-align:center;font-weight:600">{fmt_val(bench, sub, v)}</td>'
            rows += f'<tr><td style="white-space:nowrap">{sub}</td>{cells}</tr>'
        parts.append(
            f'<h3 style="color:var(--ac);margin:16px 0 6px">{BENCH_LABELS[bench]}</h3>'
            f'<div style="overflow-x:auto;margin-bottom:8px"><table class="cmp-table">'
            f'<thead><tr><th>Metric</th>{col_hdrs}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )
    return "\n".join(parts)


def _ex_header(cat, ex):
    if cat == "vb_commoneval":
        sc = ex.get("score")
        q = ex.get("quality", "")
        cs = {"good": "var(--gr)", "medium": "var(--ye)", "bad": "var(--re)"}.get(q, "var(--tx2)")
        badge = (f'<span class="badge" style="background:{cs}20;color:{cs};border:1px solid {cs}">'
                 f'GPT {sc:.1f}/5</span>') if sc is not None else ""
        return f"<strong>VoiceBench CommonEval</strong> {badge}"
    if cat == "fdb_turn_taking":
        sid = ex.get("sample_id", ex.get("key", ""))
        good = ex.get("outcome") == "took_turn"
        c = "var(--gr)" if good else "var(--re)"
        lbl = "TOOK TURN ✓" if good else "MISSED TURN ✗"
        return (f'<strong>FDB Turn-Taking</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' <code style="font-size:0.78em">{sid}</code>')
    if cat == "fdb_pause_candor":
        sid = ex.get("sample_id", ex.get("key", ""))
        good = ex.get("outcome") == "silent"
        c = "var(--gr)" if good else "var(--re)"
        lbl = "SILENT ✓" if good else "INTERRUPTED ✗"
        return (f'<strong>FDB Pause-Candor</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' <code style="font-size:0.78em">{sid}</code>')
    if cat == "bba_navigate":
        correct = ex.get("correct", False)
        expected = ex.get("expected", "")
        c = "var(--gr)" if correct else "var(--re)"
        lbl = "CORRECT ✓" if correct else "WRONG ✗"
        return (f'<strong>BBA Navigate</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' expected: <em>{expected}</em>')
    if cat == "conv_behav":
        fname = ex.get("filename", ex.get("key", ""))
        return f'<strong>conv_behav</strong> <code style="font-size:0.78em">{fname}</code>'
    return "<strong>Example</strong>"


def _ex_context(cat, ex):
    if cat == "vb_commoneval":
        return f'<div class="bubble bubble-q"><strong>Q:</strong> {ex.get("question", "")}</div>'
    if cat == "fdb_turn_taking":
        return '<div class="bubble bubble-q"><strong>Input audio →</strong></div>'
    if cat == "fdb_pause_candor":
        ctx = (ex.get("context") or "")[:150]
        return f'<div class="bubble bubble-q"><strong>User context:</strong><br>{ctx}…</div>'
    if cat == "bba_navigate":
        return '<div class="bubble bubble-q"><strong>Category:</strong> navigate</div>'
    if cat == "conv_behav":
        return '<div class="bubble bubble-q">Full-duplex conversation session recording.</div>'
    return ""


def _audio_row(cat, ckpt_name, color, ex):
    badge = (f'<span class="ckpt-badge" style="background:{color}22;color:{color};'
             f'border:1px solid {color}">{ckpt_name}</span>')
    src = (ex or {}).get("audio_src")
    audio = (f'<audio controls src="{src}" class="aplayer"></audio>'
             if src else '<em style="color:var(--tx2);font-size:0.82em">audio not available</em>')
    resp = ""
    if ex:
        if cat == "vb_commoneval":
            r = (ex.get("response") or "")[:200]
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px"><strong>A:</strong> {r}…</div>'
        elif cat in ("fdb_turn_taking", "fdb_pause_candor"):
            g = (ex.get("generation") or "").strip() or "(silent)"
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px">{g[:150]}</div>'
        elif cat == "bba_navigate":
            r = (ex.get("response") or "")[:150]
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px"><strong>R:</strong> {r}</div>'
    return f'<div class="audio-row">{badge}{resp}{audio}</div>'


def audio_section(ckpts):
    parts = [
        "<h2>Audio Comparison</h2>",
        '<p style="color:var(--tx2);font-size:0.88em;margin-bottom:16px">'
        "Representative outputs across checkpoints. Audio requires the HTTP server — "
        "run <code>bash scripts/serve_scorecard.sh --name comparison</code>.</p>",
    ]
    for cat, cat_label in AUDIO_CATS:
        ckpt_maps = [
            {e.get("key", str(i)): e for i, e in enumerate((ck.get("audio_examples") or {}).get(cat) or [])}
            for ck in ckpts
        ]
        all_keys = []
        seen = set()
        for cm in ckpt_maps:
            for k in cm:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        if not all_keys:
            continue
        parts.append(f'<div class="ex-section"><h3 style="color:var(--ac)">{cat_label}</h3>')
        for key in all_keys:
            ctx_ex = next((cm[key] for cm in ckpt_maps if key in cm), None)
            if ctx_ex is None:
                continue
            rows = [
                _audio_row(cat, ck["name"], COLORS[i % len(COLORS)], cm.get(key))
                for i, (ck, cm) in enumerate(zip(ckpts, ckpt_maps))
            ]
            visible = "".join(rows[:3])
            hidden_rows = rows[3:]
            right = visible
            if hidden_rows:
                right += (
                    f'<details class="more-ckpts">'
                    f'<summary>{len(hidden_rows)} more checkpoint(s) — click to expand</summary>'
                    f'{"".join(hidden_rows)}</details>'
                )
            parts.append(
                f'<div class="ex-card">'
                f'<div class="ex-header">{_ex_header(cat, ctx_ex)}</div>'
                f'<div class="ex-body-cmp">'
                f'<div class="ex-left">{_ex_context(cat, ctx_ex)}</div>'
                f'<div class="ex-right-cmp">{right}</div>'
                f'</div></div>'
            )
        parts.append("</div>")
    return "\n".join(parts)


CSS = """
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
        --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950;
        --ye:#d29922; --re:#f85149; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--tx);
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        line-height:1.5; padding:24px; max-width:1280px; margin:0 auto; }
a { color:var(--ac); }
h1 { font-size:1.7em; margin-bottom:6px; }
h2 { font-size:1.1em; font-weight:600; color:var(--ac); margin:24px 0 10px;
      border-bottom:1px solid var(--bd); padding-bottom:6px; }
h3 { font-size:0.95em; font-weight:600; margin-bottom:10px; }
.meta-row { display:flex; gap:20px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
.meta-item { color:var(--tx2); font-size:0.88em; }
.meta-item strong { color:var(--tx); }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.ckpt-pill { display:inline-block; padding:4px 14px; border-radius:20px;
              font-size:0.85em; font-weight:600; }
.cmp-table { width:100%; border-collapse:collapse; font-size:0.87em; }
.cmp-table th, .cmp-table td { padding:7px 12px; border:1px solid var(--bd); }
.cmp-table thead th { background:var(--bg3); }
.cmp-table tbody tr:hover { background:var(--bg2); }
.chart-wrap { background:var(--bg2); border:1px solid var(--bd); border-radius:10px;
               padding:20px; margin:16px 0; }
.bar-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0; }
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
.ex-section { margin:16px 0; }
.ex-card { background:var(--bg2); border:1px solid var(--bd); border-radius:10px;
            margin-bottom:12px; overflow:hidden; }
.ex-header { background:var(--bg3); padding:10px 16px; font-size:0.88em;
              border-bottom:1px solid var(--bd); display:flex; align-items:center;
              gap:8px; flex-wrap:wrap; }
.ex-body-cmp { display:grid; grid-template-columns:260px 1fr; }
.ex-left  { padding:14px 16px; border-right:1px solid var(--bd); }
.ex-right-cmp { padding:14px 16px; }
.audio-row { margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid var(--bg3); }
.audio-row:last-child { border-bottom:none; margin-bottom:0; }
.ckpt-badge { display:inline-block; padding:2px 8px; border-radius:4px;
               font-size:0.78em; font-weight:600; margin-bottom:4px; }
.bubble { border-radius:6px; padding:10px 12px; font-size:0.87em;
           margin-bottom:6px; line-height:1.5; }
.bubble:last-child { margin-bottom:0; }
.bubble-q { background:#0d1f3a; border-left:3px solid var(--ac); }
.bubble-a { background:#0d2a1a; border-left:3px solid var(--gr); }
.badge { display:inline-block; padding:2px 8px; border-radius:4px;
          font-size:0.8em; font-weight:600; }
details.more-ckpts summary { cursor:pointer; color:var(--ac); font-size:0.85em;
                               padding:6px 0; user-select:none; }
details.more-ckpts summary:hover { text-decoration:underline; }
@media(max-width:900px) {
  .bar-grid { grid-template-columns:1fr; }
  .ex-body-cmp { grid-template-columns:1fr; }
  .ex-left { border-right:none; border-bottom:1px solid var(--bd); }
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", nargs="*", help="Sidecar JSON files (default: asset/*.json)")
    p.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = p.parse_args()

    paths = [Path(x) for x in args.jsons] if args.jsons else sorted(Path("asset").glob("*.json"))
    if not paths:
        sys.exit("No sidecar JSON files found. Run /make-scorecard first to generate them.")

    ckpts = []
    for path in paths:
        with open(path) as f:
            ck = json.load(f)
        ck.setdefault("_path", str(path))
        ckpts.append(ck)

    print(f"Loaded {len(ckpts)} checkpoint(s): {[c['name'] for c in ckpts]}")

    out = Path(args.output) if args.output else Path("asset/comparison.html")
    today = date.today().isoformat()

    legend_html = "".join(
        f'<span class="ckpt-pill" style="background:{COLORS[i % len(COLORS)]}22;'
        f'color:{COLORS[i % len(COLORS)]};border:1px solid {COLORS[i % len(COLORS)]}">'
        f'{ck["name"]} <span style="font-weight:400;opacity:0.65">({ck.get("commit", "?")})</span>'
        f'</span>'
        for i, ck in enumerate(ckpts)
    )

    bar_grid, bar_js = bar_charts(ckpts)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2S FC Eval — Checkpoint Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<h1>S2S FC Eval — Checkpoint Comparison</h1>
<div class="meta-row">
  <span class="meta-item">Generated: <strong>{today}</strong></span>
  <span class="meta-item">Checkpoints: <strong>{len(ckpts)}</strong></span>
</div>
<div class="legend">{legend_html}</div>

<h2>Summary</h2>
{summary_table(ckpts)}

<h2>Radar Overview</h2>
<div class="chart-wrap" style="max-width:560px;margin:16px auto">
  <canvas id="radarChart" height="400"></canvas>
</div>

<h2>Per-Benchmark Scores</h2>
{bar_grid}

{detailed_tables(ckpts)}

{audio_section(ckpts)}

<script>
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {radar_data(ckpts)},
  options: {{
    responsive: true,
    scales: {{ r: {{
      min: 0, max: 100,
      ticks: {{ color:'#8b949e', backdropColor:'transparent', stepSize:20 }},
      grid: {{ color:'#30363d' }},
      pointLabels: {{ color:'#e6edf3', font:{{ size:13 }} }}
    }} }},
    plugins: {{ legend: {{ labels: {{ color:'#e6edf3' }} }} }}
  }}
}});
{bar_js}
</script>
</body>
</html>"""

    out.write_text(page)
    print(f"Written: {out}")
    print(f"View with: bash scripts/serve_scorecard.sh --name {out.stem}")


if __name__ == "__main__":
    main()
```
