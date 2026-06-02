---
name: voice-chat_analyze_detailed-spreadsheet
description: Analyze a detailed voice-chat checkpoint comparison Excel spreadsheet (multiple tabs, one tab per benchmark variant) and generate a comprehensive self-contained HTML report with figures, a detailed metrics section with clickable TOC, Top 3 recommendations with latency, and all sub-metric charts.
---

# voice-chat_analyze_detailed-spreadsheet

Analyze a detailed checkpoint comparison `.xlsx` file (37 tabs, one per benchmark variant) and generate `report_<date>.html`.

## Input

Find the `.xlsx` file in the current working directory. Ask the user if multiple xlsx files are present.

## Data pipeline

**Step 1 — Extract all tabs to JSON** (openpyxl, write to `$TMPDIR`):

```python
import sys, os, json
sys.path.insert(0, os.path.join(os.environ['TMPDIR'], 'pylibs'))
# pip install openpyxl --target=$TMPDIR/pylibs  if not available
import openpyxl

wb = openpyxl.load_workbook('file.xlsx', data_only=True)
data = {}
for name in wb.sheetnames:
    ws = wb[name]
    data[name] = [[cell.value for cell in row] for row in ws.iter_rows()]
with open(os.path.join(os.environ['TMPDIR'], 'xlsx_data.json'), 'w') as f:
    json.dump(data, f)
```

Write the script to a `.py` file and run with `PYTHONPATH=$TMPDIR/pylibs python3 script.py` — do not use heredocs for Python code containing operators like `!=` (bash escaping causes syntax errors).

**Step 2 — Compute normalized scores** using `compute_scores.py` (reads `$TMPDIR/xlsx_data.json`, outputs JSON). See normalization rules below.

**Step 3 — Generate JS arrays** from the JSON, embed inline in the HTML.

## Tab structure and key columns

Each tab has a header row followed by data rows. The `ckpt` column contains the full checkpoint name. Parse with a `rows_to_dict()` helper that converts each row to a dict keyed by the header.

| Tab group | Key tabs | Key columns |
|---|---|---|
| **vb_nonmcq** | `.alpacaeval`, `.ifeval`, `.commoneval`, `.wildvoice`, `.alpacaeval_speaker`, `.sd_qa`, `.advbench` | `gpt` (GPT score /5), `final` (IFEval), `refusal_rate` (AdvBench), `panda` (SDQA) |
| **vb_mcq** | `.bbh`, `.openbookqa`, `.mmsu` | `acc` (accuracy %) |
| **fdb_v1** | `.backchannel`, `.interruption`, `.pause_candor`, `.pause_synthetic`, `.turn_taking` | `tor_pct`, `jsd`, `latency_ms`, `rating` |
| **fdb_v1_5** | `.background_speech`, `.talking_to_other`, `.backchannel`, `.interruption` | `behavior_ratios.C_RESPOND`, `behavior_ratios.C_RESUME`, `stop_latency_ms`, `response_latency` |
| **fdb_v3** | *(headline only)* | `headline.greedy` |
| **bba** | `.formal_fallacies`, `.navigate`, `.object_counting`, `.web_of_lies` | `accuracy` |
| **bfcl** | `.simple`, `.multiple`, `.parallel`, `.parallel_multiple`, `.irrelevance` | `accuracy` |
| **bfcl** (headline) | `bfcl` | `headline.greedy` |
| **conv_behav** | *(single tab)* | `headline.greedy`, `greedy.overall.tt_precision`, `greedy.overall.tt_recall`, `greedy.overall.tt_f1`, `greedy.overall.cutoff_rate`, `greedy.overall.tt_latency` |

## Checkpoint naming

Full name format: `eXXX-stepYYYYY-tts-eartts-...`  
Short label: `eXXX@Yk` where Y = step // 1000

```python
def short(c):
    parts = c.split('-')
    exp = parts[0]
    step_part = next((p for p in parts if p.startswith('step')), None)
    k = int(step_part.replace('step','')) // 1000 if step_part else 0
    return f"{exp}@{k}k"
```

## Score normalization (all to 0–100 scale)

| Metric | Raw | Normalized |
|---|---|---|
| **AlpacaEval Full** GPT | /5 score | × 20 — use tab `vb_nonmcq.alpacaeval_full`, NOT base `vb_nonmcq.alpacaeval` |
| CommonEval GPT | /5 score | × 20 |
| WildVoice GPT | /5 score | × 20 |
| IFEval | 0–1 rate or % | × 100 if ≤ 1.0, else as-is |
| AdvBench refusal_rate | 0–1 rate | × 100 if ≤ 1.0, else as-is |
| **SDQA** | — | avg(PANDA, GPT) from `vb_nonmcq.sd_qa` — NOT PANDA alone |
| BBH / OpenBookQA / MMSU | already % | as-is |
| VoiceBench avg | — | see formula below |

**⚠ VoiceBench normalized average — use tracker formula exactly:**

```
VB_AVG = (OpenBookQA + MMSU + SDQA_avg + BBH + IFEval + AdvBench
          + 20*(CommonEval_gpt + AlpacaEval_Full_gpt + WildVoice_gpt))
         / count_of_non_null_items
```

Where `count_of_non_null_items` counts each raw column separately (e.g. CommonEval, AlpacaEval_Full, WildVoice each count as 1, and SDQA_avg counts as 1).

**DO NOT include AlpacaEval-Speaker in the average** — it is not in the tracker formula.

Also show AlpacaEval-Speaker separately in the sub-score chart, but exclude from average computation.

## BFCL priority score

`(Simple + Multiple + Irrelevance) / 3` — the three categories that matter most for production. Parallel and Parallel-Multiple are consistently near 0 across all experiments (systemic gap, not a differentiator).

## FDB v1 Conversational Dynamics — tracker-aligned summary metrics

**DO NOT use `fdb_v1.headline.greedy` as the primary FDB v1 metric.** It diverges significantly (up to 10 pts) from the tracker formula for early experiments, because `headline.greedy` includes backchannel TOR as a penalty but the tracker formula does not.

Instead, compute these three summary metrics from components:

| Metric | Formula | Source columns |
|---|---|---|
| **TOR Summary %** | `round(((1−pause_avg) + TT_TOR + Intr_TOR) / 3 × 100, 1)` | `pause_synthetic.tor_pct`, `pause_candor.tor_pct`, `turn_taking.tor_pct`, `interruption.tor_pct` |
| **Adherence Score** | GPT-4o interruption rating (/5) | `interruption.rating` |
| **Combined Latency** | `(TT_latency_ms + Intr_latency_ms) / 2` | `turn_taking.latency_ms`, `interruption.latency_ms` |

Where `pause_avg = (Synthetic_TOR% + Candor_TOR%) / 2 / 100` (convert to fraction before averaging).

```python
# TOR Summary
AT = 1.0 - (synth_pct/100 + cand_pct/100) / 2   # pause handling fraction
AU = tt_tor_pct / 100                              # turn-taking fraction
AW = intr_tor_pct / 100                            # interruption fraction
TOR_SUMM = round((AT + AU + AW) / 3 * 100, 1)    # back to %
```

Use FDBV1_TOR_SUMM (not FDBV1_HL) in Figure 2 headline chart, D12 table, and Top 3 cards.

## Metric directions

- **↑ better**: VoiceBench avg, BFCL accuracy, FDB v1 TOR Summary, FDB v1 Adherence, FDB v1.5 headline, FDB v3 headline, BBA, Conv F1, interruption TOR%, turn-taking TOR%, backchannel JSD, AdvBench
- **↓ better**: all latency_ms values, Combined Latency, backchannel TOR%, pause TOR% (Candor + Synthetic), Conv cutoff_rate
- **↑ better (FDB v1.5)**: C_RESUME ratios; **↓ better**: C_RESPOND ratios during background/TTO (model should stop, not respond)

## Report structure

Output: `report_<date>.html` (self-contained, Chart.js CDN only).

### Section 1 — Top 3 Recommended Checkpoints
Three cards (gold/silver/bronze) with `.top3-grid` CSS. Each card includes:
- Experiment label, subtitle describing its strengths
- Rows: BFCL Priority %, BFCL Simple/Multiple, BFCL Irrelevance, VoiceBench avg (tracker formula), **FDB v1 TOR % (tracker formula)**, FDB v1.5 headline, BBA headline, Conv F1, **Adherence Score (/5)**, **Combined Latency ms avg(TT,Intr) (↓)**
- An insight box (green) and a weakness/warning box (orange or red)
- Color-code values: `.best` (green), `.good` (blue), `.mid` (yellow), `.low` (red), `.na` (muted)

Select top 3 by holistic trade-off, not single-metric winner. Aim for diversity: if one ranks best on BFCL and another on FDB, include both.

### Section 2 — Key Cross-Cutting Findings
Two callout boxes highlighting systemic patterns (e.g. FDB v1 vs FDB v1.5 tension, BFCL Parallel gap).

### Figures 1–7
Each figure is a `<h2>` with one or two `.grid2` card rows.

| Figure | Content |
|---|---|
| Fig 1 | VB Normalized Average bar (tracker formula) + VB Sub-scores grouped bar (all 10 datasets incl. AlpacaEval Full and SDQA avg, all ckpts, `cw-xl` height) + **Combined Latency avg(TT,Intr) bar** + **Adherence Score bar** |
| Fig 2 | **TOR Summary % (tracker formula) bar** + **Adherence Score bar** + pause TOR grouped (Candor+Synthetic) + interruption grouped (TOR%+Latency÷10) + turn-taking grouped (TOR%+Latency÷10) |
| Fig 3 | FDB v1.5 headline bar + background speech grouped (C_RESPOND+C_RESUME) + backchannel C_RESUME bar + interruption C_RESPOND bar |
| Fig 4 | FDB v3 headline bar (note: sparse data — only a few checkpoints) |
| Fig 5 | BFCL Priority bar + Simple/Multiple grouped + Irrelevance bar + Parallel/Par-Multi grouped |
| Fig 6 | BBA headline grouped (all sub-tasks) + BBA sub-task variation grouped |
| Fig 7 | Conv F1 bar + precision/recall/cutoff grouped |

### Recommendations (R1–R7)
Numbered `.rec` boxes. Always include:
- Continue training top checkpoint(s) for more steps
- Run ablation to disentangle init checkpoint from LR (these are always confounded)
- Address BFCL Parallel/Parallel-Multiple with targeted data
- If any checkpoint has BFCL Irrelevance < 70%, flag it as a critical issue

### Detailed Section — Table of Contents + D1–D12

Start with a clickable TOC (`.toc` card with anchor links). Sections:

| ID | Content |
|---|---|
| D1 | VoiceBench: all 10 sub-scores as individual bar charts (5 pairs in `.grid2`) |
| D2 | FDB v1 backchannel: TOR% bar + JSD bar |
| D3 | FDB v1 pause handling: Candor TOR% bar + Synthetic TOR% bar |
| D4 | FDB v1 interruption: TOR% bar + grouped (Latency ms + GPT rating) |
| D5 | FDB v1 turn-taking: TOR% bar + Latency ms bar |
| D6 | FDB v1.5 background speech: C_RESPOND bar + C_RESUME bar + stop latency bar |
| D7 | FDB v1.5 talking-to-other: C_RESPOND bar + C_RESUME bar |
| D8 | FDB v1.5 backchannel C_RESUME bar + interruption C_RESPOND bar |
| D9 | BFCL: all 5 categories as individual bar charts |
| D10 | BBA: all 4 sub-tasks as individual bar charts |
| D11 | Conv behavior: precision bar + recall bar + cutoff rate bar |
| D12 | Full numeric summary table: one row per checkpoint, all headline scores, color-coded with `.best`/`.good`/`.mid`/`.low` |

Each section has a 1–2 sentence `.insight` or `.note` callout below the charts.

## JS chart helpers to embed in HTML

```js
const LABELS = [...]; // short checkpoint labels

function barColors(data) {
  // gold for max, silver for 2nd, bronze for 3rd; red for min; blue otherwise
  // handle null values; for latency/lower-is-better, invert ranking
  ...
}

function makeBar(id, label, data, opts={}) {
  // single-dataset bar chart; opts: yMin, yMax, color, yLabel, reverse
  ...
}

function makeGrouped(id, datasets, opts={}) {
  // multi-dataset grouped bar; uses LABELS; opts: yMin, yMax, yLabel, reverse
  ...
}

const DSC = (label, data, color) => ({
  label, data,
  backgroundColor: color + 'b3', borderColor: color,
  borderWidth: 1, borderRadius: 2
});
```

`barColors` highlights top-3 checkpoints in gold/silver/bronze and the minimum in red. For latency charts (lower=better), reverse the ranking.

## Style

Same dark-theme CSS as the tracker report, plus:

```css
.top3-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin:16px 0 }
.top3-card { border-radius:8px; padding:16px; position:relative }
.rank-badge { position:absolute; top:10px; right:12px; font-size:1.4em }
.top3-exp { font-size:1.2em; font-weight:700; font-family:monospace; margin-bottom:6px }
.top3-row { display:flex; justify-content:space-between; font-size:.82em; margin:3px 0 }
.toc { background:var(--surf); border:1px solid var(--bord); border-radius:8px; padding:16px; margin-bottom:20px }
.fignum { font-size:.72em; background:var(--surf2); padding:1px 6px; border-radius:3px; color:var(--muted); margin-left:6px }
```

Chart.js: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`

## Common pitfalls

- **Write Python to a `.py` file** — do not use bash heredocs for Python containing `!=`, f-strings with special chars, etc. (escaping errors).
- **Use `$TMPDIR`** for all temp files — `/tmp` is not writable in the sandbox.
- **Install openpyxl** to `$TMPDIR/pylibs` if not available: `pip install openpyxl --target=$TMPDIR/pylibs`
- **`data_only=True`** in `load_workbook` to read computed cell values instead of formulas.
- **Never zero-impute nulls** — missing evals should be excluded from averages.
- **FDB v1.5 headline anomalies**: if a checkpoint's headline is >2× the median, it likely has missing sub-metrics causing a normalization artifact — flag it explicitly.
- **FDB v3 sparsity**: typically only 2–3 checkpoints have FDB v3 data — display as a separate chart with an explicit warning not to draw strong conclusions.
