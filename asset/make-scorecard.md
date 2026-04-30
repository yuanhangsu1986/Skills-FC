---
description: Generate a single-page HTML scorecard aggregating results from all S2S FC eval benchmarks (VB non-MCQ, VB MCQ, FDB, BBA, BFCL, conv_behav). Usage: /make-scorecard [key=value ...] where keys are vb_nonmcq, vb_mcq, fdb, bba, bfcl, conv_behav, output
---

You are going to generate a comprehensive single-page HTML scorecard aggregating metrics from all S2S FC eval benchmarks.

## Arguments

Parse `$ARGUMENTS` as `key=value` tokens (whitespace-delimited). Supported keys:

| Key | Meaning |
|---|---|
| `vb_nonmcq` | Absolute path to VoiceBench non-MCQ output dir (contains `eval-results/voicebench.*/`) |
| `vb_mcq` | Absolute path to VoiceBench MCQ output dir |
| `fdb` | Absolute path to FDB output dir (contains `eval-results/fdb_v1.*/`) |
| `bba` | Absolute path to BBA output dir (contains `eval-results/bba_aggregate/`) |
| `bfcl` | Absolute path to BFCL output dir (contains `eval-results/{category}/`) |
| `conv_behav` | Absolute path to conv_behav output dir (contains `eval-results/metrics.json`) |
| `output` | Absolute path for the output HTML file (default: `./scorecard.html` in cwd) |

Any subset of benchmarks may be provided. Benchmarks with no path supplied are shown as "Not run" in the scorecard.

If no arguments are provided, ask the user to supply the benchmark output directories.

---

## Metrics locations

### VoiceBench non-MCQ (`vb_nonmcq`)
Subtests: `sd_qa`, `alpacaeval_full`, `alpacaeval`, `ifeval`, `advbench`, `commoneval`, `wildvoice`, `alpacaeval_speaker`

Per-subtest metrics file: `{vb_nonmcq}/eval-results/voicebench.{subtest}/metrics.json`
Key: `voicebench.{subtest}.greedy` → fields: `gpt` (%), `panda` (%), `gpt_asr` (%), `panda_asr` (%); some subtests may lack asr fields.

Headline: mean `gpt` accuracy across available subtests.

### VoiceBench MCQ (`vb_mcq`)
Subtests: `bbh`, `openbookqa`, `mmsu`

Same path pattern. Headline: mean `gpt` accuracy.

### FDB (`fdb`)
Dimensions: `backchannel`, `interruption`, `pause_candor`, `pause_synthetic`, `turn_taking`

Per-dimension: `{fdb}/eval-results/fdb_v1.{dim}/metrics.json`
Keys:
- `fdb_v1.backchannel.greedy` → `tor`, `jsd`, `frequency`
- `fdb_v1.interruption.greedy` → `turn` (TOR), `latency_ms`, `rating`
- `fdb_v1.pause_candor.greedy` → `turn` (TOR; lower is better)
- `fdb_v1.pause_synthetic.greedy` → `turn` (TOR; lower is better)
- `fdb_v1.turn_taking.greedy` → `turn` (TOR; higher is better), `latency_ms`

Headline: no single number — show a mini-table of TOR per dimension.

### BBA (`bba`)
`{bba}/eval-results/bba_aggregate/metrics.json`
Key: `bba.aggregate.greedy` → `accuracy` (%), `per_category`

Headline: aggregate accuracy %.

### BFCL (`bfcl`)
Categories: `simple`, `parallel`, `multiple`, `parallel_multiple`, `irrelevance`
Per-category: `{bfcl}/eval-results/{category}/metrics.json`
Key: `bfcl_fc.{category}.greedy` → `accuracy` (%), `num_samples`, `num_correct`

Headline: mean accuracy across present categories.

### conv_behav (`conv_behav`)
`{conv_behav}/eval-results/metrics.json`
Key: `conv_behav.greedy` → `tt_f1` (%), `barge_in_success_rate` (%), `bc_accuracy` (%), `cutoff_rate` (%)

Headline: TT-F1 %.

---

## Step 1 — Parse arguments and read all metrics

1. Parse `$ARGUMENTS` into a dict (split on `=`, first `=` only).
2. Set `OUTPUT_PATH` from `output` key, defaulting to `./scorecard.html`.
3. For each provided benchmark directory, read the relevant metrics files using the Read tool. For missing files, set values to `None`.
4. Build a data summary dict in your context — you will pass this into the Python script as embedded literals.

---

## Step 2 — Write the generator script

Write `/tmp/make_scorecard.py` with all metric values embedded as Python literals (no file I/O in the script beyond writing the HTML). This avoids path issues since the script runs from /tmp.

Structure:

```python
import json
from pathlib import Path
from datetime import date

OUTPUT_PATH = Path("OUTPUT_PATH")
TODAY = date.today().isoformat()

# ── Embedded metric data (filled in by Claude from Step 1 reads) ─────────────
VB_NONMCQ = {
    # subtest → {"gpt": %, "panda": %, "gpt_asr": %, "panda_asr": %} or None
    "sd_qa": None, "alpacaeval_full": None, "alpacaeval": None, "ifeval": None,
    "advbench": None, "commoneval": None, "wildvoice": None, "alpacaeval_speaker": None,
}
VB_MCQ = {"bbh": None, "openbookqa": None, "mmsu": None}

FDB = {
    # dim → dict of metrics or None
    "backchannel":    None,
    "interruption":   None,
    "pause_candor":   None,
    "pause_synthetic": None,
    "turn_taking":    None,
}
BBA = {
    # overall accuracy and per_category dict, or None if not run
    "accuracy": None, "per_category": None,
}
BFCL = {
    # category → {"accuracy": %, "num_samples": N, "num_correct": N} or None
    "simple": None, "parallel": None, "multiple": None,
    "parallel_multiple": None, "irrelevance": None,
}
CONV_BEHAV = {
    # all keys or None
    "tt_f1": None, "tt_precision": None, "tt_recall": None, "tt_latency_ms": None,
    "barge_in_success_rate": None, "barge_in_latency_ms": None,
    "bc_accuracy": None, "cutoff_rate": None,
    "user_eou_f1": None, "num_evaluated": None,
}

# ── Report-level paths (for linking to per-benchmark reports) ────────────────
VB_NONMCQ_REPORT = None   # path to per-benchmark report.html, or None
VB_MCQ_REPORT    = None
FDB_REPORT       = None
BBA_REPORT       = None
BFCL_REPORT      = None
CONV_BEHAV_REPORT = None
```

After writing the literals, the script computes derived values:

```python
def mean_not_none(d):
    vals = [v for v in d.values() if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None

def color(v, thresholds, invert=False):
    """thresholds = (green_threshold, yellow_threshold); invert = lower is better."""
    if v is None: return '#8b949e'
    g, y = thresholds
    if not invert:
        return '#3fb950' if v >= g else ('#d29922' if v >= y else '#f85149')
    else:
        return '#3fb950' if v <= g else ('#d29922' if v <= y else '#f85149')

vb_nonmcq_headline = mean_not_none({k: v['gpt'] if v else None for k, v in VB_NONMCQ.items()})
vb_mcq_headline    = mean_not_none({k: v['gpt'] if v else None for k, v in VB_MCQ.items()})
bba_headline       = BBA.get('accuracy')
bfcl_headline      = mean_not_none({k: v['accuracy'] if v else None for k, v in BFCL.items()})
conv_tt_f1         = CONV_BEHAV.get('tt_f1')

fdb_interruption_tor = FDB['interruption']['turn'] * 100 if FDB.get('interruption') else None
fdb_pause_c_tor      = FDB['pause_candor']['turn'] * 100 if FDB.get('pause_candor') else None
fdb_pause_s_tor      = FDB['pause_synthetic']['turn'] * 100 if FDB.get('pause_synthetic') else None
fdb_tt_tor           = FDB['turn_taking']['turn'] * 100 if FDB.get('turn_taking') else None
```

Then generate the full HTML string and write it to `OUTPUT_PATH`.

---

## Step 3 — HTML scorecard specification

Single self-contained file. Dark GitHub theme, Chart.js from CDN.

Same CSS variables as all other analysis skills. Additional:
```css
.scorecard-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:16px 0; }
.bench-card { background:var(--bg2); border:1px solid var(--bd); border-radius:10px; padding:20px; }
.bench-card h3 { font-size:1em; font-weight:600; margin-bottom:14px; color:var(--ac); }
.metric-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid var(--bg3); font-size:0.88em; }
.metric-row:last-child { border-bottom:none; }
.metric-val { font-weight:700; font-size:1.05em; }
.not-run { color:var(--tx2); font-style:italic; font-size:0.85em; }
@media(max-width:900px) { .scorecard-grid { grid-template-columns:1fr; } }
```

### Page structure

**Header**: "S2S FC Eval Scorecard", date `TODAY`, short subtitle "Aggregated benchmark results across all evaluation suites."

**Top summary row** (6 tiles, one per benchmark, flex-wrap):
- Each tile: benchmark name, headline number with color coding, subtitle (what the number represents)
- Color thresholds:
  - VB non-MCQ GPT avg: green ≥60%, yellow ≥40%, red <40%
  - VB MCQ GPT avg: same thresholds
  - BBA accuracy: green ≥70%, yellow ≥50%, red <50%
  - BFCL accuracy avg: green ≥70%, yellow ≥40%, red <40%
  - conv_behav TT-F1: green ≥75%, yellow ≥55%, red <55%
  - FDB: no single headline — show "See details"
- "Not run" tile (grey) if benchmark was not provided

**Radar overview chart** (center, 600px wide): 6 axes (one per benchmark, normalized 0–100):
- VB non-MCQ: headline gpt avg / 100
- VB MCQ: headline gpt avg / 100
- BBA: accuracy / 100
- BFCL: headline avg / 100
- conv_behav: tt_f1 / 100
- FDB: composite = mean(interruption TOR, turn_taking TOR, (1 - pause_candor TOR), (1 - pause_synthetic TOR)) * 100

Show radar only if ≥ 3 benchmarks have data.

**Scorecard grid** (`scorecard-grid`, 3 columns, 2 rows = 6 benchmark cards):

**VoiceBench non-MCQ card**:
- Metric rows: per-subtest GPT accuracy (sorted descending)
- Footer: mean GPT %, mean GPT-ASR %
- Link to `VB_NONMCQ_REPORT` if available

**VoiceBench MCQ card**:
- Metric rows: per-subtest GPT accuracy
- Footer: mean GPT %

**FDB card**:
- Metric rows (one per dimension):
  - Interruption: TOR {v}% ↑ · Latency {latency_ms}ms · Rating {rating}/5
  - Pause Candor: TOR {v}% ↓ (lower is better)
  - Pause Synthetic: TOR {v}% ↓
  - Turn-Taking: TOR {v}% ↑ · Latency {latency_ms}ms
  - Backchannel: TOR {v}% · JSD {jsd}
- Color each TOR value by its direction (interruption/turn_taking: higher=better; pause/backchannel: lower=better)

**BBA card**:
- Metric rows: per-category accuracy (sorted descending)
- Footer: aggregate accuracy %

**BFCL card**:
- Metric rows: per-category accuracy (sorted descending)
- Footer: mean accuracy %

**conv_behav card**:
- Metric rows: TT-F1 %, TT Precision %, TT Recall %, TT Latency ms, Barge-In Rate %, Barge-In Latency ms, BC Accuracy %, Cutoff Rate % (lower is better), User-EOU F1 % (if present)
- Footer: N evaluated

**Comparison bar chart** (below the grid): one grouped bar per benchmark showing headline number vs a "target" reference (use: VB=60, BBA=70, BFCL=70, conv_behav-TT=80; FDB not included since no single number). Green fill for ≥target, red for <target.

**Conclusions section**: two-column grid:
- Left "Strongest Areas": top 3 benchmark scores with color badges
- Right "Areas for Improvement": bottom 3 with color badges and one-line suggestion each

---

## Step 4 — Run and report

Run `python3 /tmp/make_scorecard.py`.

Report:
1. Path to `scorecard.html`
2. Summary table of all headline numbers
3. Which benchmarks were missing / not run

---

## Notes

- All metric values are embedded as Python literals in Step 2 — the script does zero file I/O except writing the output HTML. This ensures it works regardless of where the script is run.
- If a benchmark directory was provided but `metrics.json` is missing, embed `None` for that benchmark's values and show "Results incomplete" in the card.
- Normalize all VoiceBench `gpt` values carefully: they are already percentages (0–100), not fractions.
- FDB `turn` values are fractions (0–1); multiply by 100 for display.
- For per-benchmark report links: if a `report.html` exists in the benchmark's output dir, set the corresponding `*_REPORT` variable to the absolute path; otherwise `None`. Card footers include "→ Detailed report" link if the path is set (use `file://` protocol for local browsing).
- "Not run" benchmarks: show a grey tile and a grey card with "—" values; do not omit them from the layout so the scorecard always has a consistent 6-benchmark structure.
