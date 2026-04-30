---
description: Analyze a BBA (BigBench Audio) evaluation directory and produce an HTML report with per-category accuracy, run variance, failure analysis, and 8 example cards per category. Usage: /analyze-bba <output-dir> [benchmark-name]
---

You are going to analyze a BigBench Audio (BBA) evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `OUTPUT_DIR`**: absolute path to the BBA output directory (contains `eval-results/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable label; if omitted, infer from directory name

If no argument is provided, ask the user for the output directory.

---

## Data layout

```
{OUTPUT_DIR}/
└── eval-results/
    ├── {category}/           # formal_fallacies | navigate | object_counting | web_of_lies
    │   ├── output_asr.jsonl  # one JSON per sample: generation, expected_answer, question_asr, audio
    │   ├── metrics.json      # {"bba.{category}": {"greedy": {"accuracy", "run_accuracies", "total"}}}
    │   └── audio/            # model output WAV files
    └── bba_aggregate/
        └── metrics.json      # {"bba.aggregate": {"greedy": {"accuracy", "per_category"}}}
```

### output_asr.jsonl fields
| Field | Contents |
|---|---|
| `generation` | Whisper ASR of model's audio output |
| `expected_answer` | Ground-truth answer string |
| `question_asr` | Whisper ASR of input question audio |
| `audio` | `{"path": "<stale abs path>"}` — resolve filename via `Path(audio["path"]).name`, look up under `eval-results/{category}/audio/` |

---

## Step 1 — Verify and inspect

Confirm `OUTPUT_DIR/eval-results/` exists. For each of the 4 categories, read the first 2 lines of `output_asr.jsonl` and read `metrics.json` in full. Read `bba_aggregate/metrics.json` in full.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_bba.py`. Run as `python3 /tmp/analyze_bba.py`. All paths must be absolute.

```python
import json, re, sys
from pathlib import Path
from collections import Counter

OUTPUT_DIR  = Path("OUTPUT_DIR")
EVAL_DIR    = OUTPUT_DIR / "eval-results"
CATEGORIES  = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]
REPORT_HTML = OUTPUT_DIR / "report.html"
TOKEN_RE    = re.compile(r'<\$[\d.]+\$>|<\|[\d.]+\|>')

CAT_DESC = {
    "formal_fallacies":  "Tests whether the model identifies whether a logical argument is formally valid or invalid.",
    "navigate":          "Tests whether the model can determine the final position of an agent following navigation instructions.",
    "object_counting":   "Tests whether the model can count objects in a described scene based on audio.",
    "web_of_lies":       "Tests whether the model can determine truth values through a chain of truthful and deceptive speakers.",
}

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def load_json(p):
    with open(p) as f: return json.load(f)

def clean(text):
    return TOKEN_RE.sub('', text).strip()

def first_word(text):
    words = clean(text).lower().split()
    return re.sub(r'[^a-z0-9]', '', words[0]) if words else ''

def approx_correct(gen, exp):
    return first_word(gen) == first_word(exp)

def failure_category(gen, exp):
    c = clean(gen)
    if not c:
        return 'empty_response'
    wc = len(c.split())
    if wc > 20:
        return 'verbose'
    fw_gen, fw_exp = first_word(gen), first_word(exp)
    if fw_exp and (fw_gen.startswith(fw_exp[:3]) or fw_exp.startswith(fw_gen[:3])):
        return 'close_miss'
    return 'wrong_answer'

def audio_tag(abs_path, rel_src):
    return (f'<audio controls src="{rel_src}" class="aplayer"></audio>'
            if Path(abs_path).exists()
            else '<em style="color:var(--tx2)">audio unavailable</em>')
```

### Per-category processing

For each category:
1. Load `EVAL_DIR/{category}/output_asr.jsonl`
2. Load `metrics.json` → extract `accuracy`, `run_accuracies`, `total`
3. For each entry enrich with:
   - `correct = approx_correct(generation, expected_answer)`
   - `fail_cat = failure_category(gen, exp)` if not correct else `"correct"`
   - `word_count = len(clean(generation).split())`
   - `audio_abs = EVAL_DIR / category / "audio" / Path(entry["audio"]["path"]).name` (guard with `.get`)
   - `audio_rel = f"eval-results/{category}/audio/{filename}"`

### Example card selection (8 per category)

Select 4 correct (prefer shortest generation) and 4 incorrect (one per failure category if possible; prefer `empty_response`, `verbose`, `close_miss`, `wrong_answer` in that order; top-up from remaining incorrect if a bucket is empty).

---

## Step 3 — HTML report

Single file at `OUTPUT_DIR/report.html`. Dark GitHub theme, Chart.js from `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`.

```css
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d; --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950; --rd:#f85149; --or:#e3633c; --yl:#d29922; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--tx); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:14px; line-height:1.5; padding:24px; max-width:1400px; margin:0 auto; }
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
.chart-box { height:280px; background:var(--bg2); border:1px solid var(--bd); border-radius:8px; padding:16px; }
.tile { background:var(--bg2); border:1px solid var(--bd); border-radius:8px; padding:14px 18px; min-width:130px; flex:1; }
.card { background:var(--bg2); border:1px solid var(--bd); border-radius:8px; padding:16px; margin-bottom:14px; }
.card-header { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--bd); }
.block { background:var(--bg3); border-radius:6px; padding:10px 12px; }
.block-label { font-size:0.75em; font-weight:600; color:var(--tx2); text-transform:uppercase; letter-spacing:.06em; margin-bottom:5px; }
.scroll-pre { font-family:"SFMono-Regular",Consolas,monospace; font-size:0.8em; white-space:pre-wrap; word-break:break-word; overflow-y:auto; max-height:100px; color:var(--tx); }
.analysis-box { margin-top:10px; background:var(--bg3); border-radius:6px; padding:8px 12px; font-size:0.85em; color:var(--tx2); border-left:3px solid var(--yl); }
.dim-desc { color:var(--tx2); font-size:0.9em; margin-bottom:16px; border-left:3px solid var(--bd); padding-left:12px; }
```

### Structure

**Header**: "BigBench Audio — Analysis Report", model name (infer from `OUTPUT_DIR.name`), date.

**Nav**: links to each of the 4 category sections + `#conclusions`.

**Summary dashboard** (5 flex tiles — 4 categories + aggregate):
- Each tile shows category name, accuracy %, tile border-top color: green ≥70%, yellow 50–69%, red <50%
- Aggregate tile shows mean accuracy across categories

**Two overview charts** (side by side in `.charts-2col`):
1. Horizontal bar: per-category accuracy %, sorted descending, each bar colored by threshold (green/yellow/red)
2. Grouped bar: correct vs incorrect count per category (green and red datasets)

**Four `<section>` blocks** (one per category in CATEGORIES order):

```
<section id="{category}">
  <h2>§N — {Category Title}</h2>
  <p class="dim-desc">{CAT_DESC[category]}</p>
  <div class="tiles" style="display:flex;flex-wrap:wrap;gap:12px;margin:12px 0 20px">
    [accuracy tile] [run variance tile] [N samples tile]
  </div>
  <div class="charts-2col" style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0">
    [doughnut: correct/incorrect]
    [bar: word count histogram (bins: 0, 1–3, 4–10, 11–20, 20+)]
  </div>
  <h3>Example Cards (8: 4 correct + 4 incorrect)</h3>
  {8 cards}
</section>
<hr style="border:none;border-top:1px solid var(--bd);margin:40px 0">
```

**Example card** (per entry):
- Header: `#{N}` (blue), `idx {idx}` (dim), correct badge (green `CORRECT ✓` / red `INCORRECT ✗`), failure category badge if incorrect (yellow), `Words: {word_count}`
- Two-column body:
  - Left: question block (`var(--ac)` border) with `question_asr` text; expected answer block (`var(--or)` border)
  - Right: model response block (`#56d4dd` border) with `generation` text (cleaned); audio player
- Analysis box: if correct → "First word matches expected answer."; if incorrect → explain failure category with the actual vs expected first word

**Conclusions section** (`id="conclusions"`): two-column grid with Key Findings and Recommendations bullet lists.

### Charts JavaScript

Generate `new Chart(...)` calls for:
- Overview horizontal bar: categories on y-axis, accuracy on x-axis (0–100)
- Overview grouped bar: categories on x-axis, two datasets (Correct, Incorrect)
- Per-category doughnuts: `['Correct', 'Incorrect']`, `['#3fb950', '#f85149']`
- Per-category word-count bar histograms

Inject actual data values from the Python analysis; use `Chart.defaults.color = '#8b949e'; Chart.defaults.borderColor = '#30363d';` at the top of the `<script>` block.

---

## Step 4 — Run and report

Run `python3 /tmp/analyze_bba.py`.

Report:
1. Path to `report.html`
2. Per-category accuracy table (accuracy %, run variance, N)
3. Top failure categories across all categories
4. Any missing audio files

---

## Notes

- Output audio files: resolve via `Path(entry.get("audio", {}).get("path", "")).name`; check existence with absolute path `(EVAL_DIR / category / "audio" / filename).exists()` — NEVER use relative path existence checks since the script runs from /tmp.
- `generation` in `output_asr.jsonl` is the Whisper ASR of the model's speech output, NOT raw timing tokens. Timing tokens may still appear if the ASR stage preserved the original field; always strip them via `TOKEN_RE`.
- Run accuracy variance = `max(run_accuracies) - min(run_accuracies)`; if `run_accuracies` is missing, show "N/A".
- The approx_correct scoring is for display only; the official score is from LLM judge (shown in the accuracy tile from metrics.json).
