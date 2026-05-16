---
description: Analyze a VoiceBench OpenBookQA evaluation directory and produce an HTML report with accuracy breakdown, error category analysis, and 15 example cards. Usage: /vb-analyze-openbookqa <eval-dir> [benchmark-name]
---

You are going to analyze a VoiceBench OpenBookQA evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output_asr.jsonl`, `voicebench_format.jsonl`, `metrics.json`, `audio/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory before proceeding.

---

## Data schema

The OpenBookQA VoiceBench eval directory contains:

| File | Contents |
|---|---|
| `output_asr.jsonl` | One JSON object per example: `problem` (GT question text with MCQ choices A–D), `expected_answer` (reference: "A"/"B"/"C"/"D"), `audio_path` (`data/openbookqa_N.wav`, index N), `generation_text` (raw model text with timing tokens), `generation` (ASR of output, lowercase), `audio.path` (stale absolute path to output wav), `debug_info.agent_audio_asr`, `debug_info.agent_audio_wer` |
| `voicebench_format.jsonl` | Aligned line-for-line with `output_asr.jsonl`: `prompt`, `response` (lowercase ASR), `reference` (expected answer "A"/"B"/"C"/"D") |
| `metrics.json` | Aggregate metrics under key `"voicebench.openbookqa"` → `"greedy"`: `acc`, `fail`, `acc_asr`, `fail_asr`, `agent_wer`, `agent_cer` |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav`; use `Path(audio.path).name` and look up under `EVAL_DIR/audio/` |

**Input audio**: stored in tar archives at
`/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/openbookqa/recording.*.tar`
(10 tars). Files inside are named `N-N.wav` where N is the integer index from `audio_path` (`data/openbookqa_N.wav` → key N → member `N-N.wav`).

---

## Step 1 — Verify paths and inspect data

Use the Read tool to confirm `EVAL_DIR/output_asr.jsonl` and `EVAL_DIR/voicebench_format.jsonl` exist (read first 2 lines of each). Read `metrics.json` in full. Assert both JSONL files have the same line count.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_openbookqa.py`. **Run as `python3 /tmp/analyze_openbookqa.py`** — all paths must be absolute.

### Script structure

```python
import json, re, tarfile, shutil
from pathlib import Path

BASE_DIR        = Path("EVAL_DIR")
TAR_DIR         = Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/openbookqa")
ASR_JSONL       = BASE_DIR / "output_asr.jsonl"
VB_JSONL        = BASE_DIR / "voicebench_format.jsonl"
METRICS_JSON    = BASE_DIR / "metrics.json"
AUDIO_DIR       = BASE_DIR / "audio"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
REPORT_HTML     = BASE_DIR / "report.html"
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
```

**Token cleaning**: strip `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` from generation_text.

**Answer extraction** (from ASR response, lowercase):
```python
def extract_answer_mcq(response: str, reference: str) -> tuple[str, bool, bool]:
    """Returns (predicted, correct, extracted)."""
    resp = response.lower().strip()
    ref = reference.lower().strip()  # 'a', 'b', 'c', or 'd'

    # "the (correct) answer is X" pattern
    m = re.search(r'the (?:correct )?answer is[:\s]+([a-d])', resp)
    if m:
        pred = m.group(1)
        return pred, pred == ref, True

    # "answer is X" or "answer: X"
    m = re.search(r'answer[:\s]+([a-d])\b', resp)
    if m:
        pred = m.group(1)
        return pred, pred == ref, True

    # Terminal letter
    m = re.search(r'\b([a-d])\b\W*$', resp)
    if m:
        pred = m.group(1)
        return pred, pred == ref, True

    return 'unknown', False, False
```

**Answer letter distribution**: track predicted answers (A/B/C/D) vs expected answers to detect bias.

**Error categories** (checked in order; first match wins):

| Category | Condition |
|---|---|
| `empty_response` | `word_count < 5` |
| `high_wer` | `wer > 0.30` |
| `answer_fail` | `extracted == False` (could not extract answer) |
| `wrong_answer` | `correct == False` (extracted but wrong) |
| `correct` | `correct == True` |

**Input audio extraction**:
- Key N from `audio_path` via `re.search(r'openbookqa_(\d+)\.wav', ap)`
- Tar member name: `f"{N}-{N}.wav"`
- Scan all 10 tars once, build `key_to_tar: dict[int, (tar_path, member_name)]`
- Extract to `INPUT_AUDIO_DIR/{N}.wav`

**Output audio resolution**: `Path(entry["audio"]["path"]).name` → look up under `AUDIO_DIR/`

**Answer distribution analysis**: count occurrences of each predicted letter (A/B/C/D/unknown) and each expected letter to detect systematic bias.

**Select 15 diverse error examples** (budget: wrong_answer→9, answer_fail→3, empty_response→2, high_wer→2; include examples from each wrong letter prediction; top up from remaining errors if < 15).

---

## Step 3 — HTML report specification

Single self-contained HTML file, dark GitHub-style theme, Chart.js from CDN.

### CSS variables
```css
:root {
  --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
  --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff;
  --gr:#3fb950; --rd:#f85149; --or:#e3633c; --yl:#d29922;
}
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
```

Chart.js CDN: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`

### Sections

**Header**: benchmark name, model name (inferred from eval directory path), eval date. Chips: total examples, accuracy (text), accuracy (ASR), fail rate, WER.

**Metric tiles** (color-coded): overall accuracy (text), accuracy (ASR), fail rate, WER, total correct / total wrong / fail count.

**Four charts** (two rows of two):
1. Doughnut: correct / wrong_answer / answer_fail / empty / high_wer breakdown
2. Grouped bar: predicted answer distribution vs expected distribution (A/B/C/D) — shows letter bias
3. Bar: response word count distribution (bins: 0–4, 5–24, 25–49, 50–99, 100–199, 200+)
4. Bar: confusion matrix — for each expected letter, how often each letter was predicted (grouped bar with expected A/B/C/D as groups)

**Error category summary cards**: one card per non-correct category with count, name, and explanation paragraph.

**15 error example cards**: each card shows:
- Header row: `#N`, index, category badge (colored by error type), expected answer badge (green), predicted answer badge (red/orange), word count, WER
- Left column: question text (blue left-border), `<audio>` for input
- Right column (stacked):
  - **Text Output** (cyan `#56d4dd` left-border, label "Text Output"): `generation_text` cleaned, monospace, scrollable max-height 100px
  - **Speech ASR** (green left-border, label "Speech ASR"): `debug_info.agent_audio_asr` transcript, monospace, scrollable max-height 80px, followed by `<audio>` for output
- Analysis box: predicted vs expected answer, error explanation

Audio tags:
```html
<audio controls src="input_audio/{N}.wav" class="aplayer"></audio>
<audio controls src="audio/{filename}.wav" class="aplayer"></audio>
```
Fall back to italic text if file missing. **When checking file existence in the script, always check `(BASE_DIR / relative_path).exists()` — never `Path(relative_path).exists()`, which resolves relative to the script's cwd (`/tmp`) and will always be False.**

**Chart height**: set `.chart-box { height: 280px; }` in the CSS so Chart.js (with `maintainAspectRatio: false`) renders at a fixed height rather than expanding unconstrained.

**Conclusions & recommendations**: two-column grid — Key Findings and Recommendations — with bullet lists drawn from accuracy metrics, fail rate, answer distribution bias, and error breakdown.

---

## Step 4 — Run the script

Run `python3 /tmp/analyze_openbookqa.py`.

Report on completion:
1. Path to `report.html`
2. Top findings (overall acc text vs ASR gap, fail rate, answer distribution bias, dominant error category)
3. Any missing audio files

---

## Step 5 — Notes

- All output audio files in `EVAL_DIR/audio/` are padded to exactly 40 s.
- `voicebench_format.jsonl` and `output_asr.jsonl` are line-aligned (same order, same count).
- OpenBookQA has both `acc` and `fail` metrics — `fail` is the rate where the scorer could not extract an answer from the text response.
- OpenBookQA has 455 examples with 4-choice MCQ (A/B/C/D) on elementary science questions.
- Unlike BBH (which has 4 tasks) or MMSU (12 subjects), OpenBookQA is a single homogeneous benchmark — focus the analysis on error patterns, word count, and answer extraction reliability rather than category breakdown.
- The confusion matrix (expected vs predicted letter) is particularly useful for OpenBookQA to detect if the model systematically confuses certain letter choices (e.g., due to ASR artifacts).
