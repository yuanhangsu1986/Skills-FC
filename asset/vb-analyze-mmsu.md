---
description: Analyze a VoiceBench MMSU evaluation directory and produce an HTML report with per-subject accuracy breakdown, error category analysis, and 15 example cards. Usage: /vb-analyze-mmsu <eval-dir> [benchmark-name]
---

You are going to analyze a VoiceBench MMSU (Massive Multi-Subject Understanding) evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output_asr.jsonl`, `voicebench_format.jsonl`, `metrics.json`, `audio/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory before proceeding.

---

## Data schema

The MMSU VoiceBench eval directory contains:

| File | Contents |
|---|---|
| `output_asr.jsonl` | One JSON object per example: `problem` (GT question text with MCQ choices), `expected_answer` (reference: "A"/"B"/"C"/"D"), `audio_path` (`data/mmsu_<subject>_N.wav`), `generation_text` (raw model text with timing tokens), `generation` (ASR of output, lowercase), `audio.path` (stale absolute path to output wav), `debug_info.agent_audio_asr`, `debug_info.agent_audio_wer` |
| `voicebench_format.jsonl` | Aligned line-for-line with `output_asr.jsonl`: `prompt`, `response` (lowercase ASR), `reference` (expected answer "A"/"B"/"C"/"D"), `subset_for_metrics` (subject name, e.g. `law`, `biology`) |
| `metrics.json` | Aggregate metrics under key `"voicebench.mmsu"` → `"greedy"`: `acc`, `fail`, `acc_asr`, `fail_asr`, `agent_wer`, `agent_cer` |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav`; use `Path(audio.path).name` and look up under `EVAL_DIR/audio/` |

**MMSU has 12 subject categories**: biology, business, chemistry, economics, engineering, health, history, law, other, philosophy, physics, psychology. Read from the `subset_for_metrics` field in voicebench_format.jsonl.

**Input audio**: stored in tar archives at
`/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/mmsu/recording.*.tar`
(10 tars). Files inside are named `{GLOBAL_ID}_{subject}.wav` (e.g. `886_law.wav`). The global ID does NOT match the per-subject index in `audio_path`. Use text matching against cuts manifests to resolve.

**Input audio key resolution** (MMSU-specific):
- Scan all `cuts.*.jsonl.gz` files in the TAR_DIR to build a mapping: `normalize(supervision_text) → (tar_path, member_name)`
- The supervision text (first supervision per cut whose speaker is "user") matches the `problem` field exactly after whitespace normalization
- For each entry, look up `normalize(entry["problem"])` to find the correct tar member
- `normalize(text)` = `" ".join(text.split())` (collapse all whitespace)

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_mmsu.py`. **Run as `python3 /tmp/analyze_mmsu.py`** — all paths must be absolute.

### Script structure

```python
import json, re, gzip, tarfile, shutil
from pathlib import Path

BASE_DIR        = Path("EVAL_DIR")
TAR_DIR         = Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/mmsu")
ASR_JSONL       = BASE_DIR / "output_asr.jsonl"
VB_JSONL        = BASE_DIR / "voicebench_format.jsonl"
METRICS_JSON    = BASE_DIR / "metrics.json"
AUDIO_DIR       = BASE_DIR / "audio"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
REPORT_HTML     = BASE_DIR / "report.html"
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
```

**Token cleaning**: strip `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` from generation_text.

**Build text→audio mapping** (run once at startup):
```python
def build_text_to_audio_map(tar_dir: Path) -> dict[str, tuple[Path, str]]:
    """Scan all cuts manifests; return normalize(supervision_text) -> (tar_path, member_name)."""
    text_map = {}
    for cuts_gz in sorted(tar_dir.glob("cuts.*.jsonl.gz")):
        with gzip.open(cuts_gz) as f:
            for line in f:
                cut = json.loads(line)
                cut_id = cut["id"]
                # Find user supervision text
                for sup in cut.get("supervisions", []):
                    if sup.get("speaker") == "user":
                        norm = " ".join(sup["text"].split())
                        text_map[norm] = cut_id
                        break
    # Build cut_id -> (tar_path, member_name) by scanning tars
    id_to_tar = {}
    for tar_path in sorted(tar_dir.glob("recording.*.tar")):
        with tarfile.open(tar_path) as t:
            for m in t.getmembers():
                if m.name.endswith(".wav"):
                    stem = m.name.replace(".wav", "")
                    id_to_tar[stem] = (tar_path, m.name)
    # Combine: norm_text -> (tar_path, member_name)
    result = {}
    for norm_text, cut_id in text_map.items():
        if cut_id in id_to_tar:
            result[norm_text] = id_to_tar[cut_id]
    return result
```

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

**Error categories** (checked in order; first match wins):

| Category | Condition |
|---|---|
| `empty_response` | `word_count < 5` |
| `high_wer` | `wer > 0.30` |
| `answer_fail` | `extracted == False` (could not extract answer) |
| `wrong_answer` | `correct == False` (extracted but wrong) |
| `correct` | `correct == True` |

**Input audio extraction**:
- Build text map once
- For each entry: `norm_text = " ".join(entry["problem"].split())`; look up in text map → `(tar_path, member_name)`
- Extract to `INPUT_AUDIO_DIR/{idx}.wav` (idx = line number)
- Track missing audio

**Output audio resolution**: `Path(entry["audio"]["path"]).name` → look up under `AUDIO_DIR/`

**Per-subject accuracy**: group by `subset_for_metrics`, compute accuracy and fail rate per subject.

**Select 15 diverse error examples** (budget: wrong_answer→8, answer_fail→4, empty_response→2, high_wer→2; spread across subjects when possible; top up from errors if < 15).

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

**Three charts** (side by side):
1. Horizontal bar: accuracy by subject (12 subjects), sorted descending, with % labels, colored green→red by accuracy
2. Doughnut: correct / wrong_answer / answer_fail / empty / high_wer breakdown
3. Bar: response word count distribution (bins: 0–4, 5–24, 25–49, 50–99, 100–199, 200+)

**Error category summary cards**: one card per non-correct category with count, name, and explanation paragraph.

**15 error example cards**: each card shows:
- Header row: `#N`, index, subject badge (colored by subject), category badge (colored by error type), expected answer badge, word count, WER
- Left column: question text (blue left-border), `<audio>` for input
- Right column (stacked):
  - **Text Output** (cyan `#56d4dd` left-border, label "Text Output"): `generation_text` cleaned, monospace, scrollable max-height 100px
  - **Speech ASR** (green left-border, label "Speech ASR"): `debug_info.agent_audio_asr` transcript, monospace, scrollable max-height 80px, followed by `<audio>` for output
- Analysis box: predicted answer vs expected answer, error explanation

Audio tags:
```html
<audio controls src="input_audio/{idx}.wav" class="aplayer"></audio>
<audio controls src="audio/{filename}.wav" class="aplayer"></audio>
```
Fall back to italic text if file missing. **When checking file existence in the script, always check `(BASE_DIR / relative_path).exists()` — never `Path(relative_path).exists()`, which resolves relative to the script's cwd (`/tmp`) and will always be False.**

**Chart height**: set `.chart-box { height: 280px; }` in the CSS so Chart.js (with `maintainAspectRatio: false`) renders at a fixed height rather than expanding unconstrained.

**Conclusions & recommendations**: two-column grid — Key Findings and Recommendations — with bullet lists drawn from per-subject accuracy, fail rate analysis, and error distribution.

---

## Step 4 — Run the script

Run `python3 /tmp/analyze_mmsu.py`.

Report on completion:
1. Path to `report.html`
2. Top findings (best/worst subjects, overall acc text vs ASR gap, fail rate, dominant error category)
3. Any missing audio files

---

## Step 5 — Notes

- All output audio files in `EVAL_DIR/audio/` are padded to exactly 40 s.
- `voicebench_format.jsonl` and `output_asr.jsonl` are line-aligned (same order, same count).
- MMSU has both `acc` and `fail` metrics — `fail` is the rate where the scorer could not extract an answer from the text response; `fail_asr` is the same for the ASR response.
- The `subset_for_metrics` field in voicebench_format.jsonl gives the subject for each example.
- Text matching for audio: the `problem` field in output_asr.jsonl matches the user supervision text in cuts manifests after whitespace normalization.
- The 12 subjects are: biology, business, chemistry, economics, engineering, health, history, law, other, philosophy, physics, psychology.
