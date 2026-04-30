---
description: Analyze a VoiceBench SD-QA evaluation directory and produce an HTML report with per-accent accuracy breakdown, failure category analysis, and 15 playable example cards. Usage: /vb-analyze-sdqa <eval-dir> [benchmark-name]
---

You are going to analyze a VoiceBench SD-QA evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output_asr.jsonl`, `result-voicebench_format.jsonl`, `metrics.json`, `audio/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory before proceeding.

---

## Data schema

The SD-QA VoiceBench eval directory contains:

| File | Contents |
|---|---|
| `output_asr.jsonl` | One JSON object per example: `problem` (GT question text), `expected_answer` (reference answer string), `subset_for_metrics` (accent code, e.g. `"aus"`), `generation_text` (model output with timing tokens), `generation` (ASR of output, lowercase), `audio.path` (stale absolute path to output wav), `audio_path` (`data/sd_qa_{subset}_{N}.wav`, used to derive subset and index N), `debug_info.agent_audio_asr` (ASR transcript of output audio), `debug_info.agent_audio_wer` (WER) |
| `result-voicebench_format.jsonl` | Aligned line-for-line with `output_asr.jsonl`: `prompt`, `response` (lowercase ASR), `reference` (expected answer), `subset_for_metrics`, `score` (array of 3 strings, each `"Yes"` or `"No"`) |
| `metrics.json` | Aggregate metrics under key `"voicebench.sd_qa"` → `"greedy"`: `panda`, `gpt`, `panda_asr`, `gpt_asr` (accuracy percentages), `agent_wer`, `agent_cer`, `agent_ref_words` |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav`; `audio.path` in JSONL is a stale absolute path from a different machine — always use `Path(audio.path).name` and look up under `EVAL_DIR/audio/` |

**Input audio (question recordings)**: stored in tar archives at
`/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/sdqa/recording.*.tar`
(10 tars). Files inside are named `{subset}_{N}.wav` where subset and N come from `audio_path` (`data/sd_qa_{subset}_{N}.wav` → subset string and integer N).

**Accent subsets** (11 total, 553 examples each): `aus`, `gbr`, `ind_n`, `ind_s`, `irl`, `kenya`, `nga`, `nzl`, `phl`, `usa`, `zaf`

---

## Step 1 — Verify paths and inspect data

Use the Read tool to confirm `EVAL_DIR/output_asr.jsonl` and `EVAL_DIR/result-voicebench_format.jsonl` exist (read first 2 lines of each). Read `metrics.json` in full. Assert both JSONL files have the same line count.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_sdqa.py`. **Run as `cd /tmp && python3 analyze_sdqa.py`** — all paths must be absolute.

### Script structure

```python
BASE_DIR        = Path("EVAL_DIR")
TAR_DIR         = Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/sdqa")
ASR_JSONL       = BASE_DIR / "output_asr.jsonl"
RESULT_JSONL    = BASE_DIR / "result-voicebench_format.jsonl"
METRICS_JSON    = BASE_DIR / "metrics.json"
AUDIO_DIR       = BASE_DIR / "audio"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
REPORT_HTML     = BASE_DIR / "report.html"
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
```

**Token cleaning**: strip `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` from generation text.

**Correctness scoring**:
```python
yes_count = sum(1 for s in result_entry["score"] if s == "Yes")
majority_correct = yes_count >= 2
score_label = "correct" if majority_correct else "incorrect"
```

**Failure categories** (checked in order; first match wins; applied to incorrect examples only):

| Category | Condition |
|---|---|
| `empty_response` | `word_count < 3` |
| `ai_refusal` | text contains `"i don't know"`, `"i'm not sure"`, `"i cannot"`, `"as an ai"`, `"i do not know"`, `"i am not sure"` |
| `high_wer` | `wer > 0.30` |
| `verbose_wrong` | `word_count > 50` (model gave a long answer but still wrong) |
| `uncertain` | text contains `"i think"`, `"i believe"`, `"i'm not certain"`, `"i'm not 100%"`, `"might be"`, `"could be"`, `"probably"` |
| `wrong_answer` | none of the above (clean factual error) |

For correct examples, set category to `"correct"`.

**Input audio extraction**:
- Parse subset and global index N from `audio_path` field via `re.search(r'sd_qa_(\w+)_(\d+)\.wav', ap)`
- **Important**: `audio_path` uses global indices (aus=0–552, gbr=553–1105, ind_n=1106–1658, ind_s=1659–2211, irl=2212–2764, kenya=2765–3317, nga=3318–3870, nzl=3871–4423, phl=4424–4976, usa=4977–5529, zaf=5530–6082), but tar member names use per-subset indices 0–552. Convert with `SUBSET_OFFSETS = {'aus':0,'gbr':553,'ind_n':1106,'ind_s':1659,'irl':2212,'kenya':2765,'nga':3318,'nzl':3871,'phl':4424,'usa':4977,'zaf':5530}`; `local_idx = global_idx - SUBSET_OFFSETS[subset]`
- Tar member name: `f"{subset}_{local_idx}.wav"`
- Scan all 10 tars once, build `key_to_tar: dict[str, (tar_path, member_name)]` keyed by `f"{subset}_{local_idx}"`
- Extract to `INPUT_AUDIO_DIR/{subset}_{local_idx}.wav`

**Output audio resolution**:
- `Path(entry["audio"]["path"]).name` → look up under `AUDIO_DIR/`

**Select 15 diverse failure examples** (budget per category: `wrong_answer`→5, `verbose_wrong`→3, `uncertain`→2, `ai_refusal`→2, `high_wer`→2, `empty_response`→1); top up from remaining failures if < 15. Prefer examples spread across different accent subsets.

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

**Header**: benchmark name, model name, eval date. Chips: total examples, GPT accuracy (text), GPT accuracy (ASR), WER.

**Metric tiles** (color-coded): overall GPT accuracy %, overall Panda accuracy %, GPT-ASR accuracy %, WER, total correct / total examples.

**Three charts** (side by side):
1. Horizontal bar: per-accent accuracy (11 accents, sorted descending), colored green→red by accuracy value
2. Doughnut: failure category breakdown (failures only)
3. Bar: response word count distribution (bins: 0–2, 3–9, 10–19, 20–49, 50–99, 100+)

**Failure category summary cards**: one card per failure category (exclude `"correct"`) with count, name, and explanation paragraph.

**15 failure example cards**: each card shows:
- Header row: `#N`, index, accent badge (colored by subset), category badge (colored), GPT scores badge (`yes_count/3`), word count, WER
- Left column: question (GT text, blue left-border), reference answer (orange left-border, smaller text), `<audio>` for input
- Right column (stacked):
  - **Text Output** (cyan `#56d4dd` left-border, label "Text Output"): `generation_text` cleaned of timing tokens, monospace, scrollable max-height 100px
  - **Speech ASR** (red left-border if wrong / green if correct, label "Speech ASR"): `debug_info.agent_audio_asr` lowercase transcript, monospace, scrollable max-height 100px, followed by `<audio>` for output
- Analysis box: why this answer is wrong or in the failure category

Audio tags:
```html
<audio controls src="input_audio/{subset}_{N}.wav" class="aplayer"></audio>
<audio controls src="audio/{filename}.wav" class="aplayer"></audio>
```
Fall back to italic text if file missing.

**Per-accent accuracy table**: table with columns Accent, N, Correct, Accuracy %, sorted by accuracy ascending. Color the accuracy cell green (≥60%), yellow (40–59%), or red (<40%).

**Conclusions & recommendations**: two-column grid — Key Findings and Recommendations — with bullet lists drawn from the actual per-accent accuracy, overall metrics, and top failure categories.

---

## Step 4 — Run the script or instruct the user

Attempt `python3 /tmp/analyze_sdqa.py`. If Bash fails due to cwd errors, instruct user to run `cd /tmp && python3 analyze_sdqa.py`.

Report on completion:
1. Path to `report.html`
2. Top findings (overall accuracy, best/worst accent, top failure categories, WER, text vs ASR accuracy gap)
3. Any missing audio files

---

## Step 5 — Notes for reuse

- All audio output files in `EVAL_DIR/audio/` are padded to exactly 40 s (1,764,044 bytes at 22 050 Hz mono 16-bit). Do NOT use segment-end-time as a truncation signal.
- The `result-voicebench_format.jsonl` and `output_asr.jsonl` are line-aligned (same order, same count).
- Input audio in tars uses per-subset indices: `{subset}_{local_idx}.wav` (e.g. `aus_0.wav`, `gbr_0.wav`, always 0–552).
- **Critical**: `audio_path` uses GLOBAL indices (gbr starts at 553, not 0). Always convert to local index before looking up in tar: `local_idx = global_idx - SUBSET_OFFSETS[subset]`.
- Scores are `"Yes"` / `"No"` strings (not integers). A majority of 2 or more "Yes" votes = correct.
- Metrics live under `metrics["voicebench.sd_qa"]["greedy"]`. `panda` and `gpt` are accuracy percentages (0–100).
- The 11 accent subsets each have exactly 553 examples (global idx offsets: aus=0, gbr=553, ind_n=1106, ind_s=1659, irl=2212, kenya=2765, nga=3318, nzl=3871, phl=4424, usa=4977, zaf=5530).
- `audio_path` format: `data/sd_qa_{subset}_{global_N}.wav` — parse global index with `re.search(r'sd_qa_(\w+)_(\d+)\.wav', ap)`.
