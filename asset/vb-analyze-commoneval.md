---
description: Analyze a VoiceBench Commoneval evaluation directory and produce an HTML report with GPT score distribution, failure category breakdown, and 15 playable example cards. Usage: /vb-analyze-commoneval <eval-dir> [benchmark-name]
---

You are going to analyze a VoiceBench Commoneval evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output_asr.jsonl`, `result-voicebench_format.jsonl`, `metrics.json`, `audio/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory before proceeding.

---

## Data schema

The Commoneval VoiceBench eval directory contains:

| File | Contents |
|---|---|
| `output_asr.jsonl` | One JSON object per example: `problem` (GT question text), `generation_text` (model text output WITH timing tokens — strip before display, preserves casing), `generation` (clean lowercase ASR of spoken output — identical to `debug_info.agent_audio_asr`), `audio.path` (stale absolute path to output wav), `audio_path` (`data/commoneval_N.wav`, used to derive index N), `debug_info.agent_audio_asr` (clean lowercase ASR of output audio), `debug_info.agent_audio_wer` (WER), `debug_info.agent_audio_cer` (CER), `debug_info.agent_audio_segments_sec` |
| `result-voicebench_format.jsonl` | Aligned line-for-line with `output_asr.jsonl`: `prompt`, `response` (lowercase ASR), `reference` (empty string for Commoneval), `score` (array of 3 GPT score strings, e.g. `["2","3","2"]`) |
| `metrics.json` | Aggregate metrics under key `"voicebench.commoneval"` → `"greedy"`: `gpt`, `gpt_asr`, `agent_wer`, `agent_cer`, `agent_word_deletions`, `agent_ref_words` |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav`; `audio.path` in JSONL is a stale absolute path from a different machine — always use `Path(audio.path).name` and look up under `EVAL_DIR/audio/` |

**Input audio (question recordings)**: stored in tar archives at
`/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/commoneval/recording.*.tar`
(10 tars). Files inside are named `N-N.wav` where N is the integer index derived from `audio_path` (`data/commoneval_N.wav` → key N).

---

## Step 1 — Verify paths and inspect data

Use the Read tool to confirm `EVAL_DIR/output_asr.jsonl` and `EVAL_DIR/result-voicebench_format.jsonl` exist (read first 2 lines of each). Read `metrics.json` in full. Assert both JSONL files have the same line count (should be 200).

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_commoneval.py`. **Run as `cd /tmp && python3 analyze_commoneval.py`** — all paths must be absolute.

### Script structure

```python
BASE_DIR        = Path("EVAL_DIR")
TAR_DIR         = Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/commoneval")
ASR_JSONL       = BASE_DIR / "output_asr.jsonl"
RESULT_JSONL    = BASE_DIR / "result-voicebench_format.jsonl"
METRICS_JSON    = BASE_DIR / "metrics.json"
AUDIO_DIR       = BASE_DIR / "audio"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
REPORT_HTML     = BASE_DIR / "report.html"
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
```

**Token cleaning**: strip `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` from the `generation_text` field to get the model's clean text output for display (preserves original casing).

**Clean ASR text**: use `debug_info.agent_audio_asr` (equivalently `generation`) as the canonical lowercase ASR transcript for failure analysis and word-count computation. Both text output and ASR are displayed in the report — they should be very similar but differ in casing, minor transcription errors, and dropped words.

**Truncation detection** (important — do NOT use segment end time, all audio is padded to 40 s):
```python
ends_sentence = bool(re.search(r'[.?!"'"]$', gen_text.strip())) if gen_text else False
truncated = (not ends_sentence) and word_count > 50
```

**Failure categories** (checked in order; first match wins; use the clean ASR text from `debug_info.agent_audio_asr` for keyword checks):

| Category | Condition |
|---|---|
| `empty_response` | `word_count < 5` |
| `ai_refusal` | text contains `"as an ai language model"`, `"i'm sorry, but as an ai"`, `"i am unable to"`, `"i do not have the capability"`, etc. |
| `clarification_deflection` | text contains `"could you please provide"`, `"could you provide more"`, `"i need more information"`, `"please provide more details"`, etc. |
| `mid_sentence_cutoff` | `truncated` is True (no ending punctuation AND > 50 words) |
| `template_placeholder` | text matches `\[(?:your\|student\|sample\|insert\|placeholder\|customize)[^\]]*\]` |
| `high_wer` | `wer > 0.30` |
| `insufficient_depth` | long-form task (prompt contains write/compose/draft/structure/create/develop) AND `word_count < 50`; OR `word_count < 30` regardless |
| `acceptable` | none of the above |

**WER value**: read from `debug_info.agent_audio_wer` in `output_asr.jsonl`.

**GPT scores**: parse `result_entry["score"]` as list of ints → compute `avg_score` (mean) and `score_var` (variance).

**Input audio extraction**:
- Key N from `audio_path` field via `re.search(r'commoneval_(\d+)\.wav', ap)`
- Tar member name: `f"{N}-{N}.wav"`
- Scan all 10 tars once, build `key_to_tar: dict[int, (tar_path, member_name)]`
- Extract to `INPUT_AUDIO_DIR/{N}.wav`

**Output audio resolution**:
- `Path(entry["audio"]["path"]).name` → look up under `AUDIO_DIR/`

**Select 15 diverse failure examples** (budget per category: empty_response→2, mid_sentence_cutoff→3, ai_refusal→2, clarification_deflection→2, template_placeholder→2, insufficient_depth→3, high_wer→2); top up from remaining failures if < 15. Sort selected examples by index (ascending).

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

**Header**: benchmark name ("Commoneval"), model name (infer from directory path if possible, else "Unknown"), eval date. Chips: total examples, GPT score (text), GPT score (ASR), WER.

**Metric tiles** (color-coded): avg GPT score, WER, count by score bucket (1/2/3/4–5).

**Three charts** (side by side):
1. Bar: GPT score distribution (1–5), colors red→orange→yellow→green→blue
2. Doughnut: failure category breakdown (failures only)
3. Bar: response word count distribution (bins: 0–4, 5–24, 25–49, 50–99, 100–199, 200–499, 500+)

**Failure category summary cards**: one card per category with count, name, and explanation paragraph.

**15 failure example cards**: each card shows:
- Header row: `#N`, index, category badge (colored), `⚠ TRUNCATED` if applicable, GPT score badge (`avg X.X/5 (s1/s2/s3)`), word count, WER
- Top-left: input question (GT text from `problem` field, blue left-border), `<audio>` for input question
- Top-right: model text output (`generation_text` cleaned of timing tokens, teal/cyan left-border, monospace, scrollable) — labelled "Text Output"
- Bottom-right (below text output, same column): speech ASR (`debug_info.agent_audio_asr`, green left-border, monospace, scrollable) — labelled "Speech ASR", followed by `<audio>` for the output wav
- Analysis box: failure explanation paragraph

The card layout uses a 2-column grid: left = question, right = stacked text output then speech ASR + audio.

Audio tags:
```html
<audio controls src="input_audio/{N}.wav" class="aplayer"></audio>
<audio controls src="audio/{filename}.wav" class="aplayer"></audio>
```
Fall back to italic text if file missing.

**Conclusions & recommendations**: two-column grid — Key Findings and Recommendations — with bullet lists drawn from the actual category counts and metrics.

---

## Step 4 — Run the script or instruct the user

Attempt `python3 /tmp/analyze_commoneval.py`. If Bash fails due to cwd errors, instruct user to run `cd /tmp && python3 analyze_commoneval.py`.

Report on completion:
1. Path to `report.html`
2. Top findings (category counts, mean GPT score, WER, GPT text vs ASR gap)
3. Any missing audio files

---

## Step 5 — Notes for reuse

- All audio output files in `EVAL_DIR/audio/` are padded to exactly 40 s (1,764,044 bytes at 22 050 Hz mono 16-bit). Do NOT use segment-end-time as a truncation signal.
- The `result-voicebench_format.jsonl` and `output_asr.jsonl` are line-aligned (same order, same count — 200 lines each).
- Input audio in tars uses `N-N.wav` naming. The `cuts.*.jsonl.gz` files contain lhotse manifests but are not needed for this report.
- `generation_text` in `output_asr.jsonl` is the model's **text output** WITH timing tokens (`<$N.NN$>`, `<|N.NN|>`). Strip these before display. It preserves original casing and punctuation.
- `generation` in `output_asr.jsonl` is the clean lowercase **speech ASR** — identical to `debug_info.agent_audio_asr`. Use either for failure analysis and word-count computation.
- Both text output and speech ASR are shown in each example card; they should be very similar but may differ in casing, punctuation, and minor transcription errors.
- Metrics live under `metrics["voicebench.commoneval"]["greedy"]`.
- The `reference` field in `result-voicebench_format.jsonl` is always an empty string for Commoneval — do not use it for scoring or comparisons.
- Commoneval has 200 examples (shorter than alpacaeval_full's 640).
