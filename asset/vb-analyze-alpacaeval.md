---
description: Analyze a VoiceBench AlpacaEval (or AlpacaEval Full) evaluation directory and produce an HTML report with GPT score distribution, failure category breakdown, and 15 playable example cards. Usage: /vb-analyze-alpacaeval <eval-dir> [benchmark-name]
---

You are going to analyze a VoiceBench AlpacaEval evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output_asr.jsonl`, `result-voicebench_format.jsonl`, `metrics.json`, `audio/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory before proceeding.

---

## Data schema

The AlpacaEval VoiceBench eval directory contains:

| File | Contents |
|---|---|
| `output_asr.jsonl` | One JSON object per example: `problem` (GT question text), `generation_text` (model output with timing tokens), `generation` (ASR of output, lowercase), `audio.path` (stale absolute path to output wav), `audio_path` (`data/alpacaeval_full_N.wav`, used to derive index N), `debug_info.agent_audio_asr` (ASR transcript of output audio), `debug_info.agent_audio_wer` (WER), `debug_info.agent_audio_segments_sec` |
| `result-voicebench_format.jsonl` | Aligned line-for-line with `output_asr.jsonl`: `prompt`, `response` (lowercase ASR), `score` (array of 3 GPT score strings, e.g. `["3","4","3"]`) |
| `metrics.json` | Aggregate metrics under key `"voicebench.alpacaeval_full"` → `"greedy"`: `gpt`, `gpt_asr`, `agent_wer`, `agent_cer`, `agent_word_deletions`, `agent_ref_words` |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav`; `audio.path` in JSONL is a stale absolute path from a different machine — always use `Path(audio.path).name` and look up under `EVAL_DIR/audio/` |

**Input audio (question recordings)**: stored in tar archives at
`/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/alpacaeval_full/recording.*.tar`
(10 tars, ~64 files each). Files inside are named `N-N.wav` where N is the integer index derived from `audio_path` (`data/alpacaeval_full_N.wav` → key N).

For the `alpacaeval` (non-full) variant, the tar dir is `.../alpacaeval/` and file naming may differ — inspect the first tar to confirm.

---

## Step 1 — Verify paths and inspect data

Use the Read tool to confirm `EVAL_DIR/output_asr.jsonl` and `EVAL_DIR/result-voicebench_format.jsonl` exist (read first 2 lines of each). Read `metrics.json` in full. Assert both JSONL files have the same line count.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_alpacaeval.py`. **Run as `cd /tmp && python3 analyze_alpacaeval.py`** — all paths must be absolute.

### Script structure

```python
BASE_DIR        = Path("EVAL_DIR")
TAR_DIR         = Path("/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/full-duplex/benchmarks/voicebench/alpacaeval_full")
ASR_JSONL       = BASE_DIR / "output_asr.jsonl"
RESULT_JSONL    = BASE_DIR / "result-voicebench_format.jsonl"
METRICS_JSON    = BASE_DIR / "metrics.json"
AUDIO_DIR       = BASE_DIR / "audio"
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
REPORT_HTML     = BASE_DIR / "report.html"
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
```

**Token cleaning**: strip `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` from generation text.

**Truncation detection** (important — do NOT use segment end time, all audio is padded to 40 s):
```python
ends_sentence = bool(re.search(r'[.?!"'"]$', gen_text.strip())) if gen_text else False
truncated = (not ends_sentence) and word_count > 50
```

**Failure categories** (checked in order; first match wins):

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

**GPT scores**: parse `result_entry["score"]` as list of ints → compute `avg_score` (mean) and `score_var` (variance).

**Input audio extraction**:
- Key N from `audio_path` field via `re.search(r'alpacaeval_full_(\d+)\.wav', ap)`
- Tar member name: `f"{N}-{N}.wav"`
- Scan all 10 tars once, build `key_to_tar: dict[int, (tar_path, member_name)]`
- Extract to `INPUT_AUDIO_DIR/{N}.wav`

**Output audio resolution**:
- `Path(entry["audio"]["path"]).name` → look up under `AUDIO_DIR/`

**Select 15 diverse failure examples** (budget per category: empty_response→2, mid_sentence_cutoff→3, ai_refusal→2, clarification_deflection→2, template_placeholder→2, insufficient_depth→3, high_wer→2); top up from remaining failures if < 15.

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

**Header**: benchmark name, model name, eval date. Chips: total examples, GPT score (text), GPT score (ASR), WER.

**Metric tiles** (color-coded): avg GPT score, WER, count by score bucket (1/2/3/4–5).

**Three charts** (side by side):
1. Bar: GPT score distribution (1–5), colors red→orange→yellow→green→blue
2. Doughnut: failure category breakdown (failures only)
3. Bar: response word count distribution (bins: 0–4, 5–24, 25–49, 50–99, 100–199, 200–499, 500+)

**Failure category summary cards**: one card per category with count, name, and explanation paragraph.

**15 failure example cards**: each card shows:
- Header row: `#N`, index, category badge (colored), `⚠ TRUNCATED` if applicable, GPT score badge (`avg X.X/5 (s1/s2/s3)`), word count, WER
- Left column: input question (GT text, blue left-border), `<audio>` for input
- Right column (stacked):
  - **Text Output** (cyan `#56d4dd` left-border, label "Text Output"): `generation_text` cleaned of timing tokens, monospace, scrollable max-height 100px
  - **Speech ASR** (green left-border, label "Speech ASR"): `debug_info.agent_audio_asr` lowercase transcript, monospace, scrollable max-height 100px, followed by `<audio>` for output
- Analysis box: failure explanation paragraph

Audio tags:
```html
<audio controls src="input_audio/{N}.wav" class="aplayer"></audio>
<audio controls src="audio/{filename}.wav" class="aplayer"></audio>
```
Fall back to italic text if file missing.

**Conclusions & recommendations**: two-column grid — Key Findings and Recommendations — with bullet lists drawn from the actual category counts and metrics.

---

## Step 4 — Run the script or instruct the user

Attempt `python3 /tmp/analyze_alpacaeval.py`. If Bash fails due to cwd errors, instruct user to run `cd /tmp && python3 analyze_alpacaeval.py`.

Report on completion:
1. Path to `report.html`
2. Top findings (category counts, mean score, WER, GPT text vs ASR gap)
3. Any missing audio files

---

## Step 5 — Notes for reuse

- All audio output files in `EVAL_DIR/audio/` are padded to exactly 40 s (1,764,044 bytes at 22 050 Hz mono 16-bit). Do NOT use segment-end-time as a truncation signal.
- The `result-voicebench_format.jsonl` and `output_asr.jsonl` are line-aligned (same order, same count).
- Input audio in tars uses `N-N.wav` naming. The `cuts.000000.jsonl.gz` files contain lhotse manifests with GT supervision text but no ASR transcription of the input.
- The `generation` field in `output_asr.jsonl` is the lowercase ASR of the model's spoken output (same as `debug_info.agent_audio_asr` but truncated). `generation_text` is the raw model text with timing tokens.
- Metrics live under `metrics["voicebench.alpacaeval_full"]["greedy"]`. For the non-full variant, the key is `"voicebench.alpacaeval"`.
