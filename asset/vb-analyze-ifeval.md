---
description: Analyze a VoiceBench IFeval evaluation directory and produce an HTML report with failure-case examples and playable audio. Usage: /vb-analyze-ifeval <eval-dir> <audio-root> [benchmark-name]
---

You are going to analyze a speech-model benchmark evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the evaluation results directory (contains `output.jsonl`, `output_asr.jsonl`, `metrics.json`, etc.)
- **Token 2 — `AUDIO_ROOT`**: absolute path to the directory that holds input audio files (may contain `.tar` archives, subdirectories, or both)
- **Token 3 — `BENCHMARK_NAME`** *(optional)*: human-readable name for the benchmark; if omitted, infer from the directory name or file contents

If fewer than 2 arguments are provided, ask the user for the missing paths before proceeding.

---

## Step 1 — Verify paths and explore the evaluation directory

**Critical first step**: Verify that `EVAL_DIR` exists and is accessible. Use the Read tool (not Bash) to test access — read the first line of `EVAL_DIR/output_asr.jsonl` or `EVAL_DIR/output.jsonl`. If the Read tool returns "File does not exist", stop and tell the user the path is inaccessible.

**Important**: The Bash tool's working directory may be invalid (e.g., if Claude Code was launched from a directory that was later renamed or unmounted). Never rely on the Bash tool's cwd. Always use absolute paths. If any Bash command fails with a cwd error, write the analysis script to `/tmp/analyze_benchmark.py` and instruct the user to run `cd /tmp && python3 analyze_benchmark.py` from their terminal.

Read the first 5 JSONL entries (using the Read tool with `limit=5`) to identify the schema:
- Which field contains the input prompt / transcript?
- Which field contains the model's output text or ASR transcript?
- Which field contains the relative or absolute path to the **output audio** file?
- Which field contains the relative path or key for the **input audio** file?
- Which field(s) identify per-example instruction constraints?
- Do audio paths in the JSONL use absolute paths that may reference an old or different base directory?

Read `metrics.json` in full to capture all aggregate scores.

---

## Step 2 — Understand the audio layout

**Output audio**: Extract the audio path from a few JSONL entries. The path may be:
- An absolute path pointing to a different directory than `EVAL_DIR` (the file was written during eval under a different mount/workspace and the JSONL captured the original absolute path). In this case, derive just the filename and check whether `EVAL_DIR/audio/<filename>` exists.
- A relative path resolved from `EVAL_DIR`.

Use the Read tool to test whether `EVAL_DIR/audio/<some-filename>.wav` returns a binary-file error (file exists) vs. "File does not exist". Binary-file errors mean the file is there.

**Input audio**: Determine how input audio is stored under `AUDIO_ROOT`:
- **Tar archives**: look for `recording.*.tar` or `*.tar` files; list members of the first tar using python to see file naming conventions
- **Direct files**: look for `*.wav` / `*.flac` in subdirectories
- **Mixed**: handle both

Build the mapping from the key in the JSONL (integer ID, filename stem, etc.) to the full path inside the archive or on disk.

---

## Step 3 — Write the analysis script

Write a Python 3 script to `/tmp/analyze_benchmark.py`. **The script must work when run as `cd /tmp && python3 analyze_benchmark.py`** — all paths must be absolute, and the script must not depend on any working directory.

The script must:

1. **Define all paths as absolute `Path` objects** at the top of the script:
   ```python
   BASE_DIR = Path("EVAL_DIR")          # eval results directory
   TAR_DIR  = Path("AUDIO_ROOT")        # input audio archives
   ASR_JSONL = BASE_DIR / "output_asr.jsonl"   # prefer this over output.jsonl
   OUTPUT_JSONL = BASE_DIR / "output.jsonl"
   AUDIO_DIR = BASE_DIR / "audio"              # output audio files
   INPUT_AUDIO_DIR = BASE_DIR / "input_audio"  # extracted input audio
   REPORT_HTML = BASE_DIR / "report.html"
   INPUT_AUDIO_DIR.mkdir(exist_ok=True)
   ```

2. **Resolve output audio paths robustly**: the `audio.path` field in JSONL may be an absolute path referencing a different base directory. Always extract just the filename (`Path(audio_path).name`) and look it up under `AUDIO_DIR`. If it doesn't exist there, try the absolute path as a fallback, and if it exists there, copy it to `AUDIO_DIR`.

3. **Clean model output text**: strip timing/control tokens such as `<$N.NN$>`, `<|N.NN|>`, `<s>`, `</s>` using a regex.

4. **Evaluate per-example constraints** appropriate to the benchmark type:
   - *Instruction-following (IFeval-style)*: implement lightweight checkers for every instruction type present — no-comma, word-count, sentence-count, all-lowercase, all-uppercase, keyword presence/absence, placeholder counts, JSON format, quotation wrapping, postscript, highlighted sections, bullet lists, sections with markers, repeat-prompt, end-phrase, first-word, letter frequency, two-responses separator, title, constrained response
   - *QA benchmarks*: exact match, contains-match, or F1 as appropriate
   - *Other*: infer what "correct" means from the schema and metric names

5. **Categorize failures** into meaningful groups. For audio/speech benchmarks always include:
   - `audio_incompatible`: constraints requiring text formatting (markdown, all-case, placeholders, JSON, quotation, postscript, two-response separator) that cannot be verified from audio
   - `severe_truncation`: model output very short (< 20 words) relative to task complexity
   - `length_constraint`: word/sentence/paragraph count constraints not met
   - `keyword_content`: missing required keywords or using forbidden words
   - `near_miss`: only one constraint failed by a small margin
   - `format_structure`: missing sections, bullets, JSON validity issues
   - `safety_refusal`: model refuses a benign request

6. **Select 15 diverse failure examples** across as many categories as possible. For each:
   - Extract the input audio from the tar archive to `INPUT_AUDIO_DIR/<key>.wav` — skip if already exists
   - Verify the output audio file exists at `AUDIO_DIR/<filename>.wav`
   - Record all fields needed for the HTML card

7. **Write `REPORT_HTML`** with the HTML spec from Step 4.

8. **Print progress at each stage** so the user can follow along when running manually.

---

## Step 4 — HTML report specification

Single self-contained HTML file, dark GitHub-style theme, Chart.js from CDN. Include:

### Header
Benchmark name, model name (if discoverable), evaluation date. Chips: total examples, key metrics from `metrics.json`.

### Executive summary tiles
One tile per key metric, color-coded (< 20% red, 20-50% yellow, > 50% green).

### Charts (Chart.js)
- Doughnut: failure category distribution
- Bar: total / passed / failed prompt counts

### Instruction / constraint breakdown table
One row per constraint type: name, total occurrences, pass count, fail count, pass-rate bar. Highlight 0% rows in red.

### Failure category summary cards
One card per category: count, explanation of why the model fails, audio-incompatibility flag where relevant.

### 15 example failure cards
Each card:
- Example index, failure category badge, key ID, `N/M instructions failed`
- Full input prompt (not truncated)
- Per-instruction results: each constraint with PASS/FAIL badge and short reason
- **Text Output** (cyan `#56d4dd` left-border, label "Text Output"): `generation_text` cleaned of timing tokens, monospace, scrollable max-height 120px
- **Speech ASR** (green/red left-border by pass/fail, label "Speech ASR"): `debug_info.agent_audio_asr` lowercase transcript, monospace, scrollable max-height 120px
- **Input audio**: `<audio controls src="input_audio/<key>.wav" class="aplayer"></audio>` — with fallback text if file missing
- **Output audio**: `<audio controls src="audio/<filename>.wav" class="aplayer"></audio>` — below Speech ASR, with fallback text if file missing
- Failure analysis: 3-5 sentences explaining why this example failed and what it reveals

### Conclusions & recommendations

### CSS variables
```css
:root {
  --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
  --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff;
  --gr:#3fb950; --rd:#f85149; --or:#e67e22; --yl:#d29922;
}
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
```

Chart.js: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`

---

## Step 5 — Run the script or instruct the user

**Attempt to run via Bash first** (using an absolute path to python3, e.g. `python3 /tmp/analyze_benchmark.py`). If the Bash tool fails due to an invalid cwd, do NOT retry — tell the user:

> The script is ready at `/tmp/analyze_benchmark.py`. Please run it from a terminal:
> ```
> cd /tmp && python3 analyze_benchmark.py
> ```

If the script runs successfully, report:
1. Path to `report.html`
2. Top 3 findings in 5-8 bullet points (quantified)
3. List of any missing audio files (input or output)
4. Any checkers that couldn't be implemented due to missing schema info

---

## Step 6 — Update the skill for future runs

After a successful run, note any path patterns, JSONL schema quirks, or audio layout details that were non-obvious, so they can be reused if the same benchmark folder structure is encountered again.
