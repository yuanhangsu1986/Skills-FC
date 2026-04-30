---
description: Analyze a conv_behav evaluation directory and produce an HTML report covering turn-taking, barge-in, and back-channeling metrics with charts and example audio cards. Usage: /analyze-conv-behav <output-dir> [benchmark-name]
---

You are going to analyze a conversational behavior (conv_behav) evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `OUTPUT_DIR`**: absolute path to the conv_behav output directory (contains `eval-results/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable label; if omitted, infer from directory name

If no argument is provided, ask the user for the output directory.

---

## Data layout

```
{OUTPUT_DIR}/
└── eval-results/
    ├── metrics.json                               # all scored metrics
    └── validation_logs/
        ├── pred_wavs/                             # model output WAV files
        └── metadatas/
            └── {dataset_name}.json                # per-conversation timing + transcript data
```

### metrics.json schema
```json
{
  "conv_behav": {
    "greedy": {
      "tt_latency_ms":        320.5,
      "tt_precision":         85.2,
      "tt_recall":            78.1,
      "tt_f1":                81.5,
      "barge_in_success_rate": 91.0,
      "barge_in_latency_ms":  410.2,
      "bc_accuracy":          67.5,
      "cutoff_rate":          8.3,
      "user_eou_precision":   88.0,
      "user_eou_recall":      82.4,
      "user_eou_f1":          85.1,
      "user_eou_latency_ms":  290.0,
      "user_wer":             12.5,
      "num_evaluated":        150
    }
  }
}
```

### {dataset_name}.json schema
Each entry in this file represents one conversation sample and contains timing data. The exact schema depends on the NeMo inference script output; typical fields include `id`, `pred_audio_path`, and per-event timestamps. Read the first entry to discover the actual schema before writing the analysis script.

---

## Step 1 — Verify and inspect

1. Confirm `OUTPUT_DIR/eval-results/metrics.json` exists; read it in full.
2. Confirm `eval-results/validation_logs/pred_wavs/` exists; list up to 10 filenames to understand naming convention.
3. Find `eval-results/validation_logs/metadatas/` and list JSON files; read the first entry of the first JSON file found to understand the schema. Note the actual field names.
4. Count the number of WAV files in `pred_wavs/`.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_conv_behav.py`. Run as `python3 /tmp/analyze_conv_behav.py`. All paths must be absolute.

```python
import json, os, sys
from pathlib import Path

OUTPUT_DIR   = Path("OUTPUT_DIR")
EVAL_DIR     = OUTPUT_DIR / "eval-results"
PRED_WAVS    = EVAL_DIR / "validation_logs" / "pred_wavs"
METADATA_DIR = EVAL_DIR / "validation_logs" / "metadatas"
METRICS_FILE = EVAL_DIR / "metrics.json"
REPORT_HTML  = OUTPUT_DIR / "report.html"

def load_json(p):
    with open(p) as f: return json.load(f)

metrics_raw = load_json(METRICS_FILE)
metrics = metrics_raw.get("conv_behav", {}).get("greedy", {})

# Discover metadata file
meta_files = sorted(METADATA_DIR.glob("*.json")) if METADATA_DIR.exists() else []
metadata = []
dataset_name = ""
if meta_files:
    dataset_name = meta_files[0].stem
    with open(meta_files[0]) as f:
        raw = json.load(f)
    # raw may be a list or a dict with a list value — normalize to a list
    if isinstance(raw, list):
        metadata = raw
    elif isinstance(raw, dict):
        # try common list keys
        for k in ('data', 'samples', 'entries', 'conversations', dataset_name):
            if k in raw and isinstance(raw[k], list):
                metadata = raw[k]
                break
        if not metadata:
            metadata = list(raw.values())[0] if raw else []

# Collect pred_wav paths
wav_files = sorted(PRED_WAVS.glob("*.wav")) if PRED_WAVS.exists() else []
```

### Example card selection

Select up to 12 audio example "cards" from `pred_wavs/` — just audio players with file name and index. Since there is no per-sample text transcript in the standard output, each card shows:
- The WAV filename
- File size in KB (as a proxy for response length)
- Relative audio player src: `eval-results/validation_logs/pred_wavs/{filename}`

Sort files alphabetically; select first 6 and last 6 (or all if < 12).

If metadata is available and contains per-sample fields (e.g., turn-taking event timestamps), enrich cards with whatever is available. Use `json.dumps(entry, indent=2)[:400]` to show a truncated JSON snippet per card if metadata aligns with the WAV files by index.

---

## Step 3 — HTML report

Single file at `OUTPUT_DIR/report.html`. Dark GitHub theme, Chart.js from CDN.

Same CSS variables and component classes as the other analyze-* skills.

### Structure

**Header**: "Conversational Behavior — Analysis Report", model name (infer from `OUTPUT_DIR.name`), date.

**Nav**: `#turn-taking`, `#barge-in`, `#backchannel`, `#eou` (if user_eou_f1 present), `#audio-samples`, `#conclusions`.

**Summary dashboard** (metric tiles, one per key metric):
- Turn-Taking F1 (green ≥80%, yellow 60–79%, red <60%)
- TT Latency ms (green ≤400ms, yellow ≤700ms, red >700ms)
- Barge-In Rate (green ≥85%, yellow 70–84%, red <70%)
- Barge-In Latency ms (green ≤500ms, yellow ≤800ms, red >800ms)
- BC Accuracy (green ≥70%, yellow 50–69%, red <50%)
- Cutoff Rate (green ≤10%, yellow ≤20%, red >20%) ← lower is better

**Two overview charts** (side by side):
1. Radar chart with 5 axes: TT-F1 (%), Barge-In Rate (%), BC Accuracy (%), User-EOU F1 (% if present, else 0), (100 - Cutoff Rate) %. This gives a "shape" of conversational behavior quality.
2. Bar chart: Precision vs Recall for TT and (if present) User-EOU — 4 bars total.

**Four metric sections**:

**§1 Turn-Taking** (`id="turn-taking"`):
- Desc: "The model should start speaking promptly when the user finishes their turn. Precision measures how often the model's response onset aligns with actual turn boundaries; Recall measures how many turn boundaries the model responded to. Latency is the delay from turn-end to model speech onset."
- Tiles: TT-F1, TT Precision, TT Recall, TT Latency ms
- Two charts: precision/recall gauge bars + latency context (compare to FDB reference: dGSLM 352ms, Moshi 265ms)

**§2 Barge-In** (`id="barge-in"`):
- Desc: "The model should stop speaking and respond when the user interrupts (barges in). Success rate measures the fraction of interruptions the model correctly handled. Latency is the time from user speech onset to model response."
- Tiles: Barge-In Success Rate, Barge-In Latency ms
- One chart: single bar showing barge-in success rate vs 100% target

**§3 Back-Channeling** (`id="backchannel"`):
- Desc: "The model should produce brief acknowledgement sounds (back-channels like 'mhm', 'yeah') at appropriate times during the user's turn without taking the floor."
- Tiles: BC Accuracy
- One chart: single gauge bar (green if ≥70%)

**§4 User EOU Detection** (`id="eou"`) — only if `user_eou_f1` is present in metrics:
- Desc: "End-of-Utterance detection accuracy: how well the model identifies when the user has finished speaking."
- Tiles: User-EOU F1, User-EOU Precision, User-EOU Recall, User-EOU Latency ms
- Chart: precision vs recall bar

**Audio Samples** (`id="audio-samples"`):
- Grid of audio player cards (3 columns, up to 12 files)
- Each card: filename, file size KB, `<audio controls src="..." class="aplayer">`
- If metadata available: show truncated JSON snippet

**Conclusions** (`id="conclusions"`): two-column Key Findings + Recommendations.

---

## Step 4 — Run and report

Run `python3 /tmp/analyze_conv_behav.py`.

Report:
1. Path to `report.html`
2. Key metrics table: TT-F1, Barge-In Rate, BC Accuracy, Cutoff Rate, N evaluated
3. Any metadata schema observations (what fields were discovered)
4. Number of audio files found

---

## Notes

- Metrics may be partially missing if the scoring script failed for some sub-task; check each key with `.get()` and show "N/A" for missing values.
- The metadata JSON schema is not fixed — the skill must inspect the first entry at Step 1 and adapt. If the schema is unexpected, show raw JSON in the cards rather than failing.
- `cutoff_rate` is lower-is-better; invert it to `(100 - cutoff_rate)` for the radar chart axis.
- All path existence checks must use absolute paths; the script runs from /tmp.
- If `pred_wavs/` has no WAV files, skip the audio samples section and note it in the report header.
