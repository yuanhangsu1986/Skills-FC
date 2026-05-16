---
description: Analyze a Full-Duplex-Bench evaluation directory and produce an HTML report with five dedicated sections covering backchanneling, user interruption, pause handling (Candor), pause handling (synthetic), and turn-taking. Usage: /vb-analyze-fdb <eval-dir> [benchmark-name]
---

You are going to analyze a Full-Duplex-Bench (FDB) evaluation and produce a detailed HTML report. The benchmark (arXiv 2503.04721, ASRU 2025) assesses four core full-duplex dialogue behaviors through five dimensions.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `EVAL_DIR`**: absolute path to the eval-results root directory containing five subdirectories: `fdb_v1.backchannel`, `fdb_v1.interruption`, `fdb_v1.pause_candor`, `fdb_v1.pause_synthetic`, `fdb_v1.turn_taking`
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable name; if omitted, infer from the directory name

If no argument is provided, ask the user for the eval directory.

---

## Benchmark overview

Full-Duplex-Bench evaluates whether a spoken dialogue model can **listen and speak simultaneously** — the hallmark of full-duplex systems. Five dimensions:

| Dimension | Dir suffix | Dataset | N | Key metric(s) | TOR direction |
|---|---|---|---|---|---|
| Backchanneling | `backchannel` | ICC | 55 | TOR ↓, Freq ↑, JSD ↓ | Low = good (don't dominate) |
| User Interruption | `interruption` | Synthetic | 200 | TOR ↑, Latency ↓, Rating ↑ | High = good (respond to interrupt) |
| Pause Handling (Candor) | `pause_candor` | Candor | 216 | TOR ↓ | Low = good (don't interrupt pauses) |
| Pause Handling (Synthetic) | `pause_synthetic` | Synthetic | 137 | TOR ↓ | Low = good |
| Smooth Turn-Taking | `turn_taking` | Candor | 119 | TOR ↑, Latency ↓ | High = good (take turns promptly) |

**TOR (Turn-Over Rate)**: fraction of samples where the model produced speech (vs. staying silent). Direction depends on the dimension.

---

## Data schema

Each dimension directory (`EVAL_DIR/fdb_v1.<dim>/`) contains:

| File/Dir | Contents |
|---|---|
| `metrics.json` | Aggregate metrics (see per-dimension keys below) |
| `output.jsonl` | One JSON record per sample: `id`, `dataset`, `sample_id`, `problem` (user transcript or "Respond to the user's speech in the audio."), `generation` (raw model output with timing tokens), `audio` (output wav info with stale `path`), `audio_path` (input reference) |
| `audio/` | Output WAV files named `chatcmpl-<hash>.wav` — stale path in `audio.path`, resolve via `Path(audio["path"]).name` |
| `fdb_prepared/<sample_id>/` | Per-sample artifacts (see below) |

**Generation token format:**
- `<$T.TT$>` — silence token (T.TT seconds of silence)
- `<|T.TT|>` — speech onset token (model starts speaking at T.TT seconds)
- Free text after onset token = model's spoken content

**TOR detection**: `is_takeover = bool(re.search(r'<\|[\d.]+\|>', generation))`
**Onset time extraction**: `re.search(r'<\|([\d.]+)\|>', generation)` → float seconds

### fdb_prepared per-sample files

| File | Dimensions | Contents |
|---|---|---|
| `output.wav` | all | Model's generated speech |
| `output.json` | all | Parakeet ASR of output: `{text: str, chunks: [{text, timestamp: [start, end]}]}` |
| `input.wav` | interruption, turn_taking | Input audio fed to model |
| `interrupt.json` | interruption | `[{context: str, interrupt: str, timestamp: [start_sec, end_sec]}]` |
| `rating.json` | interruption | `{analysis: str, rating: int (1–5)}` (may be absent for ~3% of samples) |
| `turn_taking.json` | turn_taking | `[{text: "[TURN-TAKING]", timestamp: [start_sec, end_sec]}]` |

**fdb_prepared naming by dimension:**
- backchannel: `fdb_prepared/{N}/` (N = sample_id, 0-indexed)
- interruption: `fdb_prepared/synthetic_user_interruption_{N}/`
- pause_candor: `fdb_prepared/candor_pause_handling_{N}/`
- pause_synthetic: `fdb_prepared/synthetic_pause_handling_{N}/`
- turn_taking: `fdb_prepared/candor_turn_taking_{N}/`

Derive `fdb_prepared_dir` from each sample's `id` field: the id IS the subdirectory name (except for backchannel where id=`icc_backchannel_N` but fdb_prepared dir is just `N`).

### metrics.json keys

| Dimension | metrics.json key | Fields |
|---|---|---|
| backchannel | `fdb_v1.backchannel.greedy` | `tor`, `jsd`, `frequency` |
| interruption | `fdb_v1.interruption.greedy` | `turn` (=TOR), `latency`, `rating`, `tor_pct`, `latency_ms` |
| pause_candor | `fdb_v1.pause_candor.greedy` | `turn` (=TOR), `tor_pct` |
| pause_synthetic | `fdb_v1.pause_synthetic.greedy` | `turn` (=TOR), `tor_pct` |
| turn_taking | `fdb_v1.turn_taking.greedy` | `turn` (=TOR), `latency`, `tor_pct`, `latency_ms` |

---

## Step 1 — Verify paths and inspect data

Confirm each of the 5 subdirectories exists. Read each `metrics.json`. Read the first 2 lines of each `output.jsonl` to confirm schema. Check that `fdb_prepared` subdirectories exist.

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_fdb.py`. Run as `python3 /tmp/analyze_fdb.py`. All paths must be absolute.

### Script constants

```python
EVAL_DIR   = Path("EVAL_DIR")
MODEL_NAME = EVAL_DIR.parent.name   # infer from path
REPORT_HTML = EVAL_DIR / "report.html"
TOKEN_RE = re.compile(r'<\$[\d.]+\$>|<\|[\d.]+\|>')

def is_takeover(gen):
    return bool(re.search(r'<\|[\d.]+\|>', gen))

def onset_time(gen):
    m = re.search(r'<\|([\d.]+)\|>', gen)
    return float(m.group(1)) if m else None

def clean_gen(gen):
    return TOKEN_RE.sub('', gen).strip()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def audio_tag(abs_path, rel_from_report, fallback):
    """Use rel_from_report as src; check abs_path exists."""
    if Path(abs_path).exists():
        return f'<audio controls src="{rel_from_report}" class="aplayer"></audio>'
    return f'<em style="color:var(--tx2)">{fallback}</em>'
```

### Per-dimension data loading

For each dimension, load `output.jsonl` and the corresponding `metrics.json` fields. For each sample, also read from `fdb_prepared/`:

**Backchannel** — for each entry with `id = icc_backchannel_N`:
- `prepared_dir = DIM_DIR / "fdb_prepared" / entry["sample_id"]`
- Read `prepared_dir / "output.json"` for ASR text
- Compute: `tor = is_takeover(gen)`, `response_text = clean_gen(gen)`, `word_count = len(response_text.split())`
- Classify: `silent` if not tor; `backchannel` if tor and word_count <= 2; `full_takeover` if tor and word_count > 2
- Output audio: `prepared_dir / "output.wav"` → HTML src: `fdb_v1.backchannel/fdb_prepared/{sample_id}/output.wav`

**Interruption** — for each entry with `id = synthetic_user_interruption_N`:
- `prepared_dir = DIM_DIR / "fdb_prepared" / entry["id"]`
- Read `interrupt.json` → `context`, `interrupt_text`, `interrupt_end = interrupt_json[0]["timestamp"][1]`
- Read `rating.json` → `rating`, `analysis` (use None if file absent)
- Read `output.json` → `response_text = output_json["text"]`, `first_chunk_start = chunks[0]["timestamp"][0]` if chunks else None
- Compute: `tor = is_takeover(gen)`, `latency = onset_time(gen) - interrupt_end if tor else None`
- Input audio src: `fdb_v1.interruption/fdb_prepared/{id}/input.wav`
- Output audio src: `fdb_v1.interruption/fdb_prepared/{id}/output.wav`

**Pause Candor / Pause Synthetic** — for each entry:
- `prepared_dir = DIM_DIR / "fdb_prepared" / entry["id"]`
- Read `output.json` → `response_text = output_json["text"]`
- Compute: `tor = is_takeover(gen)`, `onset = onset_time(gen)`
- Outcome: `good` if not tor (silence = correct) else `bad` (interrupted)
- Output audio src: `fdb_v1.pause_{candor|synthetic}/fdb_prepared/{id}/output.wav`
- Input transcript: use `entry["problem"]` (already contains the user speech transcript for these dimensions)

**Turn-taking** — for each entry:
- `prepared_dir = DIM_DIR / "fdb_prepared" / entry["id"]`
- Read `turn_taking.json` → `turn_end = turn_taking_json[0]["timestamp"][1]`
- Compute: `tor = is_takeover(gen)`, `onset = onset_time(gen)`, `latency = onset - turn_end if tor else None`
- Outcome: `took_turn` if tor else `missed_turn`
- Input audio src: `fdb_v1.turn_taking/fdb_prepared/{id}/input.wav`
- Output audio src: `fdb_v1.turn_taking/fdb_prepared/{id}/output.wav`

### Example selection (per dimension, 8 cards each)

- **Backchannel**: 2 `silent`, 3 `backchannel` (ideal), 3 `full_takeover` (bad)
- **Interruption**: 4 highest-rated (rating ≥ 4), 4 lowest-rated (rating ≤ 2); sort by rating desc/asc
- **Pause Candor**: 4 `good` (silent), 4 `bad` (interrupted)
- **Pause Synthetic**: 4 `good`, 4 `bad`
- **Turn-taking**: 4 `took_turn` (prefer lowest latency), 4 `missed_turn`

Top-up from remaining samples if a category has fewer examples than budget.

---

## Step 3 — HTML report specification

Single self-contained HTML file at `EVAL_DIR/report.html`. Dark GitHub-style theme, Chart.js from CDN.

### CSS variables
```css
:root {
  --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
  --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff;
  --gr:#3fb950; --rd:#f85149; --or:#e3633c; --yl:#d29922;
}
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
.chart-box { height:280px; background:var(--bg2); border:1px solid var(--bd); border-radius:8px; padding:16px; }
```

Chart.js CDN: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`

### Global header

Benchmark title, model name, eval date, nav links to each section anchor.

**Summary dashboard**: one compact tile per dimension showing the primary metric with color coding:
- Backchannel: TOR (green if < 0.3, yellow if < 0.6, red if ≥ 0.6) + JSD
- Interruption: TOR (green if ≥ 0.9), Rating (green if ≥ 4.0), Latency
- Pause Candor: TOR (green if < 0.3, red if ≥ 0.7)
- Pause Synthetic: TOR (same thresholds as Candor)
- Turn-taking: TOR (green if ≥ 0.8), Latency

### Five sections (one per dimension)

Each section follows this template:

```
<section id="{dim-id}">
  <h2>{Dimension Title}</h2>
  <p class="dim-desc">{one paragraph explaining what this tests and what the ideal behavior is}</p>

  <!-- Metric tiles row -->
  <div class="tiles">...</div>

  <!-- Charts (2 side by side) -->
  <div class="charts-2col">
    <div class="chart-box">...</div>
    <div class="chart-box">...</div>
  </div>

  <!-- 8 example cards -->
  <h3>Example Cards</h3>
  {8 cards}
</section>
```

#### Section descriptions

**Backchanneling**: "The model listens to a user speaking a complete turn (ICC corpus, 55 samples). The ideal behavior is to produce short backchannels (e.g., 'mhm', 'yeah') or remain silent — NOT to take the floor and speak a full turn. TOR should be low; when the model does produce audio, it should be brief (backchannel, ≤ 2 words). JSD measures how well the model's backchannel timing matches human annotators."

**User Interruption**: "The model is speaking when the user interrupts mid-sentence (synthetic data, 200 samples). The ideal behavior is to immediately stop, acknowledge the interruption, and respond coherently to the new input. TOR should be high (model responds); latency should be low; GPT-4o rating (1–5) measures response coherence and relevance."

**Pause Handling (Candor / Synthetic)**: "The user pauses mid-turn (Candor: natural speech pauses 0.4–1.0s; Synthetic: ChatTTS-generated). The model should recognize the speaker still holds the floor and remain silent. TOR should be low — any takeover is an incorrect interruption."

**Smooth Turn-Taking**: "The user finishes their turn and a natural gap appears (Candor, gaps < 0.4s, 119 samples). The model should promptly take the floor. TOR should be high; latency (turn-end to first model word) should be low."

#### Dimension-specific charts

**Backchanneling** (2 charts):
1. Doughnut: Silent / Backchannel / Full Takeover — with counts and colors (blue/green/red)
2. Bar with single group: TOR, JSD, Freq (normalized to 0–1 scale; Freq × 50 for visibility) with ideal-range reference lines

**Interruption** (2 charts):
1. Bar: rating distribution (1–5), green for 4–5, yellow for 3, red for 1–2
2. Histogram: latency distribution (bins: 0–0.5s, 0.5–1s, 1–2s, 2–3s, 3s+) — only for TOR=1 samples

**Pause Candor** (2 charts):
1. Doughnut: Silent (correct) / Interrupted (wrong) — green/red
2. Single-value gauge bar: TOR with threshold reference line at 0.3

**Pause Synthetic** (same structure as Pause Candor)

**Turn-Taking** (2 charts):
1. Doughnut: Took Turn / Missed Turn — green/red
2. Histogram: latency distribution for taken turns (bins: 0–0.3s, 0.3–0.6s, 0.6–1s, 1–2s, 2s+)

#### Example card format

**Backchannel card**:
- Header: `#{N}`, idx, outcome badge (`SILENT`/`BACKCHANNEL`/`FULL TAKEOVER`), word count
- Left: user transcript (from `problem`), blue left-border
- Right: model generation (cleaned, cyan left-border) + output audio player
- Analysis: explain what the model did and whether it was correct

**Interruption card**:
- Header: `#{N}`, idx, outcome badge (`TOR=1`/`TOR=0`), rating badge (if available, colored 1–5), latency badge
- Left: context + interruption text (from interrupt.json), blue left-border; input audio player
- Right: model response text (from output.json), cyan left-border; GPT analysis (orange left-border); output audio player
- Analysis: explain rating

**Pause card** (Candor/Synthetic):
- Header: `#{N}`, idx, outcome badge (`SILENT ✓`/`INTERRUPTED ✗`), onset time if interrupted
- Left: user transcript (from `problem` — truncated to 150 chars), blue left-border
- Right: model generation (cleaned), cyan left-border; output audio player
- Analysis: explain behavior

**Turn-taking card**:
- Header: `#{N}`, idx, outcome badge (`TOOK TURN ✓`/`MISSED TURN ✗`), latency badge (ms)
- Left: turn boundary info (from turn_taking.json: turn end timestamp), input audio player
- Right: model response text (from output.json), cyan left-border; output audio player
- Analysis: explain latency or why turn was missed

All audio tags use relative paths from `EVAL_DIR/report.html`:
```html
<audio controls src="fdb_v1.{dim}/fdb_prepared/{id}/output.wav" class="aplayer"></audio>
<audio controls src="fdb_v1.{dim}/fdb_prepared/{id}/input.wav" class="aplayer"></audio>
```
**Always check `(EVAL_DIR / rel_path).exists()` — NEVER `Path(rel_path).exists()` — since the script runs from /tmp and relative paths would resolve incorrectly.**

Fall back to italic text if file missing.

### Conclusions section

Two-column grid: Key Findings (bullet list per dimension) and Recommendations (suggested improvements).

---

## Step 4 — Run the script

Run `python3 /tmp/analyze_fdb.py`.

Report on completion:
1. Path to `report.html`
2. Summary table: per dimension — TOR, latency (if applicable), rating (if applicable)
3. Standout observations (e.g. "Pause Candor TOR=70% is high — model interrupts too often")

---

## Step 5 — Notes

- `fdb_prepared` dir name equals the `id` field in output.jsonl for all dimensions **except** backchannel, where `id=icc_backchannel_N` but prepared dir is just the numeric `sample_id` (`N`).
- For backchannel: output audio in `fdb_prepared/{sample_id}/output.wav`; HTML src: `fdb_v1.backchannel/fdb_prepared/{sample_id}/output.wav`.
- For interruption & turn_taking: input audio in `fdb_prepared/{id}/input.wav`.
- For pause dims: no input.wav in fdb_prepared; use `problem` field as input transcript instead.
- Backchannel `problem` field contains the user's ICC corpus utterance transcript — use it as the conversation context.
- Pause dims `problem` field contains the user's Candor/synthetic utterance transcript — use it.
- Turn-taking `problem` = "Respond to the user's speech in the audio." — not useful as text; rely on audio + turn_taking.json.
- Missing `rating.json` in ~3% of interruption samples is normal — show "N/A" for those.
- For latency computation: `onset_time(gen) - event_end_sec`. If `onset_time` is None (TOR=0), latency is undefined.
- TOR desirability summary: Backchannel ↓ Low = good; Interruption ↑ High = good; Pause ↓ Low = good; Turn-taking ↑ High = good.
- Reference values from paper (for comparison in tiles): Backchannel TOR — Gemini Live 0.091 (best), dGSLM 0.691. Interruption TOR — Moshi 1.000. Pause TOR — Gemini Live 0.255 (best). Turn-taking TOR — dGSLM 0.975 (best). Use these as reference lines in charts.
