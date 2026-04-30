---
description: Analyze a BFCL single-turn function-channel evaluation directory and produce an HTML report with per-category accuracy, error type breakdown, and 8 example cards per category. Usage: /analyze-bfcl <output-dir> [benchmark-name]
---

You are going to analyze a BFCL (Berkeley Function Calling Leaderboard) single-turn function-channel evaluation and produce a detailed HTML report.

## Arguments

Parse `$ARGUMENTS` as whitespace-delimited tokens:
- **Token 1 — `OUTPUT_DIR`**: absolute path to the BFCL output directory (contains `eval-results/`)
- **Token 2 — `BENCHMARK_NAME`** *(optional)*: human-readable label; if omitted, infer from directory name

If no argument is provided, ask the user for the output directory.

---

## Data layout

```
{OUTPUT_DIR}/
└── eval-results/
    └── {category}/           # simple | parallel | multiple | parallel_multiple | irrelevance
        ├── output.jsonl      # one JSON per sample: id, generation, expected_call, required_fields, question_text
        └── metrics.json      # {"bfcl_fc.{category}": {"greedy": {"accuracy", "num_samples", "num_correct"}}}
```

### output.jsonl fields
| Field | Contents |
|---|---|
| `id` | Sample identifier |
| `generation` | Raw model output — correct form is `<TOOLCALL>[{...}]</TOOLCALL>`, or empty/plain-text if no tool call was made |
| `expected_call` | `[{"func_name": {"param": value, ...}}, ...]` — list of expected tool calls |
| `required_fields` | `{"func_name": [["param", "type"], ...]}` — which params must match |
| `question_text` | Text of the spoken question (ASR or prepared transcript) |

### metrics.json schema
```json
{
  "bfcl_fc.{category}": {
    "greedy": {
      "accuracy": 72.5,
      "num_samples": 400,
      "num_correct": 290
    }
  }
}
```

---

## Step 1 — Verify and inspect

Confirm `OUTPUT_DIR/eval-results/` exists. For each of the 5 categories, read the first 2 lines of `output.jsonl` and `metrics.json` in full. Note which categories are actually present (not all 5 may exist).

---

## Step 2 — Write the analysis script

Write `/tmp/analyze_bfcl.py`. Run as `python3 /tmp/analyze_bfcl.py`. All paths must be absolute.

```python
import json, re, sys
from pathlib import Path

OUTPUT_DIR  = Path("OUTPUT_DIR")
EVAL_DIR    = OUTPUT_DIR / "eval-results"
ALL_CATS    = ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"]
CATEGORIES  = [c for c in ALL_CATS if (EVAL_DIR / c / "output.jsonl").exists()]
REPORT_HTML = OUTPUT_DIR / "report.html"

CAT_DESC = {
    "simple":            "Single tool call with a single function — the model must invoke exactly one function with correct parameters.",
    "parallel":          "Multiple simultaneous tool calls to the same function — the model must invoke the function multiple times in one response.",
    "multiple":          "A single call to one of several available functions — the model must choose the right function and supply correct parameters.",
    "parallel_multiple": "Multiple simultaneous calls across different functions — requires both parallel execution and correct function selection.",
    "irrelevance":       "No tool call should be made — the user's request cannot be fulfilled by any of the available functions.",
}

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def load_json(p):
    with open(p) as f: return json.load(f)

def parse_toolcall(generation):
    """Parse <TOOLCALL>...</TOOLCALL> → list of {func_name: args_dict}."""
    m = re.search(r'<TOOLCALL>(.*?)</TOOLCALL>', generation, re.DOTALL)
    if not m: return []
    raw = m.group(1).strip()
    if not raw.startswith('['): raw = '[' + raw
    if not raw.endswith(']'): raw = raw + ']'
    try:
        calls = json.loads(raw)
        return [{tc['name']: tc.get('arguments', {})} for tc in calls if isinstance(tc, dict) and 'name' in tc]
    except Exception:
        return None  # parse error

def failure_reason(entry, parsed_calls, is_correct):
    """Return (error_type, detail_str) for incorrect entries."""
    if is_correct:
        return 'correct', ''
    gen = entry.get('generation', '')
    if not gen.strip():
        return 'no_generation', 'Model returned empty output'
    if '<TOOLCALL>' not in gen:
        return 'no_toolcall', f'No <TOOLCALL> block in output: {gen[:120]!r}'
    if parsed_calls is None:
        return 'parse_error', f'<TOOLCALL> block could not be parsed as JSON'
    expected = entry.get('expected_call', [])
    req = entry.get('required_fields', {})
    # irrelevance: should be empty call list
    if not expected:
        if parsed_calls:
            return 'spurious_call', f'Expected no tool call; got {len(parsed_calls)} call(s): {parsed_calls[0]}'
        return 'other', 'Empty expected and empty parsed'
    if len(parsed_calls) != len(expected):
        return 'wrong_count', f'Expected {len(expected)} call(s), got {len(parsed_calls)}'
    # check first call for detail
    if parsed_calls and expected:
        gen_name = list(parsed_calls[0].keys())[0] if parsed_calls[0] else ''
        exp_name = list(expected[0].keys())[0] if expected[0] else ''
        if gen_name != exp_name:
            return 'wrong_function', f'Expected {exp_name!r}, got {gen_name!r}'
        gen_args = parsed_calls[0].get(gen_name, {})
        exp_args = expected[0].get(exp_name, {})
        extra = set(gen_args) - set(exp_args)
        if extra:
            return 'extra_param', f'Unexpected parameters: {extra}'
        req_params = {p for p, _ in req.get(exp_name, req.get(gen_name, []))}
        missing = req_params - set(gen_args)
        if missing:
            return 'missing_param', f'Missing required parameters: {missing}'
        for p in req_params:
            if p in gen_args and p in exp_args:
                if str(gen_args[p]).lower().strip() != str(exp_args[p]).lower().strip():
                    return 'wrong_value', f'Parameter {p!r}: expected {exp_args[p]!r}, got {gen_args[p]!r}'
    return 'other', 'Mismatch (see generation)'
```

### Per-category processing

For each category:
1. Load `EVAL_DIR/{category}/output.jsonl`
2. Load `metrics.json` → extract `accuracy`, `num_samples`, `num_correct`
3. Re-score each entry locally (matches scoring logic in `run_bfcl_fc_scoring.py`):

```python
from nemo_skills.dataset.bfcl_single_turn_function_channel.score import _score_one

for entry in entries:
    parsed = parse_toolcall(entry['generation'])
    if parsed is None:
        parsed_calls = []
    else:
        parsed_calls = parsed
    candidate = {'tool_response': parsed_calls}
    is_correct = _score_one(candidate, entry['expected_call'], entry['required_fields'])
    err_type, err_detail = failure_reason(entry, parsed, is_correct)
    entry['_correct'] = is_correct
    entry['_err_type'] = err_type
    entry['_err_detail'] = err_detail
```

If the import fails (running outside the repo), fall back to: `is_correct = bool(parse_toolcall(entry['generation']))` (approximate).

### Example card selection (8 per category)

Select 4 correct (prefer those with the most complex expected_call — longest JSON) and 4 incorrect. For the 4 incorrect, try to cover different error types: prefer `no_toolcall`, `wrong_function`, `wrong_value`, `wrong_count`/`extra_param`/`missing_param` in that priority; top-up from remaining incorrect if buckets are empty.

---

## Step 3 — HTML report

Single file at `OUTPUT_DIR/report.html`. Dark GitHub theme, Chart.js from CDN.

Same CSS as `analyze-bba.md` (same variables, `.aplayer`, `.chart-box`, `.tile`, `.card`, `.block`, `.scroll-pre`, `.analysis-box`, `.dim-desc`).

### Structure

**Header**: "BFCL Function-Channel — Analysis Report", model name (infer from `OUTPUT_DIR.name`), date.

**Nav**: links to each category section + `#conclusions`.

**Summary dashboard** (flex tiles, one per category + aggregate):
- Each tile: category name, accuracy %, border-top color: green ≥70%, yellow 40–69%, red <40%
- Aggregate tile: mean accuracy across present categories

**Two overview charts** (side by side):
1. Horizontal bar: per-category accuracy %, sorted descending
2. Stacked bar: error type breakdown per category (colors: correct=green, no_toolcall=red, wrong_function=orange, wrong_value=yellow, wrong_count=purple `#a371f7`, extra_param/missing_param=teal `#56d4dd`, parse_error/other=grey)

**Per-category sections** (one per present category):

```
<section id="{category}">
  <h2>§N — {Category Title}</h2>
  <p class="dim-desc">{CAT_DESC[category]}</p>
  <div class="tiles">accuracy % tile | num_correct/num_samples tile</div>
  <div class="charts-2col">
    [doughnut: correct / incorrect]
    [horizontal bar: error type counts for incorrect entries]
  </div>
  <h3>Example Cards (8: 4 correct + 4 incorrect)</h3>
  {8 cards}
</section>
```

**Example card**:
- Header: `#{N}`, sample id (dim), correct/incorrect badge, error type badge (for incorrect), `Calls: {len(expected_call)}`
- Two-column body:
  - Left:
    - Question block (blue `var(--ac)` border): `question_text`
    - Expected call block (orange `var(--or)` border): formatted JSON of `expected_call` in `<pre>` (indent=2, max-height 120px scrollable)
  - Right:
    - Generated output block (cyan `#56d4dd` border if correct, red `var(--rd)` border if incorrect): raw `generation` text in `<pre>` (max-height 120px)
    - Parsed call block (dim grey border): formatted JSON of `parse_toolcall(generation)` or `"(could not parse)"` or `"(no <TOOLCALL> block)"` in `<pre>` (max-height 120px)
- Analysis box: `err_detail` string (for incorrect) or "Parsed tool call matches expected call." (for correct)

### JavaScript

Charts for:
- Overview horizontal bar
- Overview stacked bar (error types; one dataset per error type)
- Per-category doughnuts (correct/incorrect)
- Per-category horizontal bars (error type counts)

---

## Step 4 — Run and report

Run `python3 /tmp/analyze_bfcl.py`.

Report:
1. Path to `report.html`
2. Per-category accuracy table
3. Top error types across all categories
4. Which categories had the most `no_toolcall` failures (model didn't use the function channel)

---

## Notes

- The `irrelevance` category expects an EMPTY tool call list — a correct response is one where the model produces NO `<TOOLCALL>` block (or an empty one). This is the opposite of other categories.
- `parse_toolcall` returns `None` (parse error) vs `[]` (no block or empty block) vs a list — distinguish these for error classification.
- `expected_call` may be an empty list for `irrelevance`; handle this before checking indices.
- All path existence checks must use absolute paths since the script runs from /tmp.
- If `nemo_skills` is not importable, fall back to a simplified scorer that only checks whether `<TOOLCALL>` is present and the function name matches; note the fallback in the report header.
