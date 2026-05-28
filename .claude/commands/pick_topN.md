---
name: pick_topN
description: Pick the Top-N voice-chat checkpoints from an `<name>.xlsx` produced by `scripts/visualize_metrics.py`. The skill applies a multi-metric, anomaly-aware heuristic distilled from the original voice-chat analysis recipe and writes the result to `<output_dir>/<stem>_top<N>.json` so the visualizer's `--html_mode report` mode can render the final HTML report.
---

# pick_topN — Top-N checkpoint selection

This skill is the LLM-driven middle step of the two-stage report workflow
implemented by `scripts/visualize_metrics.py --html_mode report`. The visualizer
writes an Excel workbook with the per-benchmark metrics; this skill reads that
workbook, applies the selection heuristics below, and writes a picks JSON file
that the visualizer's report mode consumes.

**This skill is non-interactive: pick, write JSON, print summary, done.** It
does not ask the user "is this OK?", does not pause for approval, and does not
use AskUserQuestion. After the file is written, a concise summary is printed
to chat and the skill exits cleanly. If the user wants to override picks, they
can edit the picks JSON manually before re-running the visualizer.

## Input arguments

- `--N <int>` — Number of checkpoints to recommend. Default `3`.
- `--data_file <path>` — REQUIRED. Path to the `.xlsx` file produced by
  `scripts/visualize_metrics.py`.
- `--output_dir <dir>` — Directory where the picks JSON is written. Default
  `asset/`.

## Step 1 — Read `--data_file`

Open the xlsx supplied via `--data_file`. It is the canonical metrics workbook
produced by `scripts/visualize_metrics.py`. Do not glob the directory; use the
path the user passed in exactly. If the file does not exist, print an error
noting that the visualizer must be run first to produce the xlsx, and exit
without writing a picks file.

Compute the output filename as:

    <output_dir>/<stem-of-data_file>_top<N>.json

For example, `--data_file asset/test.xlsx --N 3` writes
`asset/test_top3.json`.

## Step 2 — Optionally peek at the sidecars

Per-ckpt headline / audio_examples information lives in the sibling sidecars
under the asset directory:

- `asset/<name>_<ckpt>.json` — full sidecar (metrics + headlines +
  audio_examples). Here `<name>` is the stem of `--data_file`.

You do not need to read all sidecars. If a candidate is unclear after the
xlsx-only pass, open the matching sidecar to confirm headline values, latency
fields, or other nuance.

## Step 3 — Compute the decision-grade metrics from the xlsx

From `vb_aggregate` sheet:

- **VB Aggregate** = column `B` (`aggregate`). This is the AJ-formula
  VoiceBench number — use it as the headline VoiceBench score (do **not** use
  the per-bench `headline.greedy` from the `vb_nonmcq` / `vb_mcq` sheets).

From `bfcl.<split>` sheets (or the main `bfcl` sheet):

- **BFCL Priority** = mean(`bfcl.simple.accuracy`, `bfcl.multiple.accuracy`,
  `bfcl.irrelevance.accuracy`). This is the "production-relevant" composite.
- **BFCL Irrelevance** = `bfcl.irrelevance.accuracy`. **Hard filter:** any ckpt
  with Irrelevance < 70% is a *critical regression* — exclude it from picks
  unless absolutely no alternative exists, and flag prominently.

From the `fdb_v1` main sheet:

- **FDB v1 TOR Summary** = round((1 − pause_avg) + TT_TOR/100 + Intr_TOR/100) /
  3 × 100, 1) where `pause_avg = (synth_tor + candor_tor) / 2 / 100`.
  - Source cols: `greedy.pause_synthetic.tor_pct`,
    `greedy.pause_candor.tor_pct`, `greedy.turn_taking.tor_pct`,
    `greedy.interruption.tor_pct`.
- **FDB v1 Adherence** = `greedy.interruption.rating` (GPT-4o /5, ↑ better).
- **FDB v1 Combined Latency** = mean(`greedy.turn_taking.latency_ms`,
  `greedy.interruption.latency_ms`). ↓ better. (Optional tie-break.)

From the `fdb_v1_5` main sheet:

- **FDB v1.5 Headline** = `headline.greedy`. **Anomaly filter:** if the value
  is `> 1.0`, mark it as a normalization artifact and *exclude* it from "best"
  consideration — it usually means sub-metrics were missing.

From the `bba` main sheet:

- **BBA Headline** = `headline.greedy`.

From the `conv_behav` main sheet:

- **Conv F1** = `greedy.overall.tt_f1`.

## Step 4 — Selection heuristics

These are *judgment* rules, not a single weighted score. Apply them together.

1. **Aim for diversity.** Among the N picks, try to have at least one ckpt that
   leans BFCL-strong and at least one that leans FDB-strong. A pure "best on
   everything" sometimes doesn't exist, and picking 3 of the same flavour is
   not useful for downstream selection.
2. **Don't pick on a single metric.** A ckpt that wins only one metric is not a
   Top pick. Combine VB Aggregate, BFCL Priority, FDB v1 TOR Summary, FDB v1.5
   Headline, Conv F1, BBA Headline, latency, and any anomaly checks.
3. **Anomaly filter (FDB v1.5).** Any FDB v1.5 headline `> 1.0` is almost
   certainly a normalization artifact (missing sub-metrics). Exclude from
   "best" — note in `weakness` if you must mention.
4. **BFCL Irrelevance < 70 % is a hard exclusion.** Never pick a ckpt with this
   regression unless there is genuinely no alternative; if you must, the
   `weakness` field must call it out as a critical production blocker.
5. **Latency tie-break.** When two candidates are otherwise comparable, prefer
   the one with lower FDB v1 Combined Latency.
6. **Step trend.** When several ckpts in the same experiment family score
   similarly, prefer the one at the latest step (more training).

For each pick assign `rank` 1, 2, 3, … in order of preference.

## Step 5 — Write the picks file

Write `<output_dir>/<stem-of-data_file>_top<N>.json` with this exact schema
(`top_N` MUST equal `--N`, and `picks` MUST have `--N` entries):

```json
{
  "top_N": 3,
  "metrics_xlsx": "asset/test.xlsx",
  "audio_judgments": {
    "model": null,
    "scope": ["fdb_v1", "fdb_v1_5"],
    "judged_at": null,
    "max_per_split": 40,
    "by_ckpt": {}
  },
  "picks": [
    {
      "rank": 1,
      "short_label": "e174@30k",
      "full_name": "e174-step30011-tts-eartts-no_bw_reb_56k_12ksteps_cs_oci_CA-0.6b-PK_ifg_gsc_ami_os_lr2_wp5",
      "subtitle": "Best holistic — strong across BFCL, FDB v1.5, and FDB v1 TOR.",
      "insight": "Top-tier BFCL Priority (71.5) combined with strong FDB v1.5 headline (0.62) and competitive FDB v1 TOR (85.6).",
      "weakness": "VoiceBench Aggregate slightly below leaders; latency higher than e175."
    },
    {
      "rank": 2,
      "short_label": "e182@15k*",
      "full_name": "e182-step15605-tts-eartts-fp32_24ksteps_rnnt_cand4_preproc_enc_att0_vci50",
      "subtitle": "Best BFCL Priority.",
      "insight": "Top-1 BFCL Priority (71.9 %); strong BFCL Simple/Multiple balance.",
      "weakness": "VoiceBench Aggregate rank 34/36; BBA rank 28/36; latency among the slowest."
    },
    {
      "rank": 3,
      "short_label": "e181@20k",
      "full_name": "e181-step20407-tts-eartts-fp32_24ksteps_rnnt_cand4_preproc_enc_att0_vci50",
      "subtitle": "Best FDB v1 conversational dynamics.",
      "insight": "Top-1 FDB v1 TOR Summary (90.4); strong Adherence (4.22/5).",
      "weakness": "FDB v1.5 rank 30/36; BBA rank 30/36; latency among the slowest."
    }
  ]
}
```

Required fields per pick:

| Field | Type | Description |
|---|---|---|
| `rank` | int | 1-indexed preference order. |
| `short_label` | str | `eXXX@Yk` where Y = step // 1000. Append `*`, `^`, `+`, … only when two ckpts have the same `eXXX@Yk` base; first occurrence keeps the bare label. |
| `full_name` | str | The exact `ckpt` value from the xlsx row (also the sidecar's `name`). |
| `subtitle` | str | One-line characterization (≤ ~80 chars). |
| `insight` | str | 2-3 sentences explaining the ckpt's strengths and why it's recommended. Quote actual metric values. |
| `weakness` | str | 1-2 sentences listing the trade-off / warning. If any hard-filter rule was relaxed (e.g. BFCL Irrelevance < 70 %), call it out here as critical. |

The top-level `audio_judgments` block stores LLM-as-judge verdicts for the
audio examples whose rule-based status is unreliable (currently
`fdb_v1` and `fdb_v1_5`, where the sidecar status is just "spoke vs silent").
Write it as the empty placeholder shown above; Step 6 populates `by_ckpt`.

The visualizer (`scripts/visualize_metrics.py`) consumes this block in
`--html_mode report`: examples with an entry under `by_ckpt[<full_name>]`
keyed by `<bench>:<example_key>` use the LLM verdict; examples without one
keep their rule-based label but render a "low conf" pill in the HTML so the
viewer knows the status is fallback.

## Step 6 — LLM-as-judge for audio examples (best-effort)

Run the audio judge script against the picks file:

```bash
python3 scripts/judge_audio_examples.py --picks <picks_path>
```

Behaviour:

- If `OPENAI_API_KEY` is not exported, the script exits non-zero with a clear
  message. Treat this as a soft warning — proceed to Step 7 and note that
  audio judgments were not generated; the visualizer will mark every FDB v1 /
  v1.5 example as "low conf" on the fallback path.
- If the script runs successfully, it rewrites the picks JSON in place,
  populating `audio_judgments.model`, `audio_judgments.judged_at`, and
  `audio_judgments.by_ckpt` with per-(ckpt, example) verdicts.
- The script only judges the top-N picks (one sidecar per pick) and caps the
  call volume via `--max_per_split` (default 40). The judge uses ASR
  transcripts only — no audio is sent to the model.

## Step 7 — Validation and concise summary (NON-INTERACTIVE)

After Steps 5 and 6:

1. **Validate silently.** Confirm: number of picks equals `--N`, every `rank`
   is unique, every `full_name` exists in the xlsx's `ckpt` column, and (for
   fast follow-up) the `BFCL Irrelevance` value of every pick is ≥ 70 % unless
   explicitly noted as a relaxation in `weakness`. If validation fails, print
   the error and exit; do not ask the user how to proceed.
2. **Print a concise summary**, then exit. The summary MUST be:
   - One short paragraph (≤ 3 sentences) stating which xlsx was used, the
     output path, the top-line rationale for the picks, and whether the LLM
     audio judge ran or was skipped (state why, e.g. "OPENAI_API_KEY not set").
   - Then one line per pick of the form
     `  #<rank>  <short_label>  —  <subtitle>`.
   - Then one line stating the next command to run, e.g.
     `Next: python3 scripts/visualize_metrics.py --name <stem> --html_mode report --top_N <N>`.
3. **Do NOT** ask "is this OK?", "shall I proceed?", "should I continue?",
   use AskUserQuestion, or otherwise wait for confirmation. The picks file is
   already written; the user can simply edit the picks JSON manually before
   re-running the visualizer if they want different picks.

## Common pitfalls

- **Do not use the per-bench `headline.greedy` from `vb_nonmcq` / `vb_mcq`
  sheets as the VoiceBench headline.** Use the `vb_aggregate` sheet's
  `aggregate` column instead — it's the only one that follows the tracker's AJ
  formula.
- **Do not include AlpacaEval-Speaker in the VoiceBench aggregate.** It is not
  part of the tracker formula.
- **Do not zero-impute missing metrics.** A `None` cell means "not run" — skip
  that metric in the composite, do not treat as `0`.
- **Do not pick more than `--N` checkpoints.** If you genuinely cannot decide
  between two candidates for the last slot, pick the one with the better
  BFCL Priority and note the trade-off in `weakness`. Do not ask the user.
- **Do not pause for user input at any step.** This skill is fully autonomous.
