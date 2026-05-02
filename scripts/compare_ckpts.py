#!/usr/bin/env python3
"""
Compare S2S FC eval results across multiple checkpoints.

Usage:
  python3 scripts/compare_ckpts.py [JSON ...] [--output PATH]

Positional args: sidecar .json paths (default: asset/*.json, sorted by name)
--output / -o : output HTML path (default: asset/comparison.html)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

COLORS = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149",
    "#a5d6ff", "#7ee787", "#ffa657", "#ff7b72",
    "#79c0ff", "#56d364", "#e3b341", "#ffa198",
]

BENCHMARKS = ["vb_nonmcq", "vb_mcq", "fdb", "bba", "bfcl", "conv_behav"]
BENCH_LABELS = {
    "vb_nonmcq":  "VB nonMCQ",
    "vb_mcq":     "VB MCQ",
    "fdb":        "FDB",
    "bba":        "BBA",
    "bfcl":       "BFCL",
    "conv_behav": "conv_behav",
}
BENCH_THRESHOLDS = {
    "vb_nonmcq":  (60, 40),
    "vb_mcq":     (60, 40),
    "fdb":        (60, 40),
    "bba":        (70, 50),
    "bfcl":       (70, 40),
    "conv_behav": (75, 55),
}
BENCH_SUBTESTS = {
    "vb_nonmcq":  ["sd_qa", "alpacaeval", "alpacaeval_full", "commoneval",
                   "wildvoice", "alpacaeval_speaker", "ifeval", "advbench"],
    "vb_mcq":     ["bbh", "openbookqa", "mmsu"],
    "fdb":        ["backchannel", "interruption", "pause_candor", "pause_synthetic", "turn_taking"],
    "bba":        ["formal_fallacies", "navigate", "object_counting", "web_of_lies"],
    "bfcl":       ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"],
    "conv_behav": ["tt_f1", "tt_precision", "tt_recall", "barge_in_success_rate",
                   "bc_accuracy", "cutoff_rate", "tt_latency_ms", "barge_in_latency_ms"],
}
RAW_METRICS = {"tt_latency_ms", "barge_in_latency_ms", "num_evaluated"}  # not percentages

AUDIO_CATS = [
    ("vb_commoneval",    "VoiceBench CommonEval — Response Quality"),
    ("fdb_turn_taking",  "FDB — Turn-Taking"),
    ("fdb_pause_candor", "FDB — Pause-Candor"),
    ("bba_navigate",     "BBA Navigate — Correct vs Wrong"),
    ("conv_behav",       "conv_behav — Agent Sessions"),
]


def color_val(bench, val):
    if val is None:
        return "var(--tx2)"
    hi, lo = BENCH_THRESHOLDS[bench]
    return "var(--gr)" if val >= hi else ("var(--ye)" if val >= lo else "var(--re)")


def headline(ckpt, bench):
    h = (ckpt.get("headlines") or {}).get(bench) or {}
    g, s = h.get("greedy"), h.get("sampling")
    return (g if g is not None else s), g, s


def fmt_val(bench, sub, v):
    if v is None:
        return "—"
    if bench == "conv_behav" and sub in RAW_METRICS:
        return f"{v:.0f}"
    return f"{v:.1f}%"


def _js(obj):
    return json.dumps(obj)


def summary_table(ckpts):
    hdrs = "".join(
        f'<th style="color:{COLORS[i % len(COLORS)]};white-space:nowrap">'
        f'{ck["name"]}<br><small style="font-weight:400;color:var(--tx2)">'
        f'{ck.get("commit", "")}</small></th>'
        for i, ck in enumerate(ckpts)
    )
    rows = ""
    for bench in BENCHMARKS:
        cells = ""
        for ck in ckpts:
            best, g, s = headline(ck, bench)
            col = color_val(bench, best)
            if g is not None:
                txt = f"{g:.1f}%"
                if s is not None:
                    txt += f'<span style="color:var(--tx2);font-size:0.8em"> / {s:.1f}%</span>'
            elif s is not None:
                txt = f'<span style="color:var(--tx2)">({s:.1f}%)</span>'
            else:
                txt = "—"
            cells += f'<td style="text-align:center;color:{col};font-weight:700">{txt}</td>'
        rows += f"<tr><td><strong>{BENCH_LABELS[bench]}</strong></td>{cells}</tr>"
    return (
        f'<div style="overflow-x:auto"><table class="cmp-table">'
        f'<thead><tr><th>Benchmark (G / S)</th>{hdrs}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def radar_data(ckpts):
    datasets = []
    for i, ck in enumerate(ckpts):
        vals = [round(headline(ck, b)[0], 1) if headline(ck, b)[0] is not None else 0
                for b in BENCHMARKS]
        c = COLORS[i % len(COLORS)]
        datasets.append({
            "label": ck["name"],
            "data": vals,
            "backgroundColor": c + "33",
            "borderColor": c,
            "borderWidth": 2,
            "pointBackgroundColor": c,
        })
    return _js({"labels": [BENCH_LABELS[b] for b in BENCHMARKS], "datasets": datasets})


def bar_charts(ckpts):
    html_parts, js_parts = [], []
    for bench in BENCHMARKS:
        cid = f"bar_{bench}"
        labels = [ck["name"] for ck in ckpts]
        g_vals, s_vals, bg_g, bg_s, bd_g, bd_s = [], [], [], [], [], []
        for i, ck in enumerate(ckpts):
            c = COLORS[i % len(COLORS)]
            _, g, s = headline(ck, bench)
            g_vals.append(round(g, 1) if g is not None else None)
            s_vals.append(round(s, 1) if s is not None else None)
            bg_g.append(c + "cc"); bg_s.append(c + "44")
            bd_g.append(c);        bd_s.append(c + "99")
        chart_data = {
            "labels": labels,
            "datasets": [
                {"label": "Greedy",   "data": g_vals, "backgroundColor": bg_g,
                 "borderColor": bd_g, "borderWidth": 1},
                {"label": "Sampling", "data": s_vals, "backgroundColor": bg_s,
                 "borderColor": bd_s, "borderWidth": 1},
            ],
        }
        html_parts.append(
            f'<div class="chart-wrap">'
            f'<h3 style="color:var(--ac);margin-bottom:8px">{BENCH_LABELS[bench]}</h3>'
            f'<canvas id="{cid}" height="100"></canvas></div>'
        )
        js_parts.append(
            f"new Chart(document.getElementById('{cid}'),{{"
            f"type:'bar',data:{_js(chart_data)},"
            f"options:{{responsive:true,"
            f"plugins:{{legend:{{labels:{{color:'#e6edf3'}}}}}},"
            f"scales:{{x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},"
            f"y:{{min:0,max:100,ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}}}})"
        )
    bar_grid = f'<div class="bar-grid">{"".join(html_parts)}</div>'
    return bar_grid, "\n".join(js_parts)


def detailed_tables(ckpts):
    parts = ["<h2>Detailed Metrics</h2>"]
    for bench in BENCHMARKS:
        subtests = BENCH_SUBTESTS.get(bench, [])
        col_hdrs = "".join(
            f'<th colspan="2" style="color:{COLORS[i % len(COLORS)]};text-align:center">'
            f'{ck["name"]}</th>'
            for i, ck in enumerate(ckpts)
        )
        mode_hdrs = "".join(
            '<th style="color:#79c0ff;font-size:0.78em;text-align:center">G</th>'
            '<th style="color:#f0b429;font-size:0.78em;text-align:center">S</th>'
            for _ in ckpts
        )
        rows = ""
        for sub in subtests:
            cells = ""
            for ck in ckpts:
                det = (ck.get("detailed") or {}).get(bench) or {}
                for mode in ("greedy", "sampling"):
                    md = det.get(mode) or {}
                    v = md.get(sub) if isinstance(md, dict) else None
                    if v is None:
                        cells += '<td style="color:var(--tx2);text-align:center">—</td>'
                    else:
                        cells += f'<td style="text-align:center;font-weight:600">{fmt_val(bench, sub, v)}</td>'
            rows += f'<tr><td style="white-space:nowrap">{sub}</td>{cells}</tr>'
        parts.append(
            f'<h3 style="color:var(--ac);margin:16px 0 6px">{BENCH_LABELS[bench]}</h3>'
            f'<div style="overflow-x:auto;margin-bottom:8px"><table class="cmp-table">'
            f'<thead><tr><th>Metric</th>{col_hdrs}</tr>'
            f'<tr><th></th>{mode_hdrs}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )
    return "\n".join(parts)


def _ex_header(cat, ex):
    if cat == "vb_commoneval":
        sc = ex.get("score")
        q = ex.get("quality", "")
        cs = {"good": "var(--gr)", "medium": "var(--ye)", "bad": "var(--re)"}.get(q, "var(--tx2)")
        badge = (f'<span class="badge" style="background:{cs}20;color:{cs};border:1px solid {cs}">'
                 f'GPT {sc:.1f}/5</span>') if sc is not None else ""
        return f"<strong>VoiceBench CommonEval</strong> {badge}"
    if cat == "fdb_turn_taking":
        sid = ex.get("sample_id", ex.get("key", ""))
        good = ex.get("outcome") == "took_turn"
        c = "var(--gr)" if good else "var(--re)"
        lbl = "TOOK TURN ✓" if good else "MISSED TURN ✗"
        return (f'<strong>FDB Turn-Taking</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' <code style="font-size:0.78em">{sid}</code>')
    if cat == "fdb_pause_candor":
        sid = ex.get("sample_id", ex.get("key", ""))
        good = ex.get("outcome") == "silent"
        c = "var(--gr)" if good else "var(--re)"
        lbl = "SILENT ✓" if good else "INTERRUPTED ✗"
        return (f'<strong>FDB Pause-Candor</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' <code style="font-size:0.78em">{sid}</code>')
    if cat == "bba_navigate":
        correct = ex.get("correct", False)
        expected = ex.get("expected", "")
        c = "var(--gr)" if correct else "var(--re)"
        lbl = "CORRECT ✓" if correct else "WRONG ✗"
        return (f'<strong>BBA Navigate</strong> '
                f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}">{lbl}</span>'
                f' expected: <em>{expected}</em>')
    if cat == "conv_behav":
        fname = ex.get("filename", ex.get("key", ""))
        return f'<strong>conv_behav</strong> <code style="font-size:0.78em">{fname}</code>'
    return "<strong>Example</strong>"


def _ex_context(cat, ex):
    if cat == "vb_commoneval":
        return f'<div class="bubble bubble-q"><strong>Q:</strong> {ex.get("question", "")}</div>'
    if cat == "fdb_turn_taking":
        return '<div class="bubble bubble-q"><strong>Input audio →</strong></div>'
    if cat == "fdb_pause_candor":
        ctx = (ex.get("context") or "")[:150]
        return f'<div class="bubble bubble-q"><strong>User context:</strong><br>{ctx}…</div>'
    if cat == "bba_navigate":
        return '<div class="bubble bubble-q"><strong>Category:</strong> navigate</div>'
    if cat == "conv_behav":
        return '<div class="bubble bubble-q">Full-duplex conversation session recording.</div>'
    return ""


def _audio_row(cat, ckpt_name, color, ex):
    badge = (f'<span class="ckpt-badge" style="background:{color}22;color:{color};'
             f'border:1px solid {color}">{ckpt_name}</span>')
    src = (ex or {}).get("audio_src")
    audio = (f'<audio controls src="{src}" class="aplayer"></audio>'
             if src else '<em style="color:var(--tx2);font-size:0.82em">audio not available</em>')
    resp = ""
    if ex:
        if cat == "vb_commoneval":
            r = (ex.get("response") or "")[:200]
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px"><strong>A:</strong> {r}…</div>'
        elif cat in ("fdb_turn_taking", "fdb_pause_candor"):
            g = (ex.get("generation") or "").strip() or "(silent)"
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px">{g[:150]}</div>'
        elif cat == "bba_navigate":
            r = (ex.get("response") or "")[:150]
            resp = f'<div class="bubble bubble-a" style="margin-bottom:4px"><strong>R:</strong> {r}</div>'
    return f'<div class="audio-row">{badge}{resp}{audio}</div>'


def audio_section(ckpts):
    parts = [
        "<h2>Audio Comparison</h2>",
        '<p style="color:var(--tx2);font-size:0.88em;margin-bottom:16px">'
        "Representative outputs across checkpoints. Audio requires the HTTP server — "
        "run <code>bash scripts/serve_scorecard.sh --name comparison</code>.</p>",
    ]
    for cat, cat_label in AUDIO_CATS:
        ckpt_maps = [
            {e.get("key", str(i)): e for i, e in enumerate((ck.get("audio_examples") or {}).get(cat) or [])}
            for ck in ckpts
        ]
        all_keys = []
        seen = set()
        for cm in ckpt_maps:
            for k in cm:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        if not all_keys:
            continue
        parts.append(f'<div class="ex-section"><h3 style="color:var(--ac)">{cat_label}</h3>')
        for key in all_keys:
            ctx_ex = next((cm[key] for cm in ckpt_maps if key in cm), None)
            if ctx_ex is None:
                continue
            rows = [
                _audio_row(cat, ck["name"], COLORS[i % len(COLORS)], cm.get(key))
                for i, (ck, cm) in enumerate(zip(ckpts, ckpt_maps))
            ]
            visible = "".join(rows[:3])
            hidden_rows = rows[3:]
            right = visible
            if hidden_rows:
                right += (
                    f'<details class="more-ckpts">'
                    f'<summary>{len(hidden_rows)} more checkpoint(s) — click to expand</summary>'
                    f'{"".join(hidden_rows)}</details>'
                )
            parts.append(
                f'<div class="ex-card">'
                f'<div class="ex-header">{_ex_header(cat, ctx_ex)}</div>'
                f'<div class="ex-body-cmp">'
                f'<div class="ex-left">{_ex_context(cat, ctx_ex)}</div>'
                f'<div class="ex-right-cmp">{right}</div>'
                f'</div></div>'
            )
        parts.append("</div>")
    return "\n".join(parts)


CSS = """
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
        --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950;
        --ye:#d29922; --re:#f85149; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--tx);
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        line-height:1.5; padding:24px; max-width:1280px; margin:0 auto; }
a { color:var(--ac); }
h1 { font-size:1.7em; margin-bottom:6px; }
h2 { font-size:1.1em; font-weight:600; color:var(--ac); margin:24px 0 10px;
      border-bottom:1px solid var(--bd); padding-bottom:6px; }
h3 { font-size:0.95em; font-weight:600; margin-bottom:10px; }
.meta-row { display:flex; gap:20px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
.meta-item { color:var(--tx2); font-size:0.88em; }
.meta-item strong { color:var(--tx); }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.ckpt-pill { display:inline-block; padding:4px 14px; border-radius:20px;
              font-size:0.85em; font-weight:600; }
.cmp-table { width:100%; border-collapse:collapse; font-size:0.87em; }
.cmp-table th, .cmp-table td { padding:7px 12px; border:1px solid var(--bd); }
.cmp-table thead th { background:var(--bg3); }
.cmp-table tbody tr:hover { background:var(--bg2); }
.chart-wrap { background:var(--bg2); border:1px solid var(--bd); border-radius:10px;
               padding:20px; margin:16px 0; }
.bar-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0; }
.aplayer { width:100%; height:34px; margin-top:4px; border-radius:4px; accent-color:var(--ac); }
.ex-section { margin:16px 0; }
.ex-card { background:var(--bg2); border:1px solid var(--bd); border-radius:10px;
            margin-bottom:12px; overflow:hidden; }
.ex-header { background:var(--bg3); padding:10px 16px; font-size:0.88em;
              border-bottom:1px solid var(--bd); display:flex; align-items:center;
              gap:8px; flex-wrap:wrap; }
.ex-body-cmp { display:grid; grid-template-columns:260px 1fr; }
.ex-left  { padding:14px 16px; border-right:1px solid var(--bd); }
.ex-right-cmp { padding:14px 16px; }
.audio-row { margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid var(--bg3); }
.audio-row:last-child { border-bottom:none; margin-bottom:0; }
.ckpt-badge { display:inline-block; padding:2px 8px; border-radius:4px;
               font-size:0.78em; font-weight:600; margin-bottom:4px; }
.bubble { border-radius:6px; padding:10px 12px; font-size:0.87em;
           margin-bottom:6px; line-height:1.5; }
.bubble:last-child { margin-bottom:0; }
.bubble-q { background:#0d1f3a; border-left:3px solid var(--ac); }
.bubble-a { background:#0d2a1a; border-left:3px solid var(--gr); }
.badge { display:inline-block; padding:2px 8px; border-radius:4px;
          font-size:0.8em; font-weight:600; }
details.more-ckpts summary { cursor:pointer; color:var(--ac); font-size:0.85em;
                               padding:6px 0; user-select:none; }
details.more-ckpts summary:hover { text-decoration:underline; }
@media(max-width:900px) {
  .bar-grid { grid-template-columns:1fr; }
  .ex-body-cmp { grid-template-columns:1fr; }
  .ex-left { border-right:none; border-bottom:1px solid var(--bd); }
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", nargs="*", help="Sidecar JSON files (default: asset/*.json)")
    p.add_argument("--output", "-o", default=None, help="Output HTML path")
    args = p.parse_args()

    paths = [Path(x) for x in args.jsons] if args.jsons else sorted(Path("asset").glob("*.json"))
    if not paths:
        sys.exit("No sidecar JSON files found. Run /make-scorecard first to generate them.")

    ckpts = []
    for path in paths:
        with open(path) as f:
            ck = json.load(f)
        ck.setdefault("_path", str(path))
        ckpts.append(ck)

    print(f"Loaded {len(ckpts)} checkpoint(s): {[c['name'] for c in ckpts]}")

    out = Path(args.output) if args.output else Path("asset/comparison.html")
    today = date.today().isoformat()

    legend_html = "".join(
        f'<span class="ckpt-pill" style="background:{COLORS[i % len(COLORS)]}22;'
        f'color:{COLORS[i % len(COLORS)]};border:1px solid {COLORS[i % len(COLORS)]}">'
        f'{ck["name"]} <span style="font-weight:400;opacity:0.65">({ck.get("commit", "?")})</span>'
        f'</span>'
        for i, ck in enumerate(ckpts)
    )

    bar_grid, bar_js = bar_charts(ckpts)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2S FC Eval — Checkpoint Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<h1>S2S FC Eval — Checkpoint Comparison</h1>
<div class="meta-row">
  <span class="meta-item">Generated: <strong>{today}</strong></span>
  <span class="meta-item">Checkpoints: <strong>{len(ckpts)}</strong></span>
</div>
<div class="legend">{legend_html}</div>

<h2>Summary</h2>
{summary_table(ckpts)}

<h2>Radar Overview</h2>
<div class="chart-wrap" style="max-width:560px;margin:16px auto">
  <canvas id="radarChart" height="400"></canvas>
</div>

<h2>Per-Benchmark Scores</h2>
{bar_grid}

{detailed_tables(ckpts)}

{audio_section(ckpts)}

<script>
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {radar_data(ckpts)},
  options: {{
    responsive: true,
    scales: {{ r: {{
      min: 0, max: 100,
      ticks: {{ color:'#8b949e', backdropColor:'transparent', stepSize:20 }},
      grid: {{ color:'#30363d' }},
      pointLabels: {{ color:'#e6edf3', font:{{ size:13 }} }}
    }} }},
    plugins: {{ legend: {{ labels: {{ color:'#e6edf3' }} }} }}
  }}
}});
{bar_js}
</script>
</body>
</html>"""

    out.write_text(page)
    print(f"Written: {out}")
    print(f"View with: bash scripts/serve_scorecard.sh --name {out.stem}")


if __name__ == "__main__":
    main()
