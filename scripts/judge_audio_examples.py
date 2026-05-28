"""LLM-as-judge labels for audio examples in a top-N picks JSON.

For each pick, read its sidecar JSON, find FDB v1 / v1.5 examples (the
datasets whose `status` is currently rule-based "spoke vs silent"), ask an
LLM to re-judge them from the ASR transcript, and write the verdicts back
into the picks JSON as an `audio_judgments` block.

Visualize_metrics consumes that block: examples with an LLM judgment use it
(high confidence); examples without one keep their rule-based label but get
a "low confidence" pill in the rendered HTML.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SCOPE = ["fdb_v1", "fdb_v1_5"]
DEFAULT_MODEL = "gpt-4o-mini"

SPLIT_RUBRIC = {
    "turn_taking": (
        "User finished speaking and yielded the floor. The agent should "
        "respond verbally. 'good' = the agent's transcript is a coherent "
        "on-topic spoken response; 'bad' = silence, filler-only, or off-topic; "
        "'ok' = response present but partially off-topic or very short."
    ),
    "interruption": (
        "User cut in while the agent was speaking. The agent should yield "
        "and respond to the interruption. 'good' = transcript shows the "
        "agent acknowledging/responding to the interruption; 'bad' = the "
        "agent ignores it and continues its prior turn; 'ok' = partial yield."
    ),
    "backchannel": (
        "User produced a short backchannel (mm-hmm / yeah). The agent "
        "should NOT take over the turn. 'good' = silence or a very short "
        "acknowledgement (<=2 words); 'bad' = full takeover / launches a "
        "new topic; 'ok' = brief but slightly overlong response."
    ),
    "pause_candor": (
        "User paused mid-utterance (natural Candor pause). The agent should "
        "stay silent and let the user continue. 'good' = silent; 'bad' = "
        "agent interrupts and takes over; 'ok' = very brief filler."
    ),
    "pause_synthetic": (
        "User paused mid-utterance (synthetic pause). The agent should stay "
        "silent and wait for the user to continue. 'good' = silent; "
        "'bad' = agent interrupts; 'ok' = brief filler."
    ),
    "background_speech": (
        "Background speech audible but not addressed to the agent. The "
        "agent should NOT respond. 'good' = silent / does not respond; "
        "'bad' = agent treats it as a query and responds; 'ok' = ambiguous."
    ),
    "talking_to_other": (
        "User is talking to a third party, not to the agent. The agent "
        "should NOT respond. 'good' = silent; 'bad' = agent jumps in; "
        "'ok' = ambiguous response."
    ),
}

SYSTEM = (
    "You are an evaluator for a voice agent. Given the rubric for one "
    "subtest split, the user/context, and the agent's ASR-transcribed "
    "response, decide whether the agent behaved correctly. Respond as "
    "compact JSON: {\"status\":\"good|bad|ok\",\"reason\":\"<=18 words\"}."
)


def find_sidecar(picks_path: Path, full_name: str) -> Path | None:
    stem = picks_path.stem.removesuffix(f"_top{picks_path.stem.split('_top')[-1]}") if "_top" in picks_path.stem else picks_path.stem
    cand = picks_path.with_name(f"{stem}_{full_name}.json")
    if cand.exists():
        return cand
    return None


def llm_judge(client, model: str, bench: str, split: str, example: dict[str, Any]) -> dict[str, str] | None:
    rubric = SPLIT_RUBRIC.get(split)
    if not rubric:
        return None
    context = str(example.get("question") or example.get("context") or "").strip()[:600]
    response = str(example.get("response") or "").strip()[:600]
    word_count = example.get("word_count")
    extra = f"\nAgent word count: {word_count}" if word_count is not None else ""
    user = (
        f"Subtest: {bench} / {split}\n"
        f"Rubric: {rubric}\n"
        f"Context (ASR / dialogue): {context}\n"
        f"Agent response (ASR): {response or '(silence)'}"
        f"{extra}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  judge error: {exc}", file=sys.stderr)
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    status = str(parsed.get("status") or "").lower()
    if status not in {"good", "bad", "ok"}:
        return None
    reason = str(parsed.get("reason") or "").strip()[:160]
    return {"status": status, "reason": reason}


def collect_examples(sidecar: dict[str, Any], scope: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {bench: {example_key_with_bench_prefix: example_dict}} for in-scope benches."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for group, examples in (sidecar.get("audio_examples") or {}).items():
        bench, _, split = group.partition(":")
        if bench not in scope:
            continue
        for ex in examples or []:
            key = ex.get("key")
            if not key:
                continue
            joined = f"{bench}:{key}"
            out.setdefault(bench, {})[joined] = ex
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", required=True, type=Path, help="path to <name>_top<N>.json")
    ap.add_argument("--scope", default=",".join(DEFAULT_SCOPE),
                    help="comma-separated bench names to LLM-judge (default: fdb_v1,fdb_v1_5)")
    ap.add_argument("--model", default=os.environ.get("VISUALIZE_METRICS_LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max_per_split", type=int, default=40,
                    help="cap on examples per (ckpt, split) to judge (default 40)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; rerun once the key is exported.")
    try:
        from openai import OpenAI
    except Exception as exc:
        sys.exit(f"openai package import failed: {exc}")

    picks_doc = json.loads(args.picks.read_text(encoding="utf-8"))
    picks = picks_doc.get("picks") or []
    if not picks:
        sys.exit("picks JSON has no picks[] entries")

    scope = [s.strip() for s in args.scope.split(",") if s.strip()]
    client = OpenAI()
    by_ckpt: dict[str, dict[str, dict[str, str]]] = {}

    t0 = time.time()
    total_calls = 0
    for pick in picks:
        full = pick.get("full_name") or ""
        sidecar_path = find_sidecar(args.picks, full)
        if sidecar_path is None:
            print(f"[skip] no sidecar for {full}", file=sys.stderr)
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        in_scope = collect_examples(sidecar, scope)
        ckpt_judgments: dict[str, dict[str, str]] = {}

        per_split_counts: dict[str, int] = {}
        for bench, ex_map in in_scope.items():
            for joined_key, ex in ex_map.items():
                split = joined_key.split(":", 2)[1]
                cap_key = f"{bench}:{split}"
                if per_split_counts.get(cap_key, 0) >= args.max_per_split:
                    continue
                per_split_counts[cap_key] = per_split_counts.get(cap_key, 0) + 1
                verdict = llm_judge(client, args.model, bench, split, ex)
                total_calls += 1
                if verdict is not None:
                    ckpt_judgments[joined_key] = verdict
        by_ckpt[full] = ckpt_judgments
        print(f"[done] {full}: {len(ckpt_judgments)} judgments")

    picks_doc["audio_judgments"] = {
        "model": args.model,
        "scope": scope,
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_per_split": args.max_per_split,
        "by_ckpt": by_ckpt,
    }
    args.picks.write_text(json.dumps(picks_doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote audio_judgments to {args.picks} ({total_calls} LLM calls, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
