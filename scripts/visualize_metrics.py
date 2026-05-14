#!/usr/bin/env python3
"""Visualize S2S FC evaluation metrics for one or more checkpoints.

Examples:
  python3 scripts/visualize_metrics.py --metrics_dirs /path/to/run --name visual
  python3 scripts/visualize_metrics.py --metrics_dirs /path/to/ckpt_a /path/to/ckpt_b --name comparison

Each metrics directory is expected to be either:
  - a run_all_benchmarks output root containing benchmark subdirectories such as
    vb_nonmcq_<commit>, fdb_v1_<commit>, bba_<commit>, ...
  - a single benchmark output directory containing eval-results/
  - an existing sidecar JSON produced by this script or the older scorecard flow

The script writes one HTML file and one sidecar JSON per metrics directory.
When multiple benchmark folders with the same dataset name are found under a
metrics directory, the latest folder is selected by git commit history if all
candidate commits exist in this repo; otherwise folder modification time is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path
from typing import Any


COLORS = [
    "#58a6ff",
    "#3fb950",
    "#d29922",
    "#f85149",
    "#a5d6ff",
    "#7ee787",
    "#ffa657",
    "#ff7b72",
    "#79c0ff",
    "#56d364",
    "#e3b341",
    "#ffa198",
]

KNOWN_MODES = ("greedy", "sampling")
SERVE_ROOT = Path("/lustre")
TOKEN_RE = re.compile(r"<\$[\d.]+\$>|<\|[\d.]+\|>|<[^>]+>")
SPEECH_TOKEN_RE = re.compile(r"<\|[\d.]+\|>")

VB_NONMCQ_SPLITS = [
    "sd_qa",
    "alpacaeval_full",
    "alpacaeval",
    "ifeval",
    "advbench",
    "commoneval",
    "wildvoice",
    "alpacaeval_speaker",
]
VB_MCQ_SPLITS = ["bbh", "openbookqa", "mmsu"]
FDB_V1_SPLITS = ["backchannel", "interruption", "pause_candor", "pause_synthetic", "turn_taking"]
FDB_V1_5_SPLITS = ["background_speech", "talking_to_other", "backchannel", "interruption"]
BBA_SPLITS = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]
BFCL_SPLITS = ["simple", "parallel", "multiple", "parallel_multiple", "irrelevance"]

BENCHMARKS = [
    "vb_nonmcq",
    "vb_mcq",
    "fdb_v1",
    "fdb_v1_5",
    "fdb_v3",
    "bba",
    "bfcl",
    "conv_behav",
]
BENCH_LABELS = {
    "vb_nonmcq": "VoiceBench non-MCQ",
    "vb_mcq": "VoiceBench MCQ",
    "fdb_v1": "FDB v1",
    "fdb_v1_5": "FDB v1.5",
    "fdb_v3": "FDB v3",
    "bba": "BigBench Audio",
    "bfcl": "BFCL",
    "conv_behav": "Turn Taking Benchmark",
}
BENCH_SPLITS = {
    "vb_nonmcq": VB_NONMCQ_SPLITS,
    "vb_mcq": VB_MCQ_SPLITS,
    "fdb_v1": FDB_V1_SPLITS,
    "fdb_v1_5": FDB_V1_5_SPLITS,
    "fdb_v3": ["tool_call"],
    "bba": BBA_SPLITS,
    "bfcl": BFCL_SPLITS,
    "conv_behav": ["overall"],
}
BENCH_PREFIXES = {
    "vb_nonmcq": ["vb_nonmcq"],
    "vb_mcq": ["vb_mcq"],
    "fdb_v1": ["fdb_v1", "fdb"],
    "fdb_v1_5": ["fdb_v1_5"],
    "fdb_v3": ["fdb_v3"],
    "bba": ["bba"],
    "bfcl": ["bfcl"],
    "conv_behav": ["conv_behav"],
}
BENCH_THRESHOLDS = {
    "vb_nonmcq": (60, 40),
    "vb_mcq": (60, 40),
    "fdb_v1": (60, 40),
    "fdb_v1_5": (60, 40),
    "fdb_v3": (70, 50),
    "bba": (70, 50),
    "bfcl": (70, 40),
    "conv_behav": (75, 55),
}

COUNT_KEYS = {"total", "num_samples", "num_correct", "num_evaluated", "total_scenarios", "num_result_files"}
MS_RE = re.compile(r"(^|[._-])(latency|duration|time).*(_ms|ms)$|_ms$")
PERCENT_KEY_RE = re.compile(
    r"(acc|accuracy|rate|f1|precision|recall|pct|percentage|tor_pct|wer|cer|panda|cutoff|refusal|pass|final|strict|loose)"
)
SCORE5_KEYS = {"gpt", "rating"}

DEFAULT_AUDIO_LABELS = {
    "vb_nonmcq:commoneval": "VoiceBench CommonEval - response quality",
    "fdb_v1:turn_taking": "FDB v1 turn-taking",
    "fdb_v1:pause_candor": "FDB v1 pause handling - Candor",
    "bba:navigate": "BBA navigate",
    "conv_behav:overall": "Turn Taking Benchmark sessions",
    # Older sidecar compatibility.
    "vb_commoneval": "VoiceBench CommonEval - response quality",
    "fdb_turn_taking": "FDB turn-taking",
    "fdb_pause_candor": "FDB pause handling - Candor",
    "bba_navigate": "BBA navigate",
    "conv_behav": "Turn Taking Benchmark sessions",
}


def h(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def cluster_name() -> str:
    return (
        os.environ.get("SLURM_CLUSTER_NAME")
        or os.environ.get("CLUSTER_NAME")
        or os.environ.get("HOSTNAME")
        or os.uname().nodename
    ).split(".", 1)[0]


def cluster_path(path: Path | str | None) -> str:
    if not path:
        return ""
    return f"{cluster_name()}:{Path(path).resolve()}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def as_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    number = as_number(value)
    if number is not None:
        out[prefix or "value"] = number
        return out
    if isinstance(value, dict):
        for key, subval in value.items():
            if not isinstance(key, str):
                continue
            subprefix = f"{prefix}.{key}" if prefix else key
            out.update(numeric_leaves(subval, subprefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            subprefix = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(numeric_leaves(item, subprefix))
    return out


def infer_mode(path: Path) -> str:
    name = path.name.lower()
    if "sampling" in name or name == "sample":
        return "sampling"
    if "greedy" in name:
        return "greedy"
    for part in reversed(path.parts):
        token = part.lower()
        if token in {"sampling", "sample"} or token.startswith("sampling_"):
            return "sampling"
        if token == "greedy" or token.startswith("greedy_"):
            return "greedy"
    text = str(path).lower()
    if "sampling" in text and "greedy" not in text:
        return "sampling"
    return "greedy"


def infer_commit(path: Path) -> str:
    matches = re.findall(r"(?<![0-9a-f])([0-9a-f]{7,12})(?![0-9a-f])", str(path), flags=re.I)
    return matches[-1] if matches else ""


_GIT_COMMIT_RANKS: dict[str, int] | None = None


def git_commit_ranks() -> dict[str, int]:
    """Return a map from commit abbreviations to git-log recency rank.

    Rank 0 is HEAD, larger ranks are older. Abbreviations are included because
    benchmark directories typically carry short commit hashes.
    """
    global _GIT_COMMIT_RANKS
    if _GIT_COMMIT_RANKS is not None:
        return _GIT_COMMIT_RANKS

    repo_root = Path(__file__).resolve().parents[1]
    try:
        proc = subprocess.run(
            ["git", "log", "--format=%H"],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        _GIT_COMMIT_RANKS = {}
        return _GIT_COMMIT_RANKS

    ranks: dict[str, int] = {}
    if proc.returncode == 0:
        for idx, full_hash in enumerate(line.strip().lower() for line in proc.stdout.splitlines() if line.strip()):
            ranks.setdefault(full_hash, idx)
            for size in range(7, min(12, len(full_hash)) + 1):
                ranks.setdefault(full_hash[:size], idx)
    _GIT_COMMIT_RANKS = ranks
    return ranks


def dir_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def select_latest_candidate(candidates: list[Path]) -> Path:
    """Pick the latest benchmark directory from duplicate dataset candidates.

    If every candidate commit is present in the current repo history, use git
    history order. If any candidate commit is absent or unavailable, fall back
    to local folder modification time for the whole candidate set.
    """
    if len(candidates) == 1:
        return candidates[0]

    ranks = git_commit_ranks()
    commits = [infer_commit(candidate).lower() for candidate in candidates]
    if commits and all(commit and commit in ranks for commit in commits):
        return sorted(
            candidates,
            key=lambda candidate: (
                ranks[infer_commit(candidate).lower()],
                -dir_mtime(candidate),
                candidate.name,
            ),
        )[0]

    return sorted(candidates, key=lambda candidate: (-dir_mtime(candidate), candidate.name))[0]


def safe_name(text: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = text.strip("._")
    return text or fallback


def display_name_for_source(source: Path) -> str:
    if source.suffix.lower() == ".json":
        return source.stem
    generic_names = {"incremental", "eval-results", "results", "metrics"}
    if source.name.lower() in generic_names and source.parent.name:
        return source.parent.name
    return source.name or str(source)


def output_html_path(name: str) -> Path:
    p = Path(name)
    if p.suffix.lower() != ".html":
        p = p.with_suffix(".html")
    if not p.is_absolute() and p.parent == Path("."):
        p = Path("asset") / p
    return p


def sidecar_paths(out_html: Path, sources: list[Path]) -> list[Path | None]:
    paths: list[Path | None] = []
    used: set[str] = set()
    for idx, source in enumerate(sources, start=1):
        if source.suffix.lower() == ".json":
            paths.append(None)
            continue
        if len(sources) == 1:
            candidate = out_html.with_suffix(".json")
        else:
            stem = safe_name(display_name_for_source(source), f"ckpt_{idx}")
            candidate = out_html.with_name(f"{out_html.stem}_{stem}.json")
        base = candidate
        n = 2
        while str(candidate) in used:
            candidate = base.with_name(f"{base.stem}_{n}{base.suffix}")
            n += 1
        used.add(str(candidate))
        paths.append(candidate)
    return paths


def is_direct_benchmark_dir(path: Path, bench: str) -> bool:
    return any(
        candidate.exists()
        for split in BENCH_SPLITS[bench]
        for candidate in metric_file_candidates(path, bench, split)
    )


def benchmark_name_matches(name: str, bench: str, prefix: str) -> bool:
    if name == prefix:
        return True
    if prefix == "fdb":
        return name.startswith("fdb_") and not name.startswith(("fdb_v1", "fdb_v3"))
    if bench == "fdb_v1" and name.startswith("fdb_v1_5"):
        return False
    return name.startswith(prefix + "_")


def find_benchmark_dir(root: Path, bench: str) -> Path | None:
    if is_direct_benchmark_dir(root, bench):
        return root
    if not root.exists() or not root.is_dir():
        return None
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for prefix in BENCH_PREFIXES[bench]:
            if benchmark_name_matches(child.name, bench, prefix) and is_direct_benchmark_dir(child, bench):
                candidates.append(child)
                break
    if not candidates:
        return None
    return select_latest_candidate(candidates)


def eval_roots(bench_dir: Path) -> list[Path]:
    roots: list[Path] = []
    er = bench_dir / "eval-results"
    if er.exists():
        roots.append(er)
    roots.append(bench_dir)
    deduped: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def result_dir_names(bench: str, split: str) -> list[str]:
    if bench in ("vb_nonmcq", "vb_mcq"):
        return [f"voicebench.{split}"]
    if bench == "fdb_v1":
        return [f"fdb_v1.{split}"]
    if bench == "fdb_v1_5":
        return [f"fdb_v1_5.{split}"]
    if bench == "fdb_v3":
        return ["fdb_v3.tool_call"]
    if bench == "bba":
        return [split, f"bba.{split}"]
    if bench == "bfcl":
        if split == "simple":
            return ["simple", "simple_python", "bfcl_fc.simple", "bfcl_fc.simple_python"]
        return [split, f"bfcl_fc.{split}"]
    if bench == "conv_behav":
        return ["."]
    return [split]


def metric_key_names(bench: str, split: str) -> list[str]:
    if bench in ("vb_nonmcq", "vb_mcq"):
        return [f"voicebench.{split}"]
    if bench == "fdb_v1":
        return [f"fdb_v1.{split}", f"fdb_v1.{split}.greedy", f"fdb_v1.{split}.sampling"]
    if bench == "fdb_v1_5":
        return [f"fdb_v1_5.{split}", f"fdb_v1_5.{split}.greedy", f"fdb_v1_5.{split}.sampling"]
    if bench == "fdb_v3":
        return ["fdb_v3.tool_call"]
    if bench == "bba":
        return [f"bba.{split}"]
    if bench == "bfcl":
        if split == "simple":
            return ["bfcl_fc.simple", "bfcl_fc.simple_python"]
        return [f"bfcl_fc.{split}"]
    if bench == "conv_behav":
        return ["conv_behav"]
    return [split]


def metric_file_candidates(bench_dir: Path, bench: str, split: str) -> list[Path]:
    candidates: list[Path] = []
    for root in eval_roots(bench_dir):
        if bench == "conv_behav":
            candidates.append(root / "metrics.json")
            continue
        for dname in result_dir_names(bench, split):
            candidates.append(root / dname / "metrics.json")
    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def result_dirs_for(bench_dir: Path, bench: str, split: str) -> list[Path]:
    dirs: list[Path] = []
    for root in eval_roots(bench_dir):
        if bench == "conv_behav":
            dirs.append(root)
        else:
            for dname in result_dir_names(bench, split):
                dirs.append(root / dname)
    deduped: list[Path] = []
    seen = set()
    for path in dirs:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def split_metric_modes(value: Any, default_mode: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    modes = {mode: value[mode] for mode in KNOWN_MODES if isinstance(value.get(mode), dict)}
    if modes:
        return modes
    return {default_mode: value}


def extract_metric_modes(raw: dict[str, Any], keys: list[str], default_mode: str) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for key in keys:
        if key in raw:
            modes.update(split_metric_modes(raw[key], default_mode))
        prefix = key + "."
        for raw_key, raw_val in raw.items():
            if not isinstance(raw_key, str) or not raw_key.startswith(prefix):
                continue
            suffix = raw_key[len(prefix) :]
            if suffix in KNOWN_MODES and isinstance(raw_val, dict):
                modes[suffix] = raw_val
    if modes:
        return modes
    # Fallback for slightly different category spellings.
    for raw_key, raw_val in raw.items():
        if not isinstance(raw_val, dict):
            continue
        if any(key in raw_key or raw_key in key for key in keys):
            modes.update(split_metric_modes(raw_val, default_mode))
    return modes


def clean_generation(text: Any) -> str:
    return TOKEN_RE.sub("", str(text or "")).strip()


def has_speech_token(text: Any) -> bool:
    return bool(SPEECH_TOKEN_RE.search(str(text or "")))


def audio_url(abs_path: Path | str | None) -> str | None:
    if not abs_path:
        return None
    path = Path(abs_path)
    if not path.exists():
        return None
    try:
        return "/" + str(path.resolve().relative_to(SERVE_ROOT))
    except ValueError:
        return path.resolve().as_uri()


def audio_path_from_record(eval_dir: Path, record: dict[str, Any]) -> Path | None:
    raw = ""
    audio = record.get("audio")
    if isinstance(audio, dict):
        raw = str(audio.get("path") or "")
    elif audio:
        raw = str(audio)
    candidates: list[Path] = []
    if raw:
        raw_path = Path(raw)
        candidates.append(raw_path)
        if raw_path.name:
            candidates.append(eval_dir / "audio" / raw_path.name)
            candidates.append(eval_dir / raw_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def audio_from_record(eval_dir: Path, record: dict[str, Any]) -> str | None:
    return audio_url(audio_path_from_record(eval_dir, record))


def fdb_prepared_dir(eval_dir: Path, split: str, record: dict[str, Any]) -> Path | None:
    sample_id = str(record.get("sample_id") or "")
    rec_id = str(record.get("id") or "")
    names: list[str] = []
    if rec_id:
        names.append(rec_id)
    if sample_id:
        names.append(sample_id)
    if split == "backchannel":
        m = re.search(r"(\d+)$", rec_id or sample_id)
        if m:
            names.append(m.group(1))
    seen = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        path = eval_dir / "fdb_prepared" / name
        if path.exists():
            return path
    return None


def example_key(record: dict[str, Any], split: str, idx: int) -> str:
    for field in ("audio_path", "id", "sample_id"):
        value = record.get(field)
        if value:
            return f"{split}:{value}"
    problem = str(record.get("problem") or record.get("question_text") or "")[:160]
    if problem:
        return f"{split}:{problem}"
    return f"{split}:idx:{idx}"


def status_from_score(score: float | None) -> str:
    if score is None:
        return "sample"
    if score >= 4.0:
        return "good"
    if score <= 2.0:
        return "bad"
    return "ok"


def normalize_answer(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def answer_correct(expected: Any, response: Any) -> bool | None:
    exp = normalize_answer(expected)
    if not exp:
        return None
    resp = normalize_answer(response)[:120]
    if exp in {"yes", "no"}:
        return resp.startswith(exp)
    return exp in resp


VOICEBENCH_SPLIT_CATEGORIES = {
    "sd_qa": "Short-answer QA",
    "alpacaeval_full": "Open-ended instruction following",
    "alpacaeval": "Open-ended instruction following",
    "alpacaeval_speaker": "Speaker-style instruction following",
    "ifeval": "Instruction-following constraints",
    "advbench": "Safety refusal",
    "commoneval": "Common-sense generation",
    "wildvoice": "Open-domain voice prompts",
    "bbh": "BBH reasoning",
    "openbookqa": "Science QA",
    "mmsu": "Multimodal speech understanding",
}
FDB_SPLIT_CATEGORIES = {
    "backchannel": "Backchannel timing",
    "interruption": "User interruption",
    "pause_candor": "Candor pause handling",
    "pause_synthetic": "Synthetic pause handling",
    "turn_taking": "Turn-taking response",
    "background_speech": "Background speech",
    "talking_to_other": "Talking to others",
}


def titleize_token(text: str) -> str:
    text = re.sub(r"[_./-]+", " ", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    acronyms = {
        "asr": "ASR",
        "bba": "BBA",
        "bbh": "BBH",
        "bfcl": "BFCL",
        "fdb": "FDB",
        "icc": "ICC",
        "mcq": "MCQ",
        "mmsu": "MMSU",
        "qa": "QA",
    }
    return " ".join(acronyms.get(part.lower(), part.capitalize()) for part in text.split())


def direct_record_category(record: dict[str, Any]) -> str:
    for key in ("category", "subcategory", "subset_for_metrics", "task", "question_type", "domain"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return titleize_token(value)
    for meta_key in ("metadata", "meta"):
        meta = record.get(meta_key)
        if not isinstance(meta, dict):
            continue
        value = direct_record_category(meta)
        if value:
            return value
    return ""


def infer_bbh_category(prompt: Any) -> str:
    prompt_lower = str(prompt or "").lower()
    if any(
        phrase in prompt_lower
        for phrase in (
            "turn right",
            "turn left",
            "turn around",
            "return to the starting point",
            "take 1 step",
            "take 2 steps",
        )
    ):
        return "Navigation reasoning"
    if "is the following sentence plausible" in prompt_lower:
        return "Sports plausibility"
    if "adjective order" in prompt_lower:
        return "Adjective ordering"
    if "tells the truth" in prompt_lower or "tell the truth" in prompt_lower:
        return "Truthfulness puzzle"
    return ""


def conv_behav_category_from_filename(filename: Any) -> str:
    name = Path(str(filename or "")).stem.lower()
    name = re.sub(r"_rank\d+$", "", name)
    name = re.sub(r"^team_\d+_", "", name)
    keywords = [
        ("career", "Career advice"),
        ("carrer", "Career advice"),
        ("restaurant", "Restaurant ordering"),
        ("trip", "Trip planning"),
        ("movie", "Movie discussion"),
        ("flower", "Flower discussion"),
        ("geography", "Geography"),
        ("smartphone", "Smartphone shopping"),
        ("buying_house", "Buying a house"),
        ("worklife", "Work-life balance"),
        ("math", "Math help"),
        ("friends", "Friends conversation"),
        ("engineering", "Engineering discussion"),
        ("medicine_art", "Medicine and art"),
        ("no_reply", "No-reply scenario"),
    ]
    for needle, label in keywords:
        if needle in name:
            return label
    return ""


_LLM_CATEGORY_CACHE: dict[str, str | None] = {}
_LLM_CATEGORY_DISABLED = False


def llm_category_for_example(bench: str, split: str, text: str) -> str | None:
    """Optional LLM categorization.

    If OPENAI_API_KEY and the openai package are available, use the LLM unless
    VISUALIZE_METRICS_LLM_CATEGORIES is set to 0/false/off. All
    import/credential/runtime failures fall back to dataset-derived labels.
    """
    global _LLM_CATEGORY_DISABLED
    if _LLM_CATEGORY_DISABLED:
        return None
    llm_setting = os.environ.get("VISUALIZE_METRICS_LLM_CATEGORIES", "auto").strip().lower()
    if llm_setting in {"0", "false", "no", "off"}:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    text = str(text or "").strip()
    if not text:
        return None
    cache_key = f"{bench}:{split}:{text[:500]}"
    if cache_key in _LLM_CATEGORY_CACHE:
        return _LLM_CATEGORY_CACHE[cache_key]
    try:
        from openai import OpenAI
    except Exception:
        _LLM_CATEGORY_DISABLED = True
        return None
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("VISUALIZE_METRICS_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return a concise 2-5 word category label for an evaluation example. "
                        "Return only the label."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Dataset: {bench}\nSplit: {split}\nExample:\n{text[:1500]}",
                },
            ],
            temperature=0,
            max_tokens=16,
        )
        label = (resp.choices[0].message.content or "").strip().strip("\"'")
    except Exception:
        _LLM_CATEGORY_DISABLED = True
        return None
    if not label:
        _LLM_CATEGORY_CACHE[cache_key] = None
        return None
    label = re.sub(r"\s+", " ", label)
    label = label[:80]
    _LLM_CATEGORY_CACHE[cache_key] = label
    return label


def dataset_category_for_example(
    bench: str,
    split: str,
    record: dict[str, Any] | None,
    fallback_text: Any = "",
) -> tuple[str, str]:
    record = record or {}
    direct = direct_record_category(record)
    if bench in ("vb_nonmcq", "vb_mcq"):
        if split == "bbh":
            inferred = infer_bbh_category(record.get("problem") or fallback_text)
            if inferred:
                return inferred, "prompt_heuristic"
        if split == "mmsu" and direct and direct.lower() not in {split.lower(), bench.lower()}:
            return direct, "dataset_metadata"
        return VOICEBENCH_SPLIT_CATEGORIES.get(split, titleize_token(split)), "split_label"
    if bench == "bba":
        if direct:
            return direct, "dataset_metadata"
        return titleize_token(split), "split_label"
    if bench in ("fdb_v1", "fdb_v1_5"):
        if direct:
            return direct, "dataset_metadata"
        scenario = FDB_SPLIT_CATEGORIES.get(split, titleize_token(split))
        raw_dataset = str(record.get("dataset") or "")
        source = raw_dataset
        source = re.sub(rf"_{re.escape(split)}$", "", source)
        source = re.sub(rf"_{re.escape(split)}_.*$", "", source)
        if source and source.lower() not in {split.lower(), raw_dataset.lower()}:
            return f"{titleize_token(source)} - {scenario}", "fdb_dataset_split"
        return scenario, "split_label"
    if bench == "fdb_v3":
        if direct:
            return direct, "dataset_metadata"
        return "Tool-call scenario", "layout_label"
    if bench == "conv_behav":
        label = conv_behav_category_from_filename(record.get("filename") or record.get("key") or fallback_text)
        return (label, "filename_topic") if label else ("", "")
    return (direct, "dataset_metadata") if direct else ("", "")


def categorize_example(
    bench: str,
    split: str,
    record: dict[str, Any] | None = None,
    fallback_text: Any = "",
) -> tuple[str, str]:
    text = fallback_text
    if record:
        text = record.get("problem") or record.get("question") or record.get("question_text") or fallback_text
    llm_label = llm_category_for_example(bench, split, str(text or ""))
    if llm_label:
        return llm_label, "llm"
    dataset_label, dataset_source = dataset_category_for_example(bench, split, record, fallback_text)
    if dataset_label:
        return dataset_label, dataset_source
    return "Uncategorized", "none"


def select_examples(examples: list[dict[str, Any]], per_status: int = 40) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        buckets[str(ex.get("status") or "sample")].append(ex)
    selected: list[dict[str, Any]] = []
    for status in ("good", "bad", "ok", "sample"):
        selected.extend(buckets.get(status, [])[:per_status])
    return selected


def load_voicebench_examples(bench_dir: Path, bench: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    label = f"{BENCH_LABELS[bench]} - {split}"
    eval_dir = next((d for d in result_dirs_for(bench_dir, bench, split) if d.exists()), None)
    if eval_dir is None:
        return label, []
    records = load_jsonl(eval_dir / "output_asr.jsonl") or load_jsonl(eval_dir / "output.jsonl")
    results = (
        load_jsonl(eval_dir / "result-voicebench_format.jsonl")
        or load_jsonl(eval_dir / "voicebench_format.jsonl")
        or []
    )
    examples: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        result = results[idx] if idx < len(results) else {}
        audio_path = audio_path_from_record(eval_dir, record)
        src = audio_url(audio_path)
        if not src:
            continue
        response = (
            record.get("debug_info", {}).get("agent_audio_asr")
            if isinstance(record.get("debug_info"), dict)
            else None
        )
        response = response or record.get("generation") or result.get("response") or record.get("generation_text") or ""
        score = None
        raw_score = result.get("score") if isinstance(result, dict) else None
        if isinstance(raw_score, list):
            nums = [as_number(x) for x in raw_score]
            nums = [x for x in nums if x is not None]
            if nums:
                score = float(sum(nums) / len(nums))
        elif as_number(raw_score) is not None:
            score = float(as_number(raw_score) or 0)
        expected = record.get("expected_answer") or result.get("reference")
        correct = answer_correct(expected, response)
        if correct is not None:
            status = "good" if correct else "bad"
        else:
            status = status_from_score(score)
        category, category_source = categorize_example(bench, split, record, record.get("problem") or response)
        examples.append(
            {
                "key": example_key(record, split, idx),
                "status": status,
                "category": category,
                "category_source": category_source,
                "score": score,
                "correct": correct,
                "question": record.get("problem") or result.get("prompt") or "",
                "expected": expected or "",
                "response": clean_generation(response),
                "audio_src": src,
                "audio_path": cluster_path(audio_path),
            }
        )
    return label, select_examples(examples)


def load_bba_examples(bench_dir: Path, split: str) -> tuple[str, list[dict[str, Any]]]:
    label = f"BBA - {split}"
    eval_dir = next((d for d in result_dirs_for(bench_dir, "bba", split) if d.exists()), None)
    if eval_dir is None:
        return label, []
    records = load_jsonl(eval_dir / "output_asr.jsonl") or load_jsonl(eval_dir / "output.jsonl")
    examples: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        audio_path = audio_path_from_record(eval_dir, record)
        src = audio_url(audio_path)
        if not src:
            continue
        response = clean_generation(record.get("generation") or record.get("response") or "")
        expected = record.get("expected_answer") or record.get("reference") or ""
        correct = answer_correct(expected, response)
        status = "sample" if correct is None else ("good" if correct else "bad")
        category, category_source = categorize_example("bba", split, record, record.get("question_asr") or record.get("problem") or "")
        examples.append(
            {
                "key": example_key(record, split, idx),
                "status": status,
                "category": category,
                "category_source": category_source,
                "correct": correct,
                "question": record.get("question_asr") or record.get("problem") or "",
                "expected": expected,
                "response": response,
                "audio_src": src,
                "audio_path": cluster_path(audio_path),
            }
        )
    return label, select_examples(examples)


def load_fdb_examples(bench_dir: Path, bench: str, split: str) -> tuple[str, list[dict[str, Any]]]:
    label = f"{BENCH_LABELS[bench]} - {split}"
    eval_dir = next((d for d in result_dirs_for(bench_dir, bench, split) if d.exists()), None)
    if eval_dir is None:
        return label, []
    records = load_jsonl(eval_dir / "output.jsonl")
    examples: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prepared = fdb_prepared_dir(eval_dir, split, record)
        if prepared is None:
            continue
        out_path = prepared / "output.wav"
        in_path = prepared / "input.wav"
        out_src = audio_url(out_path)
        in_src = audio_url(in_path)
        if not out_src and not in_src:
            continue
        generation = str(record.get("generation") or "")
        took_turn = has_speech_token(generation)
        text = clean_generation(generation)
        word_count = len(text.split())
        if split in ("turn_taking", "interruption"):
            status = "good" if took_turn else "bad"
            status_label = "took turn" if took_turn else "missed turn"
        elif split in ("pause_candor", "pause_synthetic", "background_speech", "talking_to_other"):
            status = "bad" if took_turn else "good"
            status_label = "interrupted" if took_turn else "silent"
        elif split == "backchannel":
            status = "bad" if word_count > 2 else "good"
            status_label = "full takeover" if word_count > 2 else ("backchannel" if took_turn else "silent")
        else:
            status = "sample"
            status_label = "sample"
        category, category_source = categorize_example(bench, split, record, record.get("problem") or status_label)
        examples.append(
            {
                "key": example_key(record, split, idx),
                "status": status,
                "status_label": status_label,
                "category": category,
                "category_source": category_source,
                "sample_id": record.get("sample_id") or record.get("id") or idx,
                "context": record.get("problem") or "",
                "response": text,
                "audio_src": out_src,
                "input_audio_src": in_src,
                "audio_path": cluster_path(out_path) if out_src else "",
                "input_audio_path": cluster_path(in_path) if in_src else "",
                "word_count": word_count,
            }
        )
    return label, select_examples(examples)


def load_fdb_v3_examples(bench_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    label = "FDB v3 - tool call"
    eval_dir = next((d for d in result_dirs_for(bench_dir, "fdb_v3", "tool_call") if d.exists()), None)
    if eval_dir is None:
        return label, []
    layout = eval_dir / "fdb_v3_layout"
    examples: list[dict[str, Any]] = []
    if layout.exists():
        for idx, sample_dir in enumerate(sorted(p for p in layout.iterdir() if p.is_dir())[:80]):
            wavs = sorted(sample_dir.glob("output_*.wav"))
            audio_path = wavs[0] if wavs else None
            src = audio_url(audio_path) if audio_path else None
            if not src:
                continue
            examples.append(
                {
                    "key": f"tool_call:{sample_dir.name}",
                    "status": "sample",
                    "category": "Tool-call scenario",
                    "category_source": "layout_label",
                    "sample_id": sample_dir.name,
                    "context": sample_dir.name,
                    "audio_src": src,
                    "audio_path": cluster_path(audio_path),
                }
            )
    return label, examples


def load_conv_behav_examples(bench_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    label = "Turn Taking Benchmark - agent sessions"
    eval_dir = next((d for d in result_dirs_for(bench_dir, "conv_behav", "overall") if d.exists()), None)
    if eval_dir is None:
        return label, []
    candidates: list[Path] = []
    for subdir in ("validation_logs/agent", "validation_logs/pred_wavs", "validation_logs/pred_audio"):
        candidates.extend(sorted((eval_dir / subdir).glob("*.wav")))
    examples: list[dict[str, Any]] = []
    for wav in candidates[:80]:
        src = audio_url(wav)
        if src:
            record = {"filename": wav.name, "key": wav.name}
            category, category_source = categorize_example("conv_behav", "overall", record, wav.name)
            examples.append(
                {
                    "key": f"overall:{wav.name}",
                    "status": "sample",
                    "category": category,
                    "category_source": category_source,
                    "filename": wav.name,
                    "context": "Agent session audio",
                    "audio_src": src,
                    "audio_path": cluster_path(wav),
                }
            )
    return label, examples


def load_audio_examples(bench_dirs: dict[str, Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    categories: dict[str, dict[str, str]] = {}

    for bench in ("vb_nonmcq", "vb_mcq"):
        bench_dir = bench_dirs.get(bench)
        if not bench_dir:
            continue
        for split in BENCH_SPLITS[bench]:
            label, rows = load_voicebench_examples(bench_dir, bench, split)
            if rows:
                key = f"{bench}:{split}"
                examples[key] = rows
                categories[key] = {"label": label, "dataset": bench, "split": split}

    bench_dir = bench_dirs.get("bba")
    if bench_dir:
        for split in BBA_SPLITS:
            label, rows = load_bba_examples(bench_dir, split)
            if rows:
                key = f"bba:{split}"
                examples[key] = rows
                categories[key] = {"label": label, "dataset": "bba", "split": split}

    for bench in ("fdb_v1", "fdb_v1_5"):
        bench_dir = bench_dirs.get(bench)
        if not bench_dir:
            continue
        for split in BENCH_SPLITS[bench]:
            label, rows = load_fdb_examples(bench_dir, bench, split)
            if rows:
                key = f"{bench}:{split}"
                examples[key] = rows
                categories[key] = {"label": label, "dataset": bench, "split": split}

    bench_dir = bench_dirs.get("fdb_v3")
    if bench_dir:
        label, rows = load_fdb_v3_examples(bench_dir)
        if rows:
            key = "fdb_v3:tool_call"
            examples[key] = rows
            categories[key] = {"label": label, "dataset": "fdb_v3", "split": "tool_call"}

    bench_dir = bench_dirs.get("conv_behav")
    if bench_dir:
        label, rows = load_conv_behav_examples(bench_dir)
        if rows:
            key = "conv_behav:overall"
            examples[key] = rows
            categories[key] = {"label": label, "dataset": "conv_behav", "split": "overall"}

    return examples, categories


CATEGORY_SOURCE_DESCRIPTIONS = {
    "dataset_metadata": "dataset metadata/category fields",
    "split_label": "benchmark split or task labels",
    "fdb_dataset_split": "FDB dataset field plus semantic scenario labels",
    "prompt_heuristic": "prompt text heuristics",
    "filename_topic": "known filename topic patterns",
    "layout_label": "fixed layout label",
    "none": "uncategorized examples",
}


def llm_category_status() -> str:
    setting = os.environ.get("VISUALIZE_METRICS_LLM_CATEGORIES", "auto").strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return "disabled by VISUALIZE_METRICS_LLM_CATEGORIES"
    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI  # noqa: F401
    except Exception:
        return "openai package import failed"
    if _LLM_CATEGORY_DISABLED:
        return "LLM category call failed; using fallbacks"
    return "available"


def warn_category_fallbacks(ckpt: dict[str, Any]) -> None:
    examples = ckpt.get("audio_examples") or {}
    if not examples:
        return
    by_dataset: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"sources": set(), "categories": set()})
    for cat_key, rows in examples.items():
        dataset = cat_key.split(":", 1)[0]
        for ex in rows or []:
            source = str(ex.get("category_source") or "none")
            by_dataset[dataset]["sources"].add(source)
            by_dataset[dataset]["categories"].add(str(ex.get("category") or "Uncategorized"))

    llm_status = llm_category_status()
    for dataset, data in sorted(by_dataset.items()):
        fallback_sources = sorted(source for source in data["sources"] if source != "llm")
        if not fallback_sources:
            continue
        descriptions = [CATEGORY_SOURCE_DESCRIPTIONS.get(source, source) for source in fallback_sources]
        categories = sorted(data["categories"])
        shown_categories = ", ".join(categories[:6]) + (f", +{len(categories) - 6} more" if len(categories) > 6 else "")
        print(
            "warning: "
            f"{ckpt.get('name', 'metrics')}: {BENCH_LABELS.get(dataset, dataset)} audio categorization "
            f"used fallback(s): {', '.join(descriptions)}. LLM status: {llm_status}. "
            f"Categories: {shown_categories or 'Uncategorized'}.",
            file=sys.stderr,
        )


def is_score5_metric(key: str) -> bool:
    leaf = key.lower().rsplit(".", 1)[-1]
    if leaf in SCORE5_KEYS:
        return True
    for suffix in ("_asr", "_audio", "_text"):
        if leaf.endswith(suffix) and leaf[: -len(suffix)] in SCORE5_KEYS:
            return True
    return False


def metric_percent_value(key: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    key_l = key.lower()
    val = float(value)
    if is_score5_metric(key_l):
        return val * 20 if val <= 5 else val
    if PERCENT_KEY_RE.search(key_l):
        return val * 100 if 0 <= val <= 1 else val
    return val


def first_metric(values: dict[str, float | int], keys: list[str]) -> float | None:
    for key in keys:
        if key in values and as_number(values[key]) is not None:
            return float(values[key])
    return None


def primary_split_score(bench: str, split: str, values: dict[str, float | int]) -> float | None:
    if not values:
        return None
    if bench == "vb_nonmcq":
        for key in ("gpt", "gpt_asr", "panda", "final", "strict-prompt", "loose-prompt"):
            if key in values:
                return metric_percent_value(key, values[key])
    if bench == "vb_mcq":
        for key in ("acc", "accuracy", "final"):
            if key in values:
                return metric_percent_value(key, values[key])
    if bench == "bba":
        return metric_percent_value("accuracy", first_metric(values, ["accuracy"]))
    if bench == "bfcl":
        return metric_percent_value("accuracy", first_metric(values, ["accuracy"]))
    if bench in ("fdb_v1", "fdb_v1_5"):
        tor = first_metric(values, ["tor_pct"])
        if tor is None:
            tor = first_metric(values, ["turn", "tor"])
            tor = metric_percent_value("rate", tor) if tor is not None else None
        if tor is not None:
            if split in ("pause_candor", "pause_synthetic", "backchannel", "background_speech", "talking_to_other"):
                return max(0.0, min(100.0, 100.0 - tor))
            return max(0.0, min(100.0, tor))
        for key in ("rating", "behavior_C_RESUME", "behavior_C_RESPOND"):
            if key in values:
                return metric_percent_value(key, values[key])
    if bench == "fdb_v3":
        vals = []
        for key in (
            "headline.tool_selection_acc",
            "headline.argument_acc",
            "headline.response_qual",
            "headline.turn_take_rate",
        ):
            if key in values:
                vals.append(metric_percent_value(key, values[key]))
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    if bench == "conv_behav":
        for key in ("tt_f1", "user_eou_f1", "barge_in_success_rate"):
            if key in values:
                return metric_percent_value(key, values[key])
    for key, value in values.items():
        if key not in COUNT_KEYS and not MS_RE.search(key.lower()):
            return metric_percent_value(key, value)
    return None


def compute_detailed(metrics: dict[str, Any]) -> dict[str, Any]:
    detailed: dict[str, Any] = {}
    for bench, modes in metrics.items():
        detailed[bench] = {}
        for mode, splits in modes.items():
            detailed[bench][mode] = {}
            for split, values in splits.items():
                detailed[bench][mode][split] = primary_split_score(bench, split, values)
    return detailed


def compute_headlines(metrics: dict[str, Any]) -> dict[str, Any]:
    headlines: dict[str, Any] = {}
    for bench in BENCHMARKS:
        modes = metrics.get(bench) or {}
        if not modes:
            continue
        headlines[bench] = {}
        for mode, splits in modes.items():
            if bench == "bfcl":
                correct = sum(float(v.get("num_correct", 0)) for v in splits.values() if isinstance(v, dict))
                total = sum(float(v.get("num_samples", 0)) for v in splits.values() if isinstance(v, dict))
                if total > 0:
                    headlines[bench][mode] = round(correct * 100.0 / total, 2)
                    continue
            vals = [
                primary_split_score(bench, split, values)
                for split, values in splits.items()
                if isinstance(values, dict)
            ]
            vals = [v for v in vals if v is not None]
            headlines[bench][mode] = round(sum(vals) / len(vals), 2) if vals else None
    return headlines


def load_metrics_from_dir(source: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    bench_dirs: dict[str, Path] = {}
    metric_sources: dict[str, Any] = {}
    for bench in BENCHMARKS:
        bench_dir = find_benchmark_dir(source, bench)
        if not bench_dir:
            continue
        bench_dirs[bench] = bench_dir
        for split in BENCH_SPLITS[bench]:
            metrics_file = next((p for p in metric_file_candidates(bench_dir, bench, split) if p.exists()), None)
            if not metrics_file:
                continue
            try:
                raw = load_json(metrics_file)
            except Exception as exc:
                print(f"warning: could not read {metrics_file}: {exc}", file=sys.stderr)
                continue
            modes = extract_metric_modes(raw, metric_key_names(bench, split), infer_mode(metrics_file))
            for mode, raw_metrics in modes.items():
                flat = numeric_leaves(raw_metrics)
                if not flat:
                    continue
                metrics.setdefault(bench, {}).setdefault(mode, {})[split] = flat
                metric_sources.setdefault(bench, {}).setdefault(mode, {})[split] = cluster_path(metrics_file)
    return metrics, bench_dirs, metric_sources


def normalize_old_sidecar(data: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    if not metrics:
        metrics = {}
        detailed = data.get("detailed") if isinstance(data.get("detailed"), dict) else {}
        for bench, value in detailed.items():
            target_bench = "fdb_v1" if bench == "fdb" else bench
            if not isinstance(value, dict):
                continue
            # Old sidecars are mode -> split -> number. Very old sidecars were split -> number.
            if any(mode in value for mode in KNOWN_MODES):
                for mode, splits in value.items():
                    if not isinstance(splits, dict):
                        continue
                    for split, val in splits.items():
                        num = as_number(val)
                        if num is not None:
                            metrics.setdefault(target_bench, {}).setdefault(mode, {})[split] = {"score": num}
            else:
                for split, val in value.items():
                    num = as_number(val)
                    if num is not None:
                        metrics.setdefault(target_bench, {}).setdefault("greedy", {})[split] = {"score": num}

    headlines = data.get("headlines") if isinstance(data.get("headlines"), dict) else {}
    normalized_headlines: dict[str, Any] = {}
    for bench, value in headlines.items():
        target_bench = "fdb_v1" if bench == "fdb" else bench
        if isinstance(value, dict):
            normalized_headlines[target_bench] = {m: as_number(v) for m, v in value.items() if m in KNOWN_MODES}
        else:
            normalized_headlines[target_bench] = {"greedy": as_number(value)}

    if not normalized_headlines:
        normalized_headlines = compute_headlines(metrics)
    detailed = compute_detailed(metrics)

    audio_examples = data.get("audio_examples") if isinstance(data.get("audio_examples"), dict) else {}
    audio_categories = data.get("audio_categories") if isinstance(data.get("audio_categories"), dict) else {}
    for key in audio_examples:
        audio_categories.setdefault(
            key,
            {
                "label": DEFAULT_AUDIO_LABELS.get(key, key.replace(":", " - ").replace("_", " ")),
                "dataset": key.split(":", 1)[0],
                "split": key.split(":", 1)[1] if ":" in key else key,
            },
        )

    modes = sorted({mode for bench_modes in metrics.values() for mode in bench_modes})
    return {
        "name": data.get("name") or path.stem,
        "commit": data.get("commit") or infer_commit(path),
        "date": data.get("date") or date.today().isoformat(),
        "source_dir": data.get("source_dir") or str(path),
        "metric_sources": data.get("metric_sources") if isinstance(data.get("metric_sources"), dict) else {},
        "modes": modes or list(KNOWN_MODES),
        "headlines": normalized_headlines,
        "detailed": detailed,
        "metrics": metrics,
        "audio_examples": audio_examples,
        "audio_categories": audio_categories,
    }


def selected_source_metadata(bench_dirs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for bench, path in bench_dirs.items():
        metadata[bench] = {
            "path": str(path),
            "commit": infer_commit(path),
            "mtime": dir_mtime(path),
        }
    return metadata


def checkpoint_commit_label(source: Path, bench_dirs: dict[str, Path]) -> str:
    source_commit = infer_commit(source)
    if source_commit:
        return source_commit
    commits = sorted({infer_commit(path) for path in bench_dirs.values() if infer_commit(path)})
    if len(commits) == 1:
        return commits[0]
    if len(commits) > 1:
        return "mixed"
    return ""


def build_checkpoint(source: Path) -> dict[str, Any]:
    if source.suffix.lower() == ".json":
        return normalize_old_sidecar(load_json(source), source)

    metrics, bench_dirs, metric_sources = load_metrics_from_dir(source)
    audio_examples, audio_categories = load_audio_examples(bench_dirs)
    modes = sorted({mode for bench_modes in metrics.values() for mode in bench_modes})
    return {
        "name": display_name_for_source(source) or "metrics",
        "commit": checkpoint_commit_label(source, bench_dirs),
        "date": date.today().isoformat(),
        "source_dir": str(source),
        "selected_sources": selected_source_metadata(bench_dirs),
        "metric_sources": metric_sources,
        "modes": modes or [infer_mode(source)],
        "headlines": compute_headlines(metrics),
        "detailed": compute_detailed(metrics),
        "metrics": metrics,
        "audio_examples": audio_examples,
        "audio_categories": audio_categories,
    }


def color_val(bench: str, value: float | int | None) -> str:
    if value is None:
        return "var(--tx2)"
    hi, lo = BENCH_THRESHOLDS.get(bench, (70, 50))
    return "var(--gr)" if value >= hi else ("var(--ye)" if value >= lo else "var(--re)")


def fmt_metric(key: str, value: Any) -> str:
    num = as_number(value)
    if num is None:
        return "-"
    key_l = key.lower()
    leaf = key_l.rsplit(".", 1)[-1]
    if leaf in COUNT_KEYS:
        return f"{int(round(float(num)))}"
    if MS_RE.search(key_l):
        return f"{float(num):.0f} ms"
    if is_score5_metric(key_l) and float(num) <= 5:
        return f"{float(num):.2f}/5"
    if PERCENT_KEY_RE.search(key_l):
        val = float(num) * 100 if 0 <= float(num) <= 1 and "pct" not in key_l else float(num)
        return f"{val:.1f}%"
    if isinstance(num, int) or float(num).is_integer():
        return str(int(num))
    return f"{float(num):.3f}".rstrip("0").rstrip(".")


def fmt_headline(value: Any) -> str:
    num = as_number(value)
    return "-" if num is None else f"{float(num):.1f}%"


def fmt_display_metric(metric: str, value: Any) -> str:
    num = as_number(value)
    if num is None:
        return "-"
    metric_l = metric.lower()
    if is_score5_metric(metric_l) or PERCENT_KEY_RE.search(metric_l):
        return f"{float(num):.1f}%"
    return fmt_metric(metric, num)


def mode_label(mode: str) -> str:
    return {"greedy": "G", "sampling": "S"}.get(mode, mode[:1].upper())


def all_modes_for_bench(ckpts: list[dict[str, Any]], bench: str) -> list[str]:
    modes = {
        mode
        for ckpt in ckpts
        for mode in ((ckpt.get("metrics") or {}).get(bench) or {}).keys()
    }
    if not modes:
        modes = {
            mode
            for ckpt in ckpts
            for mode in ((ckpt.get("headlines") or {}).get(bench) or {}).keys()
        }
    return [m for m in KNOWN_MODES if m in modes] + sorted(m for m in modes if m not in KNOWN_MODES)


def selected_commit_for(ckpt: dict[str, Any], bench: str) -> str:
    selected = (ckpt.get("selected_sources") or {}).get(bench) or {}
    if isinstance(selected, dict):
        return str(selected.get("commit") or "")
    return ""


def has_bench_data(ckpt: dict[str, Any], bench: str) -> bool:
    return bool(((ckpt.get("metrics") or {}).get(bench) or {}) or ((ckpt.get("headlines") or {}).get(bench) or {}))


def is_display_metric(metric: str) -> bool:
    metric_l = metric.lower()
    leaf = metric_l.rsplit(".", 1)[-1]
    if ".scenario_results." in metric_l or ".details." in metric_l:
        return False
    if ".expected_args." in metric_l or ".actual_args." in metric_l:
        return False
    if metric_l.startswith("run_accuracies."):
        return False
    if leaf in COUNT_KEYS or leaf in {"total_scenarios", "num_result_files"}:
        return False
    if metric_l.startswith(("agent_char_", "agent_word_", "agent_ref_")):
        return False
    if leaf in {"tor", "turn", "latency", "response_latency", "stop_latency"}:
        return False
    if metric_l.startswith("pass_rate.") and leaf not in {"accuracy", "pass_rate"}:
        return False
    return True


def is_display_metric_for_bench(bench: str, metric: str) -> bool:
    if not is_display_metric(metric):
        return False
    metric_l = metric.lower()
    leaf = metric_l.rsplit(".", 1)[-1]
    if bench == "fdb_v3":
        return metric_l.startswith("headline.")
    if bench == "bba":
        return leaf == "accuracy"
    if bench == "bfcl":
        return leaf == "accuracy"
    return True


def metric_value_for_summary(metric: str, value: Any) -> float | None:
    num = as_number(value)
    if num is None:
        return None
    metric_l = metric.lower()
    if is_score5_metric(metric_l) or PERCENT_KEY_RE.search(metric_l):
        return metric_percent_value(metric, num)
    return float(num)


def aggregate_metric_for_ckpt(ckpt: dict[str, Any], bench: str, metric: str) -> dict[str, float | None]:
    bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
    aggregated: dict[str, float | None] = {}
    for mode, splits in bench_metrics.items():
        vals = []
        for values in (splits or {}).values():
            if not isinstance(values, dict) or metric not in values:
                continue
            val = metric_value_for_summary(metric, values.get(metric))
            if val is not None:
                vals.append(val)
        if not vals:
            aggregated[mode] = None
        elif metric.lower().rsplit(".", 1)[-1] in COUNT_KEYS:
            aggregated[mode] = sum(vals)
        else:
            aggregated[mode] = sum(vals) / len(vals)
    return aggregated


def summary_metrics_for_bench(ckpts: list[dict[str, Any]], bench: str) -> list[str]:
    metrics: set[str] = set()
    for ckpt in ckpts:
        bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
        for splits in bench_metrics.values():
            for values in (splits or {}).values():
                if isinstance(values, dict):
                    metrics.update(metric for metric in values.keys() if is_display_metric_for_bench(bench, metric))
    return sorted(metrics)


def formatted_mode_values(values_by_mode: dict[str, float | None], metric: str, bench: str) -> str:
    if not values_by_mode:
        return "-"
    parts = []
    ordered_modes = [m for m in KNOWN_MODES if m in values_by_mode] + sorted(m for m in values_by_mode if m not in KNOWN_MODES)
    for mode in ordered_modes:
        val = values_by_mode.get(mode)
        color = color_val(bench, metric_percent_value(metric, val) if val is not None else None)
        parts.append(
            f'<span class="mode-score"><span class="mode-tag">{h(mode_label(mode))}</span> '
            f'<span style="color:{color}">{h(fmt_display_metric(metric, val))}</span></span>'
        )
    return "".join(parts) or "-"


def normalize_cluster_path(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if re.match(r"^[A-Za-z0-9_.-]+:/", text) or text.startswith("file://"):
        return text
    return cluster_path(text)


def path_chip_html(value: Any) -> str:
    full = normalize_cluster_path(value)
    if not full:
        return ""
    return f'<code class="path-chip" title="{h(full)}">{h(full)}</code>'


def path_line_html(value: Any, label: str) -> str:
    chip = path_chip_html(value)
    if not chip:
        return ""
    return f'<div class="path-line"><span class="dim">{h(label)}</span>{chip}</div>'


def metric_file_paths_for_ckpt(ckpt: dict[str, Any], bench: str, metric: str) -> list[str]:
    bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
    bench_sources = (ckpt.get("metric_sources") or {}).get(bench) or {}
    paths: list[str] = []
    seen: set[str] = set()
    for mode, splits in bench_metrics.items():
        mode_sources = bench_sources.get(mode, {}) if isinstance(bench_sources, dict) else {}
        for split, values in (splits or {}).items():
            if metric != "headline" and (not isinstance(values, dict) or metric not in values):
                continue
            path = mode_sources.get(split) if isinstance(mode_sources, dict) else None
            path = normalize_cluster_path(path)
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def metric_files_cell(ckpts: list[dict[str, Any]], bench: str, metric: str) -> str:
    entries = []
    for ckpt in ckpts:
        paths = metric_file_paths_for_ckpt(ckpt, bench, metric)
        if not paths:
            continue
        shown = "".join(path_chip_html(path) for path in paths[:2])
        if len(paths) > 2:
            full = "\n".join(paths)
            shown += f'<span class="path-more" title="{h(full)}">+{len(paths) - 2} more</span>'
        entries.append(
            f'<div class="metric-file-entry"><span class="metric-file-run">{h(ckpt["name"])}</span>{shown}</div>'
        )
    return '<td class="metric-file-cell">' + ("".join(entries) if entries else "-") + "</td>"


def summary_table(ckpts: list[dict[str, Any]]) -> str:
    hdrs = "".join(
        f'<th style="color:{COLORS[i % len(COLORS)]};white-space:nowrap">'
        f'{h(ck["name"])}<br><small style="font-weight:400;color:var(--tx2)">{h(ck.get("commit", ""))}</small></th>'
        for i, ck in enumerate(ckpts)
    )
    rows = []
    for bench in BENCHMARKS:
        metrics = summary_metrics_for_bench(ckpts, bench)
        if not metrics:
            metrics = ["headline"]
        single_metric = len(metrics) == 1
        for idx, metric in enumerate(metrics):
            cells = []
            for ckpt in ckpts:
                if metric == "headline":
                    values_by_mode = (ckpt.get("headlines") or {}).get(bench) or {}
                    cell = formatted_mode_values(values_by_mode, "score", bench)
                else:
                    cell = formatted_mode_values(aggregate_metric_for_ckpt(ckpt, bench, metric), metric, bench)
                selected_commit = selected_commit_for(ckpt, bench)
                commit_html = f'<div class="dim">{h(selected_commit)}</div>' if selected_commit and idx == 0 else ""
                cells.append(f'<td style="text-align:center;font-weight:700">{cell}{commit_html}</td>')
            dataset_cell = f"<strong>{h(BENCH_LABELS[bench])}</strong>" if idx == 0 else ""
            metric_cell = "" if single_metric and metric == "headline" else h(metric)
            files_cell = metric_files_cell(ckpts, bench, metric)
            rows.append(f"<tr><td>{dataset_cell}</td><td>{metric_cell}</td>{files_cell}{''.join(cells)}</tr>")
    return (
        '<div style="overflow-x:auto"><table class="cmp-table">'
        f"<thead><tr><th>Dataset</th><th>Metric</th><th>Metric files</th>{hdrs}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def iter_metric_rows(ckpts: list[dict[str, Any]], bench: str) -> list[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    for ckpt in ckpts:
        bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
        for mode, splits in bench_metrics.items():
            for split, values in (splits or {}).items():
                if not isinstance(values, dict):
                    continue
                for metric in values:
                    rows.add((split, metric, mode))
    mode_order = {mode: idx for idx, mode in enumerate(KNOWN_MODES)}
    return sorted(rows, key=lambda x: (x[0], x[1], mode_order.get(x[2], 99), x[2]))


def iter_display_metric_rows(ckpts: list[dict[str, Any]], bench: str) -> list[tuple[str, str, str | None]]:
    grouped: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for ckpt_idx, ckpt in enumerate(ckpts):
        bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
        for mode, splits in bench_metrics.items():
            for split, values in (splits or {}).items():
                if not isinstance(values, dict):
                    continue
                for metric in values:
                    grouped[(split, metric)][ckpt_idx].add(mode)

    rows: list[tuple[str, str, str | None]] = []
    mode_order = {mode: idx for idx, mode in enumerate(KNOWN_MODES)}
    for (split, metric), by_ckpt in grouped.items():
        needs_mode = any(len(modes) > 1 for modes in by_ckpt.values())
        if needs_mode:
            modes = sorted({mode for modes in by_ckpt.values() for mode in modes}, key=lambda m: (mode_order.get(m, 99), m))
            rows.extend((split, metric, mode) for mode in modes)
        else:
            rows.append((split, metric, None))
    return sorted(rows, key=lambda x: (x[0], x[1], mode_order.get(x[2] or "", 99), x[2] or ""))


def metric_value_for_row(ckpt: dict[str, Any], bench: str, split: str, metric: str, mode: str | None) -> Any:
    bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
    if mode is not None:
        return bench_metrics.get(mode, {}).get(split, {}).get(metric)
    for candidate_mode in [*KNOWN_MODES, *sorted(m for m in bench_metrics if m not in KNOWN_MODES)]:
        value = bench_metrics.get(candidate_mode, {}).get(split, {}).get(metric)
        if value is not None:
            return value
    return None


def detailed_comparison_table_for_bench(ckpts: list[dict[str, Any]], bench: str) -> str:
    rows = [row for row in iter_display_metric_rows(ckpts, bench) if is_display_metric_for_bench(bench, row[1])]
    if not rows:
        return ""
    hdrs = "".join(
        f'<th style="color:{COLORS[i % len(COLORS)]};text-align:center">{h(ck["name"])}</th>'
        for i, ck in enumerate(ckpts)
    )
    body = []
    for split, metric, mode in rows:
        cells = []
        for ckpt in ckpts:
            value = metric_value_for_row(ckpt, bench, split, metric, mode)
            cells.append(f'<td style="text-align:center;font-weight:600">{h(fmt_metric(metric, value))}</td>')
        metric_name = f"{metric} ({mode})" if mode else metric
        body.append(f"<tr><td>{h(split)}</td><td>{h(metric_name)}</td>{''.join(cells)}</tr>")
    return (
        f'<details class="bench-detail" open><summary>{h(BENCH_LABELS[bench])} '
        f'<span class="dim">comparison ({len(ckpts)} dirs)</span></summary>'
        '<div style="overflow-x:auto"><table class="cmp-table">'
        f"<thead><tr><th>Split</th><th>Metric</th>{hdrs}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        "</details>"
    )


def single_table_for_bench(ckpt: dict[str, Any], bench: str) -> str:
    bench_metrics = (ckpt.get("metrics") or {}).get(bench) or {}
    if not bench_metrics:
        return ""
    rows_meta = iter_metric_rows([ckpt], bench)
    metrics_by_split: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for split, metric, mode in rows_meta:
        if not is_display_metric_for_bench(bench, metric):
            continue
        val = bench_metrics.get(mode, {}).get(split, {}).get(metric)
        label = f"{metric} ({mode})" if len(bench_metrics) > 1 else metric
        metrics_by_split[(split, label)]["value"] = val
        metrics_by_split[(split, label)]["metric"] = metric
    body = []
    for (split, label), val_meta in sorted(metrics_by_split.items()):
        body.append(
            f"<tr><td>{h(split)}</td><td>{h(label)}</td>"
            f"<td style=\"text-align:right;font-weight:700\">{h(fmt_metric(val_meta['metric'], val_meta['value']))}</td></tr>"
        )
    if not body:
        return ""
    selected_commit = selected_commit_for(ckpt, bench)
    source_note = h(ckpt["name"])
    if selected_commit:
        source_note += f" ({h(selected_commit)})"
    return (
        f'<details class="bench-detail" open><summary>{h(BENCH_LABELS[bench])} '
        f'<span class="dim">single dir: {source_note}</span></summary>'
        '<div style="overflow-x:auto"><table class="cmp-table">'
        f"<thead><tr><th>Split</th><th>Metric</th><th>Value</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        "</details>"
    )


def detailed_comparison_tables(ckpts: list[dict[str, Any]]) -> str:
    parts = ["<h2>Detailed Metrics</h2>"]
    for bench in BENCHMARKS:
        section = detailed_comparison_table_for_bench(ckpts, bench)
        if section:
            parts.append(section)
    return "\n".join(parts)


def dataset_detailed_sections(ckpts: list[dict[str, Any]]) -> str:
    parts = ["<h2>Detailed Metrics</h2>"]
    for bench in BENCHMARKS:
        active_ckpts = [ckpt for ckpt in ckpts if has_bench_data(ckpt, bench)]
        if len(active_ckpts) > 1:
            section = detailed_comparison_table_for_bench(active_ckpts, bench)
        elif len(active_ckpts) == 1:
            section = single_table_for_bench(active_ckpts[0], bench)
        else:
            section = ""
        if section:
            parts.append(section)
    return "\n".join(parts)


def chart_metric_value(metric: str, value: Any) -> float | None:
    num = as_number(value)
    if num is None:
        return None
    metric_l = metric.lower()
    if is_score5_metric(metric_l) or PERCENT_KEY_RE.search(metric_l):
        return metric_percent_value(metric, num)
    return float(num)


def static_bar_chart(labels: list[str], values: list[float | None], colors: list[str], metric: str) -> str:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return ""
    metric_l = metric.lower()
    percent_axis = (is_score5_metric(metric_l) or bool(PERCENT_KEY_RE.search(metric_l))) and all(
        v is None or 0 <= v <= 100 for v in values
    )
    axis_max = 100.0 if percent_axis else max(max(numeric), 1.0)
    bars = []
    for label, value, color in zip(labels, values, colors):
        if value is None:
            bars.append(
                '<div class="static-bar-item missing">'
                '<div class="static-bar-value">-</div>'
                '<div class="static-bar-shell"><div class="static-bar" style="height:0"></div></div>'
                f'<div class="static-bar-label">{h(label)}</div></div>'
            )
            continue
        height = max(2.0, min(100.0, (float(value) / axis_max) * 100.0))
        bars.append(
            '<div class="static-bar-item">'
            f'<div class="static-bar-value">{h(fmt_display_metric(metric, value))}</div>'
            '<div class="static-bar-shell">'
            f'<div class="static-bar" style="height:{height:.2f}%;background:{color}"></div>'
            '</div>'
            f'<div class="static-bar-label" title="{h(label)}">{h(label)}</div></div>'
        )
    max_label = "100%" if percent_axis else h(fmt_metric(metric, axis_max))
    return (
        '<div class="static-chart">'
        f'<div class="static-y-label">max {max_label}</div>'
        f'<div class="static-bars">{"".join(bars)}</div>'
        '<div class="static-x-label">metrics dir</div>'
        '</div>'
    )


def bar_charts(ckpts: list[dict[str, Any]]) -> tuple[str, str]:
    html_parts = ["<h2>Per-Split Metric Comparisons</h2>"]
    for bench in BENCHMARKS:
        active_ckpts = [ckpt for ckpt in ckpts if has_bench_data(ckpt, bench)]
        if len(active_ckpts) < 2:
            continue
        rows = [row for row in iter_display_metric_rows(active_ckpts, bench) if is_display_metric_for_bench(bench, row[1])]
        if not rows:
            continue
        labels = [ck["name"] for ck in active_ckpts]
        colors = [COLORS[i % len(COLORS)] for i in range(len(active_ckpts))]
        cards = []
        for split, metric, mode in rows:
            values = [
                chart_metric_value(metric, metric_value_for_row(ck, bench, split, metric, mode))
                for ck in active_ckpts
            ]
            if sum(v is not None for v in values) < 2:
                continue
            title = f"{split} - {metric}" + (f" ({mode})" if mode else "")
            cards.append(
                f'<div class="chart-wrap"><h3>{h(title)}</h3>{static_bar_chart(labels, values, colors, metric)}</div>'
            )
        if cards:
            html_parts.append(
                f'<details class="bench-detail" open><summary>{h(BENCH_LABELS[bench])}</summary>'
                f'<div class="bar-grid">{"".join(cards)}</div></details>'
            )
    if len(html_parts) == 1:
        return "", ""
    return "\n".join(html_parts), ""


def single_tables(ckpt: dict[str, Any]) -> str:
    parts = ["<h2>Metrics</h2>"]
    for bench in BENCHMARKS:
        section = single_table_for_bench(ckpt, bench)
        if section:
            parts.append(section)
    return "\n".join(parts)


def status_kind(example: dict[str, Any] | None) -> str:
    if not example:
        return "missing"
    status = str(example.get("status") or example.get("quality") or "").lower()
    if status in {"good", "correct", "took_turn", "silent"}:
        return "good"
    if status in {"bad", "wrong", "incorrect", "missed_turn", "interrupted"}:
        return "bad"
    if example.get("correct") is True:
        return "good"
    if example.get("correct") is False:
        return "bad"
    score = as_number(example.get("score"))
    if score is not None:
        return status_from_score(float(score))
    outcome = str(example.get("outcome") or "").lower()
    if outcome in {"took_turn", "silent"}:
        return "good"
    if outcome in {"missed_turn", "interrupted"}:
        return "bad"
    return status or "sample"


def status_badge(example: dict[str, Any] | None) -> str:
    kind = status_kind(example)
    label = (example or {}).get("status_label") or (example or {}).get("quality") or kind
    color = {"good": "var(--gr)", "bad": "var(--re)", "ok": "var(--ye)", "missing": "var(--tx2)"}.get(
        kind, "var(--tx2)"
    )
    return f'<span class="badge" style="background:{color}20;color:{color};border:1px solid {color}">{h(label)}</span>'


def audio_player(src: str | None) -> str:
    if not src:
        return '<em style="color:var(--tx2);font-size:0.82em">audio not available</em>'
    return f'<audio controls src="{h(src)}" class="aplayer"></audio>'


def example_context(example: dict[str, Any]) -> str:
    bits = []
    question = example.get("question") or example.get("context")
    if question:
        bits.append(f'<div class="bubble bubble-q"><strong>Prompt</strong><br>{h(str(question)[:500])}</div>')
    expected = example.get("expected")
    if expected:
        bits.append(f'<div class="bubble"><strong>Expected:</strong> {h(expected)}</div>')
    input_audio = example.get("input_audio_src")
    if input_audio:
        bits.append(
            f'<div class="mini-label">Input audio</div>{audio_player(input_audio)}'
            f'{path_line_html(example.get("input_audio_path"), "Input path")}'
        )
    if not bits:
        filename = example.get("filename") or example.get("sample_id") or example.get("key")
        bits.append(f'<div class="bubble bubble-q">{h(filename)}</div>')
    return "".join(bits)


def example_response(example: dict[str, Any] | None, ckpt_name: str, color: str) -> str:
    if not example:
        return (
            f'<div class="audio-row"><span class="ckpt-badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}">{h(ckpt_name)}</span>'
            '<div style="color:var(--tx2)">No matching example for this checkpoint.</div></div>'
        )
    response = example.get("response") or example.get("generation") or ""
    score = as_number(example.get("score"))
    score_html = f' <span class="dim">score {float(score):.1f}</span>' if score is not None else ""
    return (
        f'<div class="audio-row"><span class="ckpt-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}">{h(ckpt_name)}</span> {status_badge(example)}{score_html}'
        f'<div class="bubble bubble-a">{h(str(response)[:500])}</div>{audio_player(example.get("audio_src"))}'
        f'{path_line_html(example.get("audio_path"), "Audio path")}</div>'
    )


def category_label(ckpts: list[dict[str, Any]], cat: str) -> str:
    for ckpt in ckpts:
        meta = (ckpt.get("audio_categories") or {}).get(cat)
        if isinstance(meta, dict) and meta.get("label"):
            return str(meta["label"])
    return DEFAULT_AUDIO_LABELS.get(cat, cat.replace(":", " - ").replace("_", " "))


def audio_categories(ckpts: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    cats: list[str] = []
    for ckpt in ckpts:
        for cat, rows in (ckpt.get("audio_examples") or {}).items():
            if rows and cat not in seen:
                cats.append(cat)
                seen.add(cat)
    return cats


def audio_dataset_key(cat: str) -> str:
    return cat.split(":", 1)[0]


def audio_dataset_label(cat: str) -> str:
    return BENCH_LABELS.get(audio_dataset_key(cat), audio_dataset_key(cat).replace("_", " "))


def example_category_value(example: dict[str, Any] | None) -> str:
    if not example:
        return "Uncategorized"
    return str(example.get("category") or "Uncategorized")


def category_source_value(example: dict[str, Any] | None) -> str:
    if not example:
        return "none"
    source = str(example.get("category_source") or "none")
    return {
        "llm": "LLM",
        "dataset_metadata": "dataset metadata",
        "split_label": "split/task fallback",
        "fdb_dataset_split": "FDB dataset+scenario fallback",
        "prompt_heuristic": "prompt heuristic fallback",
        "filename_topic": "filename topic fallback",
        "layout_label": "layout fallback",
        "none": "uncategorized",
    }.get(source, source)


def category_heading(category: str, source: str) -> str:
    return (
        f'<h4 class="category-group-title"><span>Category: {h(category)}</span> '
        f'<span class="dim">source: {h(source)}</span></h4>'
    )


def maps_for_category(cat_maps: list[dict[str, dict[str, Any]]], category: str) -> list[dict[str, dict[str, Any]]]:
    return [
        {key: ex for key, ex in cmap.items() if example_category_value(ex) == category}
        for cmap in cat_maps
    ]


def categories_for_maps(cat_maps: list[dict[str, dict[str, Any]]]) -> list[str]:
    categories = sorted({example_category_value(ex) for cmap in cat_maps for ex in cmap.values()})
    return categories or ["Uncategorized"]


def audio_category_single_html(ckpt: dict[str, Any], cat: str, cat_idx: int, include_heading: bool = True) -> str:
    rows = (ckpt.get("audio_examples") or {}).get(cat) or []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in rows:
        by_category[example_category_value(ex)].append(ex)

    parts = [
        f'<section class="ex-section" id="audio-{cat_idx}">'
    ]
    if include_heading:
        parts.append(
            f'<h3>{h(category_label([ckpt], cat))} '
            f'<span class="dim">single dir: {h(ckpt["name"])}</span></h3>'
        )
    for category, category_rows in sorted(by_category.items()):
        source = category_source_value(category_rows[0] if category_rows else None)
        parts.append(category_heading(category, source))
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ex in category_rows:
            buckets[status_kind(ex)].append(ex)
        for status in ("good", "bad", "ok", "sample"):
            examples = buckets.get(status) or []
            if not examples:
                continue
            parts.append(f'<div class="status-title">{h(status.title())}</div>')
            for ex in examples[:4]:
                parts.append(
                    '<div class="ex-card">'
                    f'<div class="ex-header">{status_badge(ex)} <code>{h(ex.get("key", ""))}</code></div>'
                    f'<div class="ex-body-cmp"><div class="ex-left">{example_context(ex)}</div>'
                    f'<div class="ex-right-cmp">{example_response(ex, ckpt["name"], COLORS[0])}</div></div></div>'
                )
    parts.append('<a class="back-link" href="#audio-top">Back to Audio Examples</a></section>')
    return "\n".join(parts)


def contrast_key_for_target(cat_maps: list[dict[str, dict[str, Any]]], target_idx: int) -> str | None:
    target = cat_maps[target_idx]
    for key, example in target.items():
        if status_kind(example) != "good":
            continue
        ok = True
        for idx, cmap in enumerate(cat_maps):
            if idx == target_idx:
                continue
            if status_kind(cmap.get(key)) != "bad":
                ok = False
                break
        if ok:
            return key
    return None


def audio_section_multi(ckpts: list[dict[str, Any]], html_name: str) -> str:
    cats = audio_categories(ckpts)
    if not cats:
        return "<h2>Audio Examples</h2><p class=\"not-run\">No audio examples available.</p>"
    nav = "".join(
        f'<a class="cat-pill" href="#audio-{i}">{h(audio_dataset_label(cat))}: {h(category_label(ckpts, cat).split(" - ", 1)[-1])}</a>'
        for i, cat in enumerate(cats)
    )
    parts = [
        '<h2 id="audio-top">Audio Examples</h2>',
        '<p class="dim">Categories with multiple dirs search for contrastive samples where exactly one checkpoint is good and every other checkpoint is bad. Categories with one dir use the single-run example view.</p>',
        f'<div class="category-nav">{nav}</div>',
    ]
    cats_by_dataset: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for cat_idx, cat in enumerate(cats):
        cats_by_dataset[audio_dataset_key(cat)].append((cat_idx, cat))

    for dataset, dataset_cats in cats_by_dataset.items():
        parts.append(f'<details class="bench-detail audio-detail" open><summary>{h(BENCH_LABELS.get(dataset, dataset))}</summary><div>')
        for cat_idx, cat in dataset_cats:
            active_idxes = [
                idx for idx, ckpt in enumerate(ckpts) if (ckpt.get("audio_examples") or {}).get(cat)
            ]
            if len(active_idxes) == 1:
                parts.append(audio_category_single_html(ckpts[active_idxes[0]], cat, cat_idx))
                continue
            active_ckpts = [ckpts[idx] for idx in active_idxes]
            cat_maps = [
                {str(ex.get("key", idx)): ex for idx, ex in enumerate((ckpt.get("audio_examples") or {}).get(cat) or [])}
                for ckpt in active_ckpts
            ]
            parts.append(f'<section class="ex-section" id="audio-{cat_idx}"><h3>{h(category_label(active_ckpts, cat))}</h3>')
            for category in categories_for_maps(cat_maps):
                filtered_maps = maps_for_category(cat_maps, category)
                source = next(
                    (category_source_value(ex) for cmap in filtered_maps for ex in cmap.values()),
                    "uncategorized",
                )
                parts.append(category_heading(category, source))
                any_card = False
                for target_idx, target_ckpt in enumerate(active_ckpts):
                    key = contrast_key_for_target(filtered_maps, target_idx)
                    if key is None:
                        parts.append(
                            f'<div class="ex-card"><div class="ex-header"><strong>{h(target_ckpt["name"])}</strong> '
                            '<span class="dim">No good-vs-bad contrastive sample found in this category.</span></div></div>'
                        )
                        continue
                    any_card = True
                    ctx = filtered_maps[target_idx][key]
                    rows = "".join(
                        example_response(cmap.get(key), active_ckpts[i]["name"], COLORS[i % len(COLORS)])
                        for i, cmap in enumerate(filtered_maps)
                    )
                    parts.append(
                        f'<div class="ex-card"><div class="ex-header"><strong>{h(target_ckpt["name"])}</strong> '
                        f'<span class="dim">only-good example</span><code>{h(key)}</code></div>'
                        f'<div class="ex-body-cmp"><div class="ex-left">{example_context(ctx)}</div>'
                        f'<div class="ex-right-cmp">{rows}</div></div></div>'
                    )
                if not any_card:
                    parts.append('<p class="not-run">No strict contrastive examples were found for this category.</p>')
            parts.append('<a class="back-link" href="#audio-top">Back to Audio Examples</a></section>')
        parts.append("</div></details>")
    return "\n".join(parts)


def audio_section_single(ckpt: dict[str, Any], html_name: str) -> str:
    cats = audio_categories([ckpt])
    if not cats:
        return "<h2>Audio Examples</h2><p class=\"not-run\">No audio examples available.</p>"
    nav = "".join(
        f'<a class="cat-pill" href="#audio-{i}">{h(audio_dataset_label(cat))}: {h(category_label([ckpt], cat).split(" - ", 1)[-1])}</a>'
        for i, cat in enumerate(cats)
    )
    parts = [
        '<h2 id="audio-top">Audio Examples</h2>',
        '<p class="dim">Examples are grouped by category and performance label.</p>',
        f'<div class="category-nav">{nav}</div>',
    ]
    cats_by_dataset: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for cat_idx, cat in enumerate(cats):
        cats_by_dataset[audio_dataset_key(cat)].append((cat_idx, cat))
    for dataset, dataset_cats in cats_by_dataset.items():
        parts.append(f'<details class="bench-detail audio-detail" open><summary>{h(BENCH_LABELS.get(dataset, dataset))}</summary><div>')
        for cat_idx, cat in dataset_cats:
            parts.append(audio_category_single_html(ckpt, cat, cat_idx))
        parts.append("</div></details>")
    return "\n".join(parts)


CSS = """
:root { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --bd:#30363d;
        --tx:#e6edf3; --tx2:#8b949e; --ac:#58a6ff; --gr:#3fb950;
        --ye:#d29922; --re:#f85149; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--tx);
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       line-height:1.5; padding:24px; max-width:1440px; margin:0 auto; }
h1 { font-size:1.7em; margin:0 0 6px; }
h2 { font-size:1.1em; font-weight:600; color:var(--ac); margin:26px 0 10px;
     border-bottom:1px solid var(--bd); padding-bottom:6px; }
h3 { font-size:0.95em; font-weight:600; color:var(--ac); margin:10px 0; }
h4 { font-size:0.9em; margin:14px 0 8px; color:var(--tx); }
a { color:var(--ac); }
code { color:#a5d6ff; font-size:0.82em; }
.dim { color:var(--tx2); font-size:0.88em; }
.not-run { color:var(--tx2); font-style:italic; }
.meta-row { display:flex; gap:18px; flex-wrap:wrap; align-items:center; margin:8px 0 14px; }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.ckpt-pill, .cat-pill { display:inline-block; padding:4px 12px; border-radius:999px;
                        font-size:0.84em; font-weight:600; border:1px solid var(--bd);
                        background:var(--bg2); color:var(--tx); text-decoration:none; }
.category-nav { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 18px; }
.cmp-table { width:100%; border-collapse:collapse; font-size:0.86em; }
.cmp-table th, .cmp-table td { padding:7px 10px; border:1px solid var(--bd); vertical-align:top; }
.cmp-table thead th { background:var(--bg3); }
.cmp-table tbody tr:hover { background:var(--bg2); }
.mode-score { display:block; white-space:nowrap; }
.mode-tag { color:var(--tx2); font-size:0.78em; margin-right:4px; }
.metric-file-cell { min-width:240px; max-width:440px; }
.metric-file-entry { display:flex; align-items:center; gap:6px; margin:2px 0; min-width:0; }
.metric-file-run { color:var(--tx2); font-size:0.76em; min-width:64px; max-width:120px;
                   overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.path-chip { display:inline-block; max-width:260px; overflow:hidden; text-overflow:ellipsis;
             white-space:nowrap; vertical-align:bottom; border:1px solid var(--bd);
             border-radius:4px; padding:1px 5px; background:var(--bg); color:#a5d6ff; }
.path-chip:hover { max-width:min(900px, 86vw); white-space:normal; overflow:visible;
                   position:relative; z-index:20; background:var(--bg3); }
.path-line { display:flex; align-items:center; gap:6px; margin-top:5px; min-width:0; }
.path-more { color:var(--tx2); font-size:0.78em; white-space:nowrap; }
.bench-detail { margin:14px 0; border:1px solid var(--bd); border-radius:8px; background:var(--bg2); }
.bench-detail summary { cursor:pointer; padding:10px 14px; color:var(--ac); font-weight:700; }
.bench-detail > div, .bench-detail > .bar-grid { padding:0 12px 12px; }
.bar-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; }
.chart-wrap { background:var(--bg); border:1px solid var(--bd); border-radius:8px; padding:12px; min-width:0; }
.chart-box { height:220px; }
.static-chart { min-height:230px; display:flex; flex-direction:column; gap:8px; }
.static-y-label, .static-x-label { color:var(--tx2); font-size:0.76em; text-align:center; }
.static-bars { flex:1; display:flex; align-items:end; justify-content:center; gap:10px; min-height:170px;
               border-left:1px solid var(--bd); border-bottom:1px solid var(--bd); padding:8px 8px 0; }
.static-bar-item { flex:1; min-width:42px; max-width:95px; display:flex; flex-direction:column; align-items:center;
                   justify-content:end; gap:5px; height:100%; }
.static-bar-value { color:var(--tx); font-size:0.72em; min-height:1.1em; text-align:center; overflow-wrap:anywhere; }
.static-bar-shell { height:130px; width:100%; display:flex; align-items:end; justify-content:center; }
.static-bar { width:70%; min-height:2px; border-radius:4px 4px 0 0; }
.static-bar-label { color:var(--tx2); font-size:0.72em; width:100%; text-align:center; overflow:hidden;
                    text-overflow:ellipsis; white-space:nowrap; }
.aplayer { width:100%; height:34px; margin-top:5px; border-radius:4px; accent-color:var(--ac); }
.ex-section { margin:16px 0; }
.audio-detail > div { padding:0 12px 12px; }
.ex-card { background:var(--bg2); border:1px solid var(--bd); border-radius:8px;
           margin-bottom:12px; overflow:hidden; }
.ex-header { background:var(--bg3); padding:10px 14px; font-size:0.88em;
             border-bottom:1px solid var(--bd); display:flex; align-items:center;
             gap:8px; flex-wrap:wrap; }
.ex-body-cmp { display:grid; grid-template-columns:300px 1fr; }
.ex-left { padding:14px; border-right:1px solid var(--bd); min-width:0; }
.ex-right-cmp { padding:14px; min-width:0; }
.audio-row { margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid var(--bg3); }
.audio-row:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.ckpt-badge { display:inline-block; padding:2px 8px; border-radius:4px;
              font-size:0.78em; font-weight:700; margin-bottom:5px; }
.bubble { border-radius:6px; padding:9px 11px; font-size:0.86em; margin:6px 0; overflow-wrap:anywhere; }
.bubble-q { background:#0d1f3a; border-left:3px solid var(--ac); }
.bubble-a { background:#0d2a1a; border-left:3px solid var(--gr); white-space:pre-wrap; }
.badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.78em; font-weight:700; }
.mini-label { color:var(--tx2); font-size:0.78em; font-weight:700; text-transform:uppercase; margin-top:8px; }
.category-group-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; color:var(--tx); }
.status-title { color:var(--tx2); font-size:0.82em; font-weight:700; margin:8px 0 6px; text-transform:uppercase; }
.back-link { display:inline-block; margin:2px 0 14px; color:var(--ac); font-size:0.84em; text-decoration:none; }
.back-link:hover { text-decoration:underline; }
@media(max-width:1100px) { .bar-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } }
@media(max-width:800px) {
  body { padding:14px; }
  .bar-grid { grid-template-columns:1fr; }
  .ex-body-cmp { grid-template-columns:1fr; }
  .ex-left { border-right:none; border-bottom:1px solid var(--bd); }
}
"""


def legend_html(ckpts: list[dict[str, Any]]) -> str:
    return "".join(
        f'<span class="ckpt-pill" style="border-color:{COLORS[i % len(COLORS)]};color:{COLORS[i % len(COLORS)]}">'
        f'{h(ck["name"])} <span class="dim">({h(ck.get("commit") or "?")})</span></span>'
        for i, ck in enumerate(ckpts)
    )


def page_html(ckpts: list[dict[str, Any]], out_html: Path) -> str:
    is_multi = len(ckpts) > 1
    today = date.today().isoformat()
    title = "S2S FC Eval - Checkpoint Comparison" if is_multi else f"S2S FC Eval - {ckpts[0]['name']}"
    charts_html = ""
    charts_js = ""
    if is_multi:
        charts_html, charts_js = bar_charts(ckpts)
        details = dataset_detailed_sections(ckpts)
        audio = audio_section_multi(ckpts, out_html.stem)
    else:
        details = single_tables(ckpts[0])
        audio = audio_section_single(ckpts[0], out_html.stem)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{h(title)}</h1>
<div class="meta-row">
  <span class="dim">Generated: <strong>{h(today)}</strong></span>
  <span class="dim">Runs: <strong>{len(ckpts)}</strong></span>
  <span class="dim">Output: <strong>{h(str(out_html))}</strong></span>
</div>
<div class="legend">{legend_html(ckpts)}</div>

<h2>Summary</h2>
{summary_table(ckpts)}

{charts_html}

{details}

{audio}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_dirs",
        nargs="+",
        required=True,
        help="One or more checkpoint metric directories. Existing sidecar JSON files are accepted for compatibility.",
    )
    parser.add_argument(
        "--name",
        default="visual.html",
        help="Output HTML name or path. If no directory is given, writes under asset/. Default: visual.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [Path(p).expanduser() for p in args.metrics_dirs]
    out_html = output_html_path(args.name)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    sidecars = sidecar_paths(out_html, sources)
    ckpts: list[dict[str, Any]] = []
    for source, sidecar in zip(sources, sidecars):
        if not source.exists():
            sys.exit(f"metrics_dir does not exist: {source}")
        ckpt = build_checkpoint(source)
        warn_category_fallbacks(ckpt)
        ckpts.append(ckpt)
        if sidecar is not None:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump(ckpt, f, indent=2, ensure_ascii=False)
            print(f"Sidecar: {sidecar}")

    html = page_html(ckpts, out_html)
    out_html.write_text(html, encoding="utf-8")
    print(f"HTML: {out_html}")
    print(f"View with: bash scripts/serve_scorecard.sh --name {out_html.stem}")


if __name__ == "__main__":
    main()
