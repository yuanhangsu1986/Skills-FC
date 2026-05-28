#!/usr/bin/env python3
"""Visualize S2S FC evaluation metrics for one or more checkpoints.

Examples:
  python3 scripts/visualize_metrics.py --metrics_dirs /path/to/run --name visual
  python3 scripts/visualize_metrics.py --metrics_dirs /path/to/ckpt_a /path/to/ckpt_b --name comparison
  # mix a list-file (one path per line) with extra dirs on the command line:
  python3 scripts/visualize_metrics.py --metrics_dirs ckpt_list.txt /path/to/run_b --name combined

Each --metrics_dirs entry is treated as:
  - a directory or sidecar .json   → used as-is
  - a non-JSON regular file        → read as a list (one path per line; '#' comments OK)
  - anything else                  → kept and reported as missing downstream

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
import textwrap
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
    """Per-source sidecar file path. Naming convention:

      <html_stem>_<basename>.json

    where <basename> is the source path's basename. If two non-JSON sources
    share the same basename, fall back to encoding the full POSIX path with
    `/` → `-` so the resulting filename is unique. JSON sources (already
    sidecars) return None — the caller doesn't rewrite them.
    """
    paths: list[Path | None] = []
    # Count basenames among non-JSON sources to detect collisions.
    base_counts: dict[str, int] = {}
    for s in sources:
        if s.suffix.lower() == ".json":
            continue
        b = s.name or "ckpt"
        base_counts[b] = base_counts.get(b, 0) + 1
    used: set[str] = set()
    for idx, source in enumerate(sources, start=1):
        if source.suffix.lower() == ".json":
            paths.append(None)
            continue
        base_name = source.name or f"ckpt_{idx}"
        if base_counts.get(base_name, 0) > 1:
            # Collision: encode full path with `/` → `-` to disambiguate.
            full = source.as_posix().lstrip("/").replace("/", "-")
            suffix = safe_name(full, f"ckpt_{idx}")
        else:
            suffix = safe_name(base_name, f"ckpt_{idx}")
        candidate = out_html.with_name(f"{out_html.stem}_{suffix}.json")
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
                "judgment_source": "fallback",
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


def compute_vb_aggregate(metrics: dict[str, Any]) -> dict[str, float | None]:
    """VoiceBench aggregate per mode, matching the AJ-column formula in
    asset/Voice_bench_FDB_evaluation_fc.xlsx (Benchmark sheet):

        (P + R + Y + AD + AH + AI + 20*(T + AB + AF)) / COUNTA(P,R,Y,AD,AH,AI,T,AB,AF)

    Component → metric key:
      P  = vb_mcq.openbookqa.acc          R  = vb_mcq.mmsu.acc
      AD = vb_mcq.bbh.acc                 AH = vb_nonmcq.ifeval.final (×100 if 0-1)
      AI = vb_nonmcq.advbench.refusal_rate  (×100 if 0-1)
      Y  = mean(vb_nonmcq.sd_qa.panda, vb_nonmcq.sd_qa.gpt)
      T  = vb_nonmcq.commoneval.gpt   (0-5 scale, ×20 → 0-100)
      AB = vb_nonmcq.alpacaeval_full.gpt
      AF = vb_nonmcq.wildvoice.gpt
    """
    def _num(d, *keys):
        if not isinstance(d, dict):
            return None
        for k in keys:
            v = d.get(k)
            if v is None: continue
            try: return float(v)
            except (TypeError, ValueError): return None
        return None
    def _pct(v):
        if v is None: return None
        # Heuristic: turn 0-1 fractions into 0-100; leave 0-100 numbers alone.
        return v * 100.0 if 0.0 <= v <= 1.0 else v

    modes = set((metrics.get("vb_mcq") or {}).keys()) | set((metrics.get("vb_nonmcq") or {}).keys())
    out: dict[str, float | None] = {}
    for mode in modes:
        vm = (metrics.get("vb_mcq")    or {}).get(mode, {}) or {}
        vn = (metrics.get("vb_nonmcq") or {}).get(mode, {}) or {}
        P  = _num(vm.get("openbookqa"), "acc", "accuracy")
        R  = _num(vm.get("mmsu"),       "acc", "accuracy")
        AD = _num(vm.get("bbh"),        "acc", "accuracy")
        T  = _num(vn.get("commoneval"),      "gpt")
        AB = _num(vn.get("alpacaeval_full"), "gpt")
        AF = _num(vn.get("wildvoice"),       "gpt")
        AH = _pct(_num(vn.get("ifeval"),   "final", "strict-prompt", "loose-prompt"))
        AI = _pct(_num(vn.get("advbench"), "refusal_rate", "refusal"))
        sp = _num(vn.get("sd_qa"), "panda")
        sg = _num(vn.get("sd_qa"), "gpt")
        if sp is not None and sg is not None: Y = (sp + sg) / 2
        elif sp is not None:                  Y = sp
        elif sg is not None:                  Y = sg
        else:                                 Y = None
        comps  = [P, R, Y, AD, AH, AI, T, AB, AF]
        counta = sum(1 for v in comps if v is not None)
        if counta == 0:
            out[mode] = None; continue
        total = 0.0
        for v in (P, R, Y, AD, AH, AI):
            if v is not None: total += v
        for v in (T, AB, AF):
            if v is not None: total += 20.0 * v
        out[mode] = round(total / counta, 2)
    return out


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


def _deep_merge_prefer_first(into: dict, override: dict) -> None:
    """Deep-merge `override` into `into`, keeping existing leaves in `into`.

    A key from `override` is added to `into` only if `into` lacks it. When both
    sides have the same key and both values are dicts, recurse so older commits
    can supply sub-keys absent from the latest. Non-dict leaves already in
    `into` are never overwritten (latest-wins-per-leaf).
    """
    if not isinstance(into, dict) or not isinstance(override, dict):
        return
    for k, v in override.items():
        if k not in into:
            into[k] = v
        elif isinstance(v, dict) and isinstance(into[k], dict):
            _deep_merge_prefer_first(into[k], v)


def _ordered_bench_candidates(source: Path, bench: str) -> list[Path]:
    """Return all `<bench>_<commit>` candidate dirs directly under `source`,
    ordered latest-first.

    - The `<commit>` segment must NOT contain an underscore (rejects trailing
      `_`-suffixed dirs and `fdb_v3_chen_chen_*` style names).
    - The bench dir must contain at least one readable split.
    - Ordering uses `select_latest_candidate` (git rank, mtime fallback) by
      repeatedly picking the latest from the remaining set.

    If `source` itself looks like a direct benchmark dir (single-bench layout
    that is its own bench dir), it is returned as the sole candidate.

    Note: this only scans `source/` directly — it does NOT auto-traverse into
    `source/incremental`, `source/offline`, etc. If the caller wants those
    layouts handled (e.g. the fake_rnnt convention), they should provide a
    pre-built sidecar JSON or point `--metrics_dirs` at the appropriate
    subfolder directly.
    """
    if is_direct_benchmark_dir(source, bench):
        return [source]

    parents = [source]
    candidates: list[Path] = []
    seen: set[str] = set()
    for parent in parents:
        if not parent.exists() or not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            matched_prefix: str | None = None
            for prefix in BENCH_PREFIXES[bench]:
                if benchmark_name_matches(child.name, bench, prefix):
                    matched_prefix = prefix
                    break
            if matched_prefix is None:
                continue
            # The commit segment is whatever follows `<prefix>_`. Reject if it
            # contains an underscore (e.g. `bba_7f5f2792_`, `fdb_v3_chen_chen_*`).
            commit_segment = child.name[len(matched_prefix) + 1 :] if child.name != matched_prefix else ""
            if "_" in commit_segment:
                continue
            if not is_direct_benchmark_dir(child, bench):
                continue
            key = str(child)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(child)

    if not candidates:
        return []

    ordered: list[Path] = []
    remaining = list(candidates)
    while remaining:
        latest = select_latest_candidate(remaining)
        ordered.append(latest)
        remaining = [c for c in remaining if c != latest]
    return ordered


def load_metrics_from_dir(source: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    bench_dirs: dict[str, Path] = {}
    metric_sources: dict[str, Any] = {}
    for bench in BENCHMARKS:
        ordered = _ordered_bench_candidates(source, bench)
        if not ordered:
            continue
        bench_dirs[bench] = ordered[0]
        for split in BENCH_SPLITS[bench]:
            # Collect per-candidate raw dicts latest-first, then merge with
            # earlier (latest) commits taking precedence at the leaf level.
            merged_raw: dict[str, Any] | None = None
            primary_metrics_file: Path | None = None
            for cand in ordered:
                metrics_file = next(
                    (p for p in metric_file_candidates(cand, bench, split) if p.exists()),
                    None,
                )
                if not metrics_file:
                    continue
                try:
                    raw = load_json(metrics_file)
                except Exception as exc:
                    print(f"warning: could not read {metrics_file}: {exc}", file=sys.stderr)
                    continue
                if merged_raw is None:
                    merged_raw = raw
                    primary_metrics_file = metrics_file
                else:
                    _deep_merge_prefer_first(merged_raw, raw)
            if merged_raw is None or primary_metrics_file is None:
                continue
            modes = extract_metric_modes(
                merged_raw, metric_key_names(bench, split), infer_mode(primary_metrics_file)
            )
            for mode, raw_metrics in modes.items():
                flat = numeric_leaves(raw_metrics)
                if not flat:
                    continue
                metrics.setdefault(bench, {}).setdefault(mode, {})[split] = flat
                metric_sources.setdefault(bench, {}).setdefault(mode, {})[split] = cluster_path(
                    primary_metrics_file
                )
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
        f'<th class="ckpt-col" style="color:{COLORS[i % len(COLORS)]}">'
        f'{h(ck["name"])}<br><small style="font-weight:400;color:var(--tx2)">{h(ck.get("commit", ""))}</small></th>'
        for i, ck in enumerate(ckpts)
    )
    rows = []
    # Top-of-table synthetic row: VoiceBench Aggregate (vb_mcq+vb_nonmcq AJ formula).
    agg_cells = []
    for ckpt in ckpts:
        agg = compute_vb_aggregate(ckpt.get("metrics") or {})
        cell = formatted_mode_values(agg, "score", "vb_nonmcq")
        agg_cells.append(f'<td style="text-align:center;font-weight:700">{cell}</td>')
    rows.append(
        '<tr class="bench-average" data-bench="__vb_aggregate__">'
        '<td><strong>VoiceBench Aggregate</strong></td>'
        '<td><em>AJ formula</em></td>'
        + "".join(agg_cells)
        + '<td class="metric-file-cell">-</td></tr>'
    )
    for bench in BENCHMARKS:
        metrics = summary_metrics_for_bench(ckpts, bench)
        if not metrics:
            metrics = ["headline"]
        single_metric = len(metrics) == 1
        # Add an extra "average" row for benches without a native aggregate subtest.
        # Value comes straight from compute_headlines() — see primary_split_score for
        # per-subtest score rules and the headline-averaging logic in compute_headlines.
        bench_rows: list[tuple[str, bool]] = []
        adds_average = bench != "bba"
        if adds_average:
            bench_rows.append(("__average__", True))
            # The synthetic "headline" placeholder duplicates the average; drop it.
            bench_rows.extend((m, False) for m in metrics if m != "headline")
        else:
            bench_rows.extend((m, False) for m in metrics)
        n_collapsed = sum(1 for _, is_avg in bench_rows if not is_avg) if adds_average else 0
        for idx, (metric, is_avg) in enumerate(bench_rows):
            cells = []
            for ckpt in ckpts:
                if is_avg or metric == "headline":
                    values_by_mode = (ckpt.get("headlines") or {}).get(bench) or {}
                    cell = formatted_mode_values(values_by_mode, "score", bench)
                else:
                    cell = formatted_mode_values(aggregate_metric_for_ckpt(ckpt, bench, metric), metric, bench)
                selected_commit = selected_commit_for(ckpt, bench)
                commit_html = f'<div class="dim">{h(selected_commit)}</div>' if selected_commit and idx == 0 else ""
                cells.append(f'<td style="text-align:center;font-weight:700">{cell}{commit_html}</td>')
            dataset_cell = f"<strong>{h(BENCH_LABELS[bench])}</strong>" if idx == 0 else ""
            if is_avg:
                toggle = (f' <button class="metric-toggle" type="button" data-bench="{h(bench)}">'
                          f'▾ show {n_collapsed} more</button>') if n_collapsed > 0 else ""
                metric_cell = f"<em>average</em>{toggle}"
            elif single_metric and metric == "headline":
                metric_cell = ""
            else:
                metric_cell = h(metric)
            files_cell = '<td class="metric-file-cell">-</td>' if is_avg else metric_files_cell(ckpts, bench, metric)
            # Tag rows for the JS toggle: hidden metric rows for benches with an average row.
            if adds_average and not is_avg:
                tr_open = f'<tr class="metric-row" data-bench="{h(bench)}" hidden>'
            elif adds_average and is_avg:
                tr_open = f'<tr class="bench-average" data-bench="{h(bench)}">'
            else:
                tr_open = "<tr>"
            rows.append(f"{tr_open}<td>{dataset_cell}</td><td>{metric_cell}</td>{''.join(cells)}{files_cell}</tr>")
    return (
        '<div style="overflow-x:auto"><table class="cmp-table">'
        f"<thead><tr><th>Dataset</th><th>Metric</th>{hdrs}<th>Metric files</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
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
        f'<th class="ckpt-col" style="color:{COLORS[i % len(COLORS)]};text-align:center">{h(ck["name"])}</th>'
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
    # Top-of-section: VoiceBench Aggregate chart across all ckpts.
    if len(ckpts) >= 2:
        agg_labels = [ck["name"] for ck in ckpts]
        agg_colors = [COLORS[i % len(COLORS)] for i in range(len(ckpts))]
        agg_values: list[float | None] = []
        for ck in ckpts:
            agg = compute_vb_aggregate(ck.get("metrics") or {})
            pick = None
            for m in KNOWN_MODES:
                if agg.get(m) is not None: pick = agg[m]; break
            if pick is None:
                for v in agg.values():
                    if v is not None: pick = v; break
            agg_values.append(pick)
        if sum(v is not None for v in agg_values) >= 2:
            chart = static_bar_chart(agg_labels, agg_values, agg_colors, "score")
            if chart:
                html_parts.append(
                    '<details class="bench-detail" open><summary>VoiceBench Aggregate</summary>'
                    f'<div class="bar-grid"><div class="chart-wrap"><h3>aggregate (AJ formula)</h3>{chart}</div></div>'
                    '</details>'
                )
    for bench in BENCHMARKS:
        active_ckpts = [ckpt for ckpt in ckpts if has_bench_data(ckpt, bench)]
        if len(active_ckpts) < 2:
            continue
        labels = [ck["name"] for ck in active_ckpts]
        colors = [COLORS[i % len(COLORS)] for i in range(len(active_ckpts))]
        # Per-bench "average" chart from the precomputed headline (same value as the
        # average row in the summary table). Skipped for bba (built-in aggregate).
        avg_card = ""
        if bench != "bba":
            avg_values: list[float | None] = []
            for ck in active_ckpts:
                hd = (ck.get("headlines") or {}).get(bench) or {}
                pick = None
                for m in KNOWN_MODES:
                    if hd.get(m) is not None:
                        pick = hd[m]
                        break
                if pick is None:
                    for v in hd.values():
                        if v is not None:
                            pick = v
                            break
                avg_values.append(pick)
            if sum(v is not None for v in avg_values) >= 2:
                chart = static_bar_chart(labels, avg_values, colors, "score")
                if chart:
                    avg_card = f'<div class="chart-wrap"><h3>average</h3>{chart}</div>'
        # Other metric charts (one per (split, metric, mode)) — same as before.
        rows = [row for row in iter_display_metric_rows(active_ckpts, bench) if is_display_metric_for_bench(bench, row[1])]
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
        if not avg_card and not cards:
            continue
        # Assemble the bench section: average shown first; other charts collapsed.
        section = [f'<details class="bench-detail" open><summary>{h(BENCH_LABELS[bench])}</summary>']
        if avg_card:
            section.append(f'<div class="bar-grid">{avg_card}</div>')
        if cards:
            if avg_card:
                section.append(
                    f'<details class="bench-detail"><summary>Other metrics ({len(cards)})</summary>'
                    f'<div class="bar-grid">{"".join(cards)}</div></details>'
                )
            else:
                section.append(f'<div class="bar-grid">{"".join(cards)}</div>')
        section.append("</details>")
        html_parts.append("".join(section))
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
    main = f'<span class="badge" style="background:{color}20;color:{color};border:1px solid {color}">{h(label)}</span>'
    if (example or {}).get("judgment_source") == "fallback":
        amber = "var(--ye)"
        title = "Rule-based fallback (no LLM judgment available); status may be unreliable."
        main += (
            f'<span class="badge" title="{h(title)}" '
            f'style="background:{amber}20;color:{amber};border:1px dashed {amber};margin-left:4px">low conf</span>'
        )
    return main


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

/* Metric-row toggle button on the "average" row of each dataset in the summary table. */
.metric-toggle {
  margin-left: 8px; padding: 2px 6px; font-size: 0.82em; font-weight: 500;
  background: var(--bg3); color: var(--ac); border: 1px solid var(--bd);
  border-radius: 4px; cursor: pointer; vertical-align: middle;
}
.metric-toggle:hover { background: var(--bg2); }

/* Ckpt-name column headers wrap onto multiple lines so many columns can fit. */
.cmp-table th.ckpt-col {
  white-space:normal; word-break:break-all; overflow-wrap:break-word;
  max-width:150px; min-width:70px; line-height:1.25; font-size:0.82em;
  vertical-align:bottom;
}

/* Many-ckpts adaptive mode — toggled by Python via body.many-ckpts when ≥6 ckpts.
   Goal: keep tables/plots readable when there are too many columns for the default
   side-by-side layout to fit on screen. */
body.many-ckpts { max-width:100%; padding:16px; }
body.many-ckpts .cmp-table { font-size:0.78em; }
body.many-ckpts .cmp-table th,
body.many-ckpts .cmp-table td { padding:4px 6px; }
body.many-ckpts .cmp-table th.ckpt-col {
  max-width:110px; min-width:60px; font-size:0.72em;
}
body.many-ckpts .metric-file-cell { max-width:240px; }
/* Bar plots: 1 plot per row, vertical x-axis labels, narrower bars. */
body.many-ckpts .bar-grid { grid-template-columns:1fr; }
body.many-ckpts .static-chart { min-height:280px; }
body.many-ckpts .static-bars {
  gap:4px; padding:6px 6px 4px; justify-content:start;
  overflow-x:auto;
}
body.many-ckpts .static-bar-item {
  min-width:28px; max-width:60px; flex:1 0 28px; padding-bottom:0;
}
body.many-ckpts .static-bar-shell { height:120px; }
body.many-ckpts .static-bar { width:80%; }
body.many-ckpts .static-bar-value { font-size:0.66em; }
body.many-ckpts .static-bar-label {
  writing-mode:vertical-rl; transform:rotate(180deg);
  white-space:nowrap; text-overflow:clip; overflow:visible;
  height:90px; width:auto; margin:4px auto 0;
  font-size:0.7em; line-height:1; text-align:right;
}
body.many-ckpts .chart-wrap { padding:8px; }
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
    # Audio Examples section is intentionally omitted in default mode; the
    # `--html_mode report` mode renders curated lone-winner audio cards.
    audio = ""
    if is_multi:
        charts_html, charts_js = bar_charts(ckpts)
        details = dataset_detailed_sections(ckpts)
    else:
        details = single_tables(ckpts[0])
    body_class = "many-ckpts" if len(ckpts) >= 6 else ""
    body_open = f'<body class="{body_class}">' if body_class else "<body>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(title)}</title>
<style>{CSS}</style>
</head>
{body_open}
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
<script>
// Toggle hidden metric rows for each dataset in the summary table.
document.querySelectorAll('.metric-toggle').forEach(function(btn) {{
  btn.addEventListener('click', function(e) {{
    e.preventDefault();
    var bench = btn.dataset.bench;
    var rows = document.querySelectorAll('tr.metric-row[data-bench="' + bench + '"]');
    if (!rows.length) return;
    var willShow = rows[0].hidden;
    rows.forEach(function(r) {{ r.hidden = !willShow; }});
    btn.textContent = willShow ? '▴ hide metrics' : '▾ show ' + rows.length + ' more';
  }});
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Report-mode HTML rendering (`--html_mode report`).
#
# Renders a dark-theme dashboard styled after `asset/report_may20.html`:
#   - Top-N gold/silver/bronze cards.
#   - Navigation bar with anchors to every major section.
#   - Figures 1-7 (Chart.js bar / grouped bar).
#   - Recommendations.
#   - Detailed D1-D12 sections (TOC + JS-rendered D12 numeric summary table).
#   - Curated lone-winner Audio Examples (per (bench, subtest), up to N
#     contrastive cards where each top-N ckpt is the sole "good").
#   - Back-to-top buttons at every section boundary.
# All helpers are private to this module and grouped together.
# ---------------------------------------------------------------------------
_REPORT_RANK_COLORS = ["#ffd700", "#c0c0c0", "#cd7f32", "#58a6ff", "#bc8cff", "#39c5cf"]
_REPORT_RANK_MEDALS = ["\U0001F947", "\U0001F948", "\U0001F949"]  # gold/silver/bronze

_REPORT_BENCH_PRETTY = {
    "vb_nonmcq": "VoiceBench non-MCQ",
    "vb_mcq": "VoiceBench MCQ",
    "bba": "BigBench Audio",
    "fdb_v1": "FDB v1",
    "fdb_v1_5": "FDB v1.5",
    "fdb_v3": "FDB v3",
    "conv_behav": "Turn Taking Benchmark",
    "bfcl": "BFCL",
}

_REPORT_TOP_CARD_METRICS: list[tuple[str, str, str, bool]] = [
    # (label, data-key, format-spec, reverse?)
    ("BFCL Priority", "BFCL_PRIO", "{:.1f}%", False),
    ("BFCL Simple / Multiple", "_simp_mult", "{:.1f} / {:.1f}", False),
    ("BFCL Irrelevance", "BFCL_IRR", "{:.1f}%", False),
    ("VoiceBench avg", "VB_AVG", "{:.1f}", False),
    ("FDB v1 TOR", "FDBV1_TOR_SUMM", "{:.1f}", False),
    ("FDB v1.5 headline", "FDBV15_HL", "{:.2f}", False),
    ("BBA headline", "BBA_HL", "{:.1f}", False),
    ("Conv F1", "CONV_F1", "{:.1f}", False),
    ("Adherence (/5)", "FDBV1_ADH", "{:.2f}", False),
    ("Combined Latency (↓ ms)", "FDBV1_LAT_AVG", "{:.0f} ms", True),
]

_REPORT_STYLE_BLOCK = r"""<style>
:root{
  --bg:#0d1117;--surf:#161b22;--surf2:#21262d;--bord:#30363d;
  --txt:#e6edf3;--muted:#8b949e;
  --green:#3fb950;--blue:#58a6ff;--orange:#d29922;
  --red:#f85149;--purple:#bc8cff;--yellow:#e3b341;--cyan:#39c5cf;
  --gold:#ffd700;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;line-height:1.6;padding:24px;max-width:1500px;margin:0 auto}
h1{font-size:1.7em;color:var(--blue);margin-bottom:4px}
h2{font-size:1.1em;color:var(--orange);margin:32px 0 10px;border-bottom:1px solid var(--bord);padding-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
h3{font-size:.98em;color:var(--blue);margin:16px 0 6px}
h4{font-size:.9em;color:var(--muted);margin:12px 0 4px;font-style:italic}
.subtitle{color:var(--muted);font-size:.85em;margin-bottom:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.card{background:var(--surf);border:1px solid var(--bord);border-radius:8px;padding:16px}
.cw{position:relative;height:260px}
.cw-lg{position:relative;height:340px}
.cw-xl{position:relative;height:420px}
table{width:100%;border-collapse:collapse;font-size:.78em}
th{background:var(--surf2);color:var(--muted);padding:6px 8px;text-align:left;font-weight:600;font-size:.75em;text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0;z-index:1}
td{padding:5px 8px;border-bottom:1px solid var(--bord)}
tr:hover td{background:var(--surf2)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-size:.8em}
.best{color:var(--green);font-weight:700}
.good{color:var(--blue)}
.mid{color:var(--yellow)}
.low{color:var(--red)}
.na{color:var(--muted)}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.72em;font-weight:600}
.b1{background:rgba(255,215,0,.15);color:var(--gold);border:1px solid var(--gold)}
.b2{background:rgba(192,192,192,.12);color:#c0c0c0;border:1px solid #c0c0c0}
.b3{background:rgba(205,127,50,.12);color:#cd7f32;border:1px solid #cd7f32}
.insight{background:rgba(63,185,80,.06);border-left:3px solid var(--green);border-radius:0 6px 6px 0;padding:9px 14px;margin:10px 0;font-size:.87em}
.warn{background:rgba(248,81,73,.06);border-left:3px solid var(--red);border-radius:0 6px 6px 0;padding:9px 14px;margin:10px 0;font-size:.87em}
.note{background:rgba(210,153,34,.06);border-left:3px solid var(--orange);border-radius:0 6px 6px 0;padding:9px 14px;margin:10px 0;font-size:.87em}
.rec{padding:10px 14px;margin:8px 0;background:var(--surf2);border-radius:6px;border-left:3px solid var(--purple);font-size:.88em}
.rec strong{color:var(--purple)}
.ov{overflow-x:auto}
p{margin:6px 0;font-size:.88em;color:var(--muted)}
.fig-label{font-size:.8em;color:var(--muted);margin-top:6px;text-align:center}
.toc{background:var(--surf);border:1px solid var(--bord);border-radius:8px;padding:16px;margin-bottom:20px}
.toc h3{color:var(--orange);margin-bottom:10px}
.toc a{color:var(--blue);text-decoration:none;font-size:.88em}
.toc a:hover{text-decoration:underline}
.toc li{margin:3px 0}
.top3-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:16px 0}
.top3-card{border-radius:8px;padding:16px;position:relative}
.rank-badge{position:absolute;top:10px;right:12px;font-size:1.4em}
.top3-exp{font-size:1.2em;font-weight:700;font-family:monospace;margin-bottom:6px}
.top3-row{display:flex;justify-content:space-between;font-size:.82em;margin:3px 0}
.top3-row span:first-child{color:var(--muted)}
section{margin-bottom:30px}
.fignum{font-size:.72em;background:var(--surf2);padding:1px 6px;border-radius:3px;color:var(--muted);margin-left:6px}

/* Audio-Examples + cmp styles */
.aplayer { width:100%; height:34px; margin-top:5px; border-radius:4px; accent-color:var(--blue); }
.ex-section { margin:16px 0; }
.audio-detail > div { padding:0 12px 12px; }
.ex-card { background:var(--surf); border:1px solid var(--bord); border-radius:8px;
           margin-bottom:12px; overflow:hidden; }
.ex-header { background:var(--surf2); padding:10px 14px; font-size:0.88em;
             border-bottom:1px solid var(--bord); display:flex; align-items:center;
             gap:8px; flex-wrap:wrap; }
.ex-body-cmp { display:grid; grid-template-columns:300px 1fr; }
.ex-left { padding:14px; border-right:1px solid var(--bord); min-width:0; }
.ex-right-cmp { padding:14px; min-width:0; }
.audio-row { margin-bottom:10px; padding-bottom:10px; border-bottom:1px solid var(--surf2); }
.audio-row:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.ckpt-badge { display:inline-block; padding:2px 8px; border-radius:4px;
              font-size:0.78em; font-weight:700; margin-bottom:5px; }
.bubble { border-radius:6px; padding:9px 11px; font-size:0.86em; margin:6px 0; overflow-wrap:anywhere; }
.bubble-q { background:#0d1f3a; border-left:3px solid var(--blue); }
.bubble-a { background:#0d2a1a; border-left:3px solid var(--green); white-space:pre-wrap; }
.mini-label { color:var(--muted); font-size:0.78em; font-weight:700; text-transform:uppercase; margin-top:8px; }
.bench-detail { margin:14px 0; border:1px solid var(--bord); border-radius:8px; background:var(--surf); }
.bench-detail summary { cursor:pointer; padding:10px 14px; color:var(--blue); font-weight:700; }
.bench-detail > div { padding:0 12px 12px; }

/* Nav-bar + back-to-top */
.nav-bar { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 24px;
           padding:10px 12px; background:var(--surf); border:1px solid var(--bord);
           border-radius:8px; font-size:.86em }
.nav-bar a { color:var(--blue); text-decoration:none; padding:4px 10px;
             border-radius:4px; background:var(--surf2) }
.nav-bar a:hover { background:var(--bord); text-decoration:none }
.back-to-top { margin:20px 0; text-align:right }
.back-to-top a { color:var(--muted); font-size:.85em; text-decoration:none;
                 padding:4px 10px; border-radius:4px; background:var(--surf) }
.back-to-top a:hover { color:var(--blue); background:var(--surf2) }
</style>
"""

_REPORT_NAV_BAR_HTML = """<nav class="nav-bar" id="top-nav">
  <a href="#top">Top</a>
  <a href="#fig1">Fig 1 — VoiceBench</a>
  <a href="#fig2">Fig 2 — FDB v1</a>
  <a href="#fig3">Fig 3 — FDB v1.5</a>
  <a href="#fig4">Fig 4 — FDB v3</a>
  <a href="#fig5">Fig 5 — BFCL</a>
  <a href="#fig6">Fig 6 — BBA</a>
  <a href="#fig7">Fig 7 — Conv</a>
  <a href="#recs">Recommendations</a>
  <a href="#detailed">Detailed</a>
  <a href="#audio-top">Audio Examples</a>
</nav>"""

_REPORT_BACK_TO_TOP = '<div class="back-to-top"><a href="#top-nav">↑ back to top</a></div>'


def _report_step_from_name(full_name: str) -> int:
    m = re.search(r"step(\d+)", full_name)
    return int(m.group(1)) if m else 0


def _report_exp_from_name(full_name: str) -> str:
    return full_name.split("-", 1)[0]


def _report_short_label(full_name: str) -> str:
    return f"{_report_exp_from_name(full_name)}@{_report_step_from_name(full_name) // 1000}k"


def _report_short_labels_with_disambig(full_names: list[str]) -> list[str]:
    """eXXX@Yk; if multiple ckpts share a label, append *, ^, +, ... in order."""
    suffixes = ["", "*", "^", "+", "#", "&"]
    base = [_report_short_label(n) for n in full_names]
    counts: dict[str, int] = {}
    out: list[str] = []
    for b in base:
        c = counts.get(b, 0)
        out.append(b + (suffixes[c] if c < len(suffixes) else f"!{c}"))
        counts[b] = c + 1
    return out


def _report_as_number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _report_rank_class(value: float | None, all_values: list[float | None],
                       reverse: bool = False) -> str:
    """Return 'best'/'good'/'mid'/'low'/'na' based on rank vs others.

    'reverse=True' means lower-is-better (latency etc).
    """
    if value is None:
        return "na"
    arr = [v for v in all_values if v is not None]
    if not arr:
        return "na"
    arr_sorted = sorted(arr, reverse=not reverse)
    try:
        rank = arr_sorted.index(value)
    except ValueError:
        return "low"
    if rank < 3:
        return "best"
    if rank < 6:
        return "good"
    if rank < 12:
        return "mid"
    return "low"


def _report_color_for_rank(idx: int) -> str:
    return _REPORT_RANK_COLORS[idx] if idx < len(_REPORT_RANK_COLORS) else "#58a6ff"


def _report_audio_url(audio_path: str | None) -> str:
    """Strip CLUSTER: prefix and /lustre root for browser-served URL."""
    if not audio_path:
        return ""
    p = audio_path
    if ":" in p[:32]:
        p = p.split(":", 1)[1]
    if p.startswith("/lustre"):
        p = p[len("/lustre"):]
    return p


def _report_g(metrics: dict[str, Any], *path: str) -> float | None:
    """Walk metrics by dotted path: metrics['vb_nonmcq']['greedy']['commoneval']['gpt']."""
    cur: Any = metrics
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return _report_as_number(cur)


def _report_derive_metrics(ckpts: list[dict[str, Any]]) -> dict[str, list[float | None]]:
    """Compute every per-ckpt array embedded in the HTML report."""
    out: dict[str, list[float | None]] = {}

    def col(*spec):
        return [_report_g(c.get("metrics") or {}, *spec) for c in ckpts]

    # VB aggregate (AJ formula) — reuse the canonical helper.
    vb_avg: list[float | None] = []
    for c in ckpts:
        agg = compute_vb_aggregate(c.get("metrics") or {})
        vb_avg.append(agg.get("greedy") or (next(iter(agg.values()), None) if agg else None))
    out["VB_AVG"] = vb_avg

    out["VB_ALPACA"] = [
        (v * 20) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "alpacaeval", "gpt")) is not None else None
        for c in ckpts
    ]
    out["VB_ALPACA_FULL"] = [
        (v * 20) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "alpacaeval_full", "gpt")) is not None else None
        for c in ckpts
    ]
    out["VB_IFEVAL"] = [
        (v * 100) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "ifeval", "final")) is not None and v <= 1.0
        else _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "ifeval", "final")
        for c in ckpts
    ]
    out["VB_COMMON"] = [
        (v * 20) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "commoneval", "gpt")) is not None else None
        for c in ckpts
    ]
    out["VB_WILD"] = [
        (v * 20) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "wildvoice", "gpt")) is not None else None
        for c in ckpts
    ]
    sdqa_avg: list[float | None] = []
    for c in ckpts:
        m = c.get("metrics") or {}
        p = _report_g(m, "vb_nonmcq", "greedy", "sd_qa", "panda")
        g = _report_g(m, "vb_nonmcq", "greedy", "sd_qa", "gpt")
        if p is not None and g is not None:
            sdqa_avg.append((p + g) / 2)
        elif p is not None:
            sdqa_avg.append(p)
        elif g is not None:
            sdqa_avg.append(g)
        else:
            sdqa_avg.append(None)
    out["VB_SDQA"] = sdqa_avg
    out["VB_SPK"] = [
        (v * 20) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "alpacaeval_speaker", "gpt")) is not None else None
        for c in ckpts
    ]
    out["VB_ADV"] = [
        (v * 100) if (v := _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "advbench", "refusal_rate")) is not None and v <= 1.0
        else _report_g(c.get("metrics") or {}, "vb_nonmcq", "greedy", "advbench", "refusal_rate")
        for c in ckpts
    ]
    out["VB_BBH"] = col("vb_mcq", "greedy", "bbh", "acc")
    out["VB_OPBK"] = col("vb_mcq", "greedy", "openbookqa", "acc")
    out["VB_MMSU"] = col("vb_mcq", "greedy", "mmsu", "acc")

    # BFCL.
    out["BFCL_SIMP"] = col("bfcl", "greedy", "simple", "accuracy")
    out["BFCL_MULT"] = col("bfcl", "greedy", "multiple", "accuracy")
    out["BFCL_PAR"] = col("bfcl", "greedy", "parallel", "accuracy")
    out["BFCL_PM"] = col("bfcl", "greedy", "parallel_multiple", "accuracy")
    out["BFCL_IRR"] = col("bfcl", "greedy", "irrelevance", "accuracy")
    out["BFCL_AVG"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "bfcl", "greedy")
        for c in ckpts
    ]
    # BFCL priority = mean(simple, multiple, irrelevance) (ignore Nones).
    prio: list[float | None] = []
    for s, m, ir in zip(out["BFCL_SIMP"], out["BFCL_MULT"], out["BFCL_IRR"]):
        vals = [v for v in (s, m, ir) if v is not None]
        prio.append(round(sum(vals) / len(vals), 3) if vals else None)
    out["BFCL_PRIO"] = prio

    # FDB v1.
    out["FDBV1_HL"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "fdb_v1", "greedy")
        for c in ckpts
    ]
    out["FDBV1_BC"] = col("fdb_v1", "greedy", "backchannel", "tor_pct")
    out["FDBV1_CAND"] = col("fdb_v1", "greedy", "pause_candor", "tor_pct")
    out["FDBV1_SYN"] = col("fdb_v1", "greedy", "pause_synthetic", "tor_pct")
    out["FDBV1_INTR_TOR"] = col("fdb_v1", "greedy", "interruption", "tor_pct")
    out["FDBV1_INTR_LAT"] = col("fdb_v1", "greedy", "interruption", "latency_ms")
    out["FDBV1_INTR_RAT"] = col("fdb_v1", "greedy", "interruption", "rating")
    out["FDBV1_TT_TOR"] = col("fdb_v1", "greedy", "turn_taking", "tor_pct")
    out["FDBV1_TT_LAT"] = col("fdb_v1", "greedy", "turn_taking", "latency_ms")
    out["FDBV1_JSD"] = col("fdb_v1", "greedy", "backchannel", "jsd")

    tor_summ: list[float | None] = []
    adh: list[float | None] = []
    lat_avg: list[float | None] = []
    for i in range(len(ckpts)):
        sp = out["FDBV1_SYN"][i]
        cp = out["FDBV1_CAND"][i]
        tt = out["FDBV1_TT_TOR"][i]
        intr = out["FDBV1_INTR_TOR"][i]
        if None not in (sp, cp, tt, intr):
            pause_frac = (sp / 100 + cp / 100) / 2  # type: ignore[operator]
            ts = ((1 - pause_frac) + (tt / 100) + (intr / 100)) / 3 * 100  # type: ignore[operator]
            tor_summ.append(round(ts, 1))
        else:
            tor_summ.append(None)
        adh.append(out["FDBV1_INTR_RAT"][i])
        tt_l = out["FDBV1_TT_LAT"][i]
        in_l = out["FDBV1_INTR_LAT"][i]
        if tt_l is not None and in_l is not None:
            lat_avg.append(round((tt_l + in_l) / 2, 2))
        elif tt_l is not None:
            lat_avg.append(tt_l)
        elif in_l is not None:
            lat_avg.append(in_l)
        else:
            lat_avg.append(None)
    out["FDBV1_TOR_SUMM"] = tor_summ
    out["FDBV1_ADH"] = adh
    out["FDBV1_LAT_AVG"] = lat_avg

    # FDB v1.5.
    out["FDBV15_HL"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "fdb_v1_5", "greedy")
        for c in ckpts
    ]
    out["FDBV15_BG_RESP"] = col("fdb_v1_5", "greedy", "background_speech",
                                "behavior_ratios.C_RESPOND")
    out["FDBV15_BG_RES"] = col("fdb_v1_5", "greedy", "background_speech",
                               "behavior_ratios.C_RESUME")
    out["FDBV15_BG_SL"] = col("fdb_v1_5", "greedy", "background_speech", "stop_latency_ms")
    out["FDBV15_TTO_RESP"] = col("fdb_v1_5", "greedy", "talking_to_other",
                                 "behavior_ratios.C_RESPOND")
    out["FDBV15_TTO_RES"] = col("fdb_v1_5", "greedy", "talking_to_other",
                                "behavior_ratios.C_RESUME")
    out["FDBV15_BC_RES"] = col("fdb_v1_5", "greedy", "backchannel",
                               "behavior_ratios.C_RESUME")
    out["FDBV15_INTR_RESP"] = col("fdb_v1_5", "greedy", "interruption",
                                  "behavior_ratios.C_RESPOND")

    # FDB v3.
    out["FDBV3_HL"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "fdb_v3", "greedy")
        for c in ckpts
    ]

    # BBA.
    out["BBA_HL"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "bba", "greedy")
        for c in ckpts
    ]
    out["BBA_FF"] = col("bba", "greedy", "formal_fallacies", "accuracy")
    out["BBA_NAV"] = col("bba", "greedy", "navigate", "accuracy")
    out["BBA_OC"] = col("bba", "greedy", "object_counting", "accuracy")
    out["BBA_WL"] = col("bba", "greedy", "web_of_lies", "accuracy")

    # Conv behavior.
    out["CONV_HL"] = [
        _report_g({"headlines": c.get("headlines")}, "headlines", "conv_behav", "greedy")
        for c in ckpts
    ]
    out["CONV_PREC"] = col("conv_behav", "greedy", "overall", "tt_precision")
    out["CONV_REC"] = col("conv_behav", "greedy", "overall", "tt_recall")
    out["CONV_F1"] = col("conv_behav", "greedy", "overall", "tt_f1")
    out["CONV_CUT"] = col("conv_behav", "greedy", "overall", "cutoff_rate")
    if not any(v is not None for v in out["CONV_F1"]):
        out["CONV_F1"] = out["CONV_HL"][:]

    return out


def _report_js_array(values: list[Any]) -> str:
    """JSON-encode a python list as a compact JS array (None -> null)."""
    def _enc(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            if v != v:  # NaN
                return "null"
            s = f"{v:.6g}"
            return s
        return json.dumps(v, ensure_ascii=False)
    return "[" + ",".join(_enc(v) for v in values) + "]"


def _top_n_cards(picks: list[dict[str, Any]], metric_arrays: dict[str, list[Any]],
                 pick_idxs: list[int]) -> str:
    """Render the row of Top-N ranking cards."""
    cards: list[str] = []
    for r, (pick, idx) in enumerate(zip(picks, pick_idxs)):
        if idx < 0:
            print(f"WARNING: pick {r}'s full_name {pick.get('full_name')!r} "
                  f"not found in metrics ckpts; skipping card", file=sys.stderr)
            continue
        cards.append(_render_top_card(r, pick, metric_arrays, idx))
    return "\n".join(cards)


def _render_top_card(rank: int, pick: dict[str, Any],
                     metric_arrays: dict[str, list[Any]], idx: int) -> str:
    """Render one Top-N card."""
    color = _report_color_for_rank(rank)
    medal = _REPORT_RANK_MEDALS[rank] if rank < len(_REPORT_RANK_MEDALS) else f"#{rank + 1}"
    short = pick.get("short_label", "?")
    subtitle = pick.get("subtitle", "")
    insight = pick.get("insight", "")
    weakness = pick.get("weakness", "")

    rows: list[str] = []
    fdb15_clean = [v for v in metric_arrays["FDBV15_HL"]
                   if v is not None and v <= 1.0]
    for label, key, fmt, reverse in _REPORT_TOP_CARD_METRICS:
        if key == "_simp_mult":
            s = metric_arrays["BFCL_SIMP"][idx]
            m = metric_arrays["BFCL_MULT"][idx]
            if s is None or m is None:
                continue
            arr_s = metric_arrays["BFCL_SIMP"]
            arr_m = metric_arrays["BFCL_MULT"]
            cls = _report_rank_class(((arr_s.index(s) if s in arr_s else 0)
                                      + (arr_m.index(m) if m in arr_m else 0)) / 2,
                                     [float(i) for i in range(len(arr_s))], reverse=True)
            rows.append(
                f'<div class="top3-row"><span>{label}</span>'
                f'<span class="{cls}">{fmt.format(s, m)}</span></div>'
            )
            continue
        val = metric_arrays.get(key, [None] * len(metric_arrays["VB_AVG"]))[idx]
        if val is None:
            continue
        if key == "FDBV15_HL":
            cls = _report_rank_class(val, fdb15_clean, reverse=reverse)
        else:
            cls = _report_rank_class(val, metric_arrays[key], reverse=reverse)
        try:
            text = fmt.format(val)
        except (TypeError, ValueError):
            text = str(val)
        rows.append(
            f'<div class="top3-row"><span>{label}</span>'
            f'<span class="{cls}">{text}</span></div>'
        )

    return (
        f'<div class="top3-card card" style="border-color:{color}">\n'
        f'  <div class="rank-badge">{medal}</div>\n'
        f'  <div class="top3-exp" style="color:{color}">{escape(short)}</div>\n'
        f'  <p style="color:var(--muted);font-size:.8em;margin-bottom:10px">'
        f'{escape(subtitle)}</p>\n'
        + "\n".join("  " + r for r in rows) + "\n"
        f'  <div class="insight" style="margin-top:10px">{escape(insight)}</div>\n'
        f'  <div class="note" style="margin-top:6px">{escape(weakness)}</div>\n'
        f'</div>'
    )


def _curated_audio_section(ckpts: list[dict[str, Any]],
                           picks: list[dict[str, Any]]) -> str:
    """Build the curated lone-winner Audio Examples HTML section.

    For each (bench:split) group and each top-N ckpt, find audio examples
    where that ckpt is the *sole* 'good' status (others are not 'good') and
    emit one ex-card per such finding.
    """
    GREEN, RED = "#3fb950", "#f85149"
    name_to_idx = {c.get("name") or c.get("source", ""): i for i, c in enumerate(ckpts)}
    pick_idxs: list[int] = []
    for p in picks:
        full = p.get("full_name", "")
        pick_idxs.append(name_to_idx.get(full, -1))

    pick_meta = []
    for r, p in enumerate(picks):
        full = p.get("full_name", "")
        pick_meta.append({
            "full": full,
            "label": p.get("short_label",
                           _report_short_label(full) if full else "?"),
            "color": _report_color_for_rank(r),
        })

    # Build group order from picks first; fallback to all ckpts.
    group_order: list[str] = []
    seen: set[str] = set()
    for idx in pick_idxs:
        if idx < 0:
            continue
        ae = (ckpts[idx].get("audio_examples") or {})
        for g in ae.keys():
            if g not in seen:
                seen.add(g)
                group_order.append(g)
    if not group_order:
        for c in ckpts:
            for g in (c.get("audio_examples") or {}).keys():
                if g not in seen:
                    seen.add(g)
                    group_order.append(g)

    grouped: dict[str, list[tuple[str, str]]] = {}
    for g in group_order:
        bench, _, split = g.partition(":")
        grouped.setdefault(bench, []).append((g, split))

    parts: list[str] = [
        '<h2 id="audio-top">Audio Examples</h2>',
        '<p class="subtitle">Contrastive examples: per subtest, '
        f'one card per top-{len(picks)} ckpt where it is the sole &quot;good&quot;.</p>'
    ]

    for bench, items in grouped.items():
        for g, split in items:
            per_ckpt: list[dict[str, dict[str, Any]]] = []
            for idx in pick_idxs:
                if idx < 0:
                    per_ckpt.append({})
                    continue
                ae = (ckpts[idx].get("audio_examples") or {}).get(g, [])
                per_ckpt.append({e.get("key"): e for e in ae})

            cards: list[str] = []
            for ci, by_key in enumerate(per_ckpt):
                candidates: list[str] = []
                for k, e in by_key.items():
                    if e.get("status") != "good":
                        continue
                    others_ok = True
                    for cj in range(len(per_ckpt)):
                        if cj == ci:
                            continue
                        oe = per_ckpt[cj].get(k)
                        if oe is None or oe.get("status") == "good":
                            others_ok = False
                            break
                    if others_ok:
                        candidates.append(k)
                if not candidates:
                    continue
                candidates.sort()
                winner_key = candidates[0]
                base = per_ckpt[ci][winner_key]
                question = base.get("question") or ""
                expected = base.get("expected") or ""
                category = base.get("category") or ""

                audio_rows: list[str] = []
                for cj, meta in enumerate(pick_meta):
                    e = per_ckpt[cj].get(winner_key)
                    if e is None:
                        resp = "(missing in sidecar)"
                        st = "missing"
                        url = ""
                    else:
                        resp = e.get("response") or ""
                        st = e.get("status") or "none"
                        url = _report_audio_url(e.get("audio_src") or e.get("audio_path") or "")
                    is_winner = (cj == ci)
                    if is_winner:
                        badge_bg, badge_fg = f"{GREEN}20", GREEN
                        badge_label = "good"
                    else:
                        badge_bg, badge_fg = f"{RED}20", RED
                        badge_label = escape(st)
                    color = meta["color"]
                    ckpt_badge = (
                        f'<span class="ckpt-badge" '
                        f'style="background:{color}22;color:{color};border:1px solid {color}">'
                        f'{escape(meta["label"])}</span>'
                    )
                    status_badge_html = (
                        f'<span class="badge" '
                        f'style="background:{badge_bg};color:{badge_fg};border:1px solid {badge_fg}">'
                        f'{badge_label}</span>'
                    )
                    audio_tag = (
                        f'<audio class="aplayer" controls preload="none" '
                        f'src="{escape(url, quote=True)}"></audio>'
                        if url else '<span class="dim">(no audio)</span>'
                    )
                    audio_rows.append(
                        f'<div class="audio-row">{ckpt_badge} {status_badge_html}'
                        f'<div class="bubble bubble-a">{escape(resp)}</div>'
                        f'{audio_tag}</div>'
                    )

                header_meta = [f"key: {escape(winner_key)}"]
                if category:
                    header_meta.append(f"category: {escape(category)}")
                meta_html = " &middot; ".join(header_meta)
                cards.append(
                    '<div class="ex-card">'
                    '<div class="ex-header">'
                    f'<strong>{escape(pick_meta[ci]["label"])} wins</strong> '
                    f'<span class="dim">{meta_html}</span>'
                    '</div>'
                    '<div class="ex-body-cmp">'
                    '<div class="ex-left">'
                    '<div class="mini-label">Question</div>'
                    f'<div class="bubble bubble-q">{escape(question)}</div>'
                    '<div class="mini-label">Expected</div>'
                    f'<div class="bubble bubble-a">{escape(expected)}</div>'
                    '</div>'
                    f'<div class="ex-right-cmp">{"".join(audio_rows)}</div>'
                    '</div>'
                    '</div>'
                )
            if not cards:
                continue
            bench_pretty = _REPORT_BENCH_PRETTY.get(bench, bench)
            summary = f"{bench_pretty} — {split}"
            parts.append(
                f'<details class="bench-detail audio-detail" open>'
                f'<summary>{escape(summary)}</summary>'
                f'<div>{"".join(cards)}</div>'
                f'</details>'
            )
    parts.append('<div class="back-to-top"><a href="#top-nav">↑ back to top</a></div>')
    return "\n".join(parts) + "\n"


def _fig_section(metric_arrays: dict[str, list[Any]],
                 labels: list[str],
                 picks: list[dict[str, Any]],
                 fdbv3_cov_text: str) -> tuple[str, str, str, str, str, str, str]:
    """Build Figures 1-7 HTML chunks + cross-cutting insight strings.

    Returns (fig1_html, fig2_html, fig3_html, fig4_html, fig5_html, fig6_html, fig7_html).
    """
    BACK = _REPORT_BACK_TO_TOP

    def _topn(arr_key: str, *, reverse: bool = False, mask: Any = None,
              fmt: str = "{:.1f}") -> str:
        arr = metric_arrays[arr_key]
        entries = [(labels[i], v) for i, v in enumerate(arr)
                   if v is not None and (mask is None or mask(v))]
        entries.sort(key=lambda kv: kv[1], reverse=not reverse)
        return ", ".join(f"<code>{escape(lbl)}</code> ({fmt.format(v)})"
                         for lbl, v in entries[:3])

    vb_top_html = _topn("VB_AVG")
    lat_top_html = _topn("FDBV1_LAT_AVG", reverse=True, fmt="{:.0f}ms")
    tor_top_html = _topn("FDBV1_TOR_SUMM")
    fdbv15_top_html = _topn("FDBV15_HL", mask=lambda v: v <= 1.5, fmt="{:.2f}")
    bfcl_top_html = _topn("BFCL_PRIO", fmt="{:.1f}%")
    bba_top_html = _topn("BBA_HL", fmt="{:.1f}%")
    conv_top_html = _topn("CONV_F1", fmt="{:.1f}%")

    crit_list = sorted(
        [(labels[i], v) for i, v in enumerate(metric_arrays["BFCL_IRR"])
         if v is not None and v < 70],
        key=lambda kv: kv[1])[:3]
    crit_html = ", ".join(f"<code>{escape(lbl)}</code> ({v:.1f}%)" for lbl, v in crit_list)
    if not crit_html:
        crit_html = "(none below 70% in this set)"

    fig1 = "\n".join([
        BACK,
        '<h2 id="fig1">Figure 1 — VoiceBench: Normalized Average Accuracy <span class="fignum">All sub-scores normalized 0–100 before averaging</span></h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>VoiceBench Normalized Average (all sub-tasks)</h3>'
        '<div class="cw"><canvas id="f1_avg"></canvas></div>'
        '<p class="fig-label">Tracker formula: BBH + OpenBookQA + MMSU + SDQA_avg + IFEval + AdvBench + 20×(CommonEval + AlpacaEval_Full + WildVoice), divided by number of non-null inputs.</p></div>',
        '  <div class="card"><h3>VoiceBench Sub-scores — All 10 Datasets, All Experiments</h3>'
        '<div class="cw-xl"><canvas id="f1_sub"></canvas></div>'
        '<p class="fig-label">All 10 sub-tasks (normalized 0–100). IFEval is a persistent weakness; AdvBench and AlpacaEval-Spk are high across the board.</p></div>',
        '</div>',
        '<div class="grid2" style="margin-top:12px">',
        '  <div class="card"><h3>FDB v1 — Combined Response Latency avg(TT, Intr) ms ↓ better</h3>'
        '<div class="cw"><canvas id="f1_lat_avg"></canvas></div></div>',
        '  <div class="card"><h3>FDB v1 — Adherence Score (GPT-4o rating, /5, ↑ better)</h3>'
        '<div class="cw"><canvas id="f1_adh_sum"></canvas></div></div>',
        '</div>',
        f'<div class="insight"><strong>VoiceBench top-3 (tracker formula):</strong> {vb_top_html}. '
        f'<strong>Fastest combined latency (avg TT+Intr):</strong> {lat_top_html}.</div>',
    ])

    fig2 = "\n".join([
        BACK,
        '<h2 id="fig2">Figure 2 — Full Duplex Bench v1: Conversational Dynamics</h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>FDB v1 — TOR Summary % (tracker formula, ↑ better)</h3>'
        '<div class="cw"><canvas id="f2_hl"></canvas></div></div>',
        '  <div class="card"><h3>FDB v1 — Adherence Score (GPT-4o /5, ↑ better)</h3>'
        '<div class="cw"><canvas id="f2_adh"></canvas></div></div>',
        '</div>',
        '<div class="grid2" style="margin-top:16px">',
        '  <div class="card"><h3>FDB v1 — Pause TOR: Candor + Synthetic (↓ better)</h3>'
        '<div class="cw"><canvas id="f2_pause"></canvas></div></div>',
        '  <div class="card"><h3>FDB v1 — Interruption: TOR% (↑) and Latency ms (↓)</h3>'
        '<div class="cw"><canvas id="f2_intr"></canvas></div></div>',
        '</div>',
        '<div class="grid2" style="margin-top:16px">',
        '  <div class="card"><h3>FDB v1 — Turn-Taking: TOR% (↑) and Latency ms (↓)</h3>'
        '<div class="cw"><canvas id="f2_tt"></canvas></div></div>',
        '</div>',
        f'<div class="note"><strong>FDB v1 TOR Summary top-3 (tracker formula):</strong> {tor_top_html}.</div>',
    ])

    fig3 = "\n".join([
        BACK,
        '<h2 id="fig3">Figure 3 — Full Duplex Bench v1.5: Extended Behavioral Scenarios</h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>FDB v1.5 — Headline Score (↑ better)</h3>'
        '<div class="cw"><canvas id="f3_hl"></canvas></div>'
        '<p class="fig-label">⚠ Any headline values above 1.0 are anomalous (likely missing sub-metrics); treat with caution.</p></div>',
        '  <div class="card"><h3>FDB v1.5 — Background Speech: C_RESPOND vs C_RESUME</h3>'
        '<div class="cw"><canvas id="f3_bg"></canvas></div></div>',
        '</div>',
        '<div class="grid2" style="margin-top:16px">',
        '  <div class="card"><h3>FDB v1.5 — Backchannel Resume Rate (↑ better)</h3>'
        '<div class="cw"><canvas id="f3_bc"></canvas></div></div>',
        '  <div class="card"><h3>FDB v1.5 — Interruption Respond Rate (↑ better)</h3>'
        '<div class="cw"><canvas id="f3_intr"></canvas></div></div>',
        '</div>',
        f'<div class="insight"><strong>FDB v1.5 headline top-3 (excluding anomalies):</strong> {fdbv15_top_html}.</div>',
    ])

    fig4 = "\n".join([
        BACK,
        '<h2 id="fig4">Figure 4 — Full Duplex Bench v3</h2>',
        '<div class="card" style="max-width:600px"><h3>FDB v3 — Headline Score</h3>'
        '<div class="cw"><canvas id="f4_hl"></canvas></div></div>',
        f'<div class="warn"><strong>FDB v3 data is available for {fdbv3_cov_text} checkpoints.</strong> Missing checkpoints should be evaluated before drawing strong conclusions.</div>',
    ])

    fig5 = "\n".join([
        BACK,
        '<h2 id="fig5">Figure 5 — BFCL: Function Calling Performance</h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>BFCL — Priority Composite (mean of Simple, Multiple, Irrelevance)</h3>'
        '<div class="cw"><canvas id="f5_prio"></canvas></div></div>',
        '  <div class="card"><h3>BFCL — Simple and Multiple (↑ better)</h3>'
        '<div class="cw"><canvas id="f5_sm"></canvas></div></div>',
        '</div>',
        '<div class="grid2" style="margin-top:16px">',
        '  <div class="card"><h3>BFCL — Irrelevance (↑ better)</h3>'
        '<div class="cw"><canvas id="f5_irr"></canvas></div></div>',
        '  <div class="card"><h3>BFCL — Parallel and Parallel-Multiple (↑ better)</h3>'
        '<div class="cw"><canvas id="f5_par"></canvas></div></div>',
        '</div>',
        f'<div class="insight"><strong>BFCL Priority top-3:</strong> {bfcl_top_html}.</div>',
        f'<div class="warn"><strong>Critical BFCL Irrelevance issues (&lt;70%):</strong> {crit_html}. These checkpoints attempt tool calls on irrelevant queries — production blocker.</div>',
    ])

    fig6 = "\n".join([
        BACK,
        '<h2 id="fig6">Figure 6 — Big Bench Audio (BBA): Reasoning</h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>BBA — Headline and Sub-task Accuracies</h3>'
        '<div class="cw-lg"><canvas id="f6_all"></canvas></div></div>',
        '  <div class="card"><h3>BBA — Web of Lies and Object Counting (most variable)</h3>'
        '<div class="cw-lg"><canvas id="f6_var"></canvas></div></div>',
        '</div>',
        f'<div class="insight"><strong>BBA headline top-3:</strong> {bba_top_html}.</div>',
    ])

    fig7 = "\n".join([
        BACK,
        '<h2 id="fig7">Figure 7 — Turn Taking (Conv Behavior)</h2>',
        '<div class="grid2">',
        '  <div class="card"><h3>Conv Behavior — Turn-Taking F1 (↑ better)</h3>'
        '<div class="cw"><canvas id="f7_f1"></canvas></div></div>',
        '  <div class="card"><h3>Conv Behavior — Precision, Recall, Cutoff Rate</h3>'
        '<div class="cw"><canvas id="f7_sub"></canvas></div></div>',
        '</div>',
        f'<div class="insight"><strong>Conv Behavior F1 top-3:</strong> {conv_top_html}.</div>',
    ])

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


def _recommendations_section(picks: list[dict[str, Any]],
                             metric_arrays: dict[str, list[Any]],
                             labels: list[str],
                             fdbv3_cov_text: str) -> str:
    """Render the Recommendations card list."""
    crit_list = sorted(
        [(labels[i], v) for i, v in enumerate(metric_arrays["BFCL_IRR"])
         if v is not None and v < 70],
        key=lambda kv: kv[1])[:3]
    crit_html = ", ".join(f"<code>{escape(lbl)}</code> ({v:.1f}%)" for lbl, v in crit_list)
    if not crit_html:
        crit_html = "(none below 70% in this set)"
    best_short = picks[0].get("short_label", "?") if picks else "?"
    short_list = ", ".join(escape(p.get("short_label", "?")) for p in picks)
    return "\n".join([
        _REPORT_BACK_TO_TOP,
        '<h2 id="recs">Recommendations for Future Experiments</h2>',
        f'<div class="rec"><strong>R1 — Address BFCL Irrelevance regressions (urgent if &lt;70%).</strong> Critical: {crit_html}. Investigate data blends for these and add targeted irrelevance fine-tuning where needed.</div>',
        f'<div class="rec"><strong>R2 — Complete FDB v3 coverage (data gap).</strong> FDB v3 is currently available for {fdbv3_cov_text} checkpoints. Prioritize evaluating the top holistic ckpts ({short_list}) so FDB v3 can contribute to selection.</div>',
        f'<div class="rec"><strong>R3 — Continue training the top-holistic checkpoint (<code>{escape(best_short)}</code>).</strong> If a positive trend is visible across earlier steps of the same experiment, extend training and re-evaluate at the next step bucket.</div>',
        '<div class="rec"><strong>R4 — Address BFCL Parallel (systemic gap).</strong> All checkpoints score very low on Parallel and Parallel-Multiple. Targeted training data with parallel function-call examples should improve these categories.</div>',
        '<div class="rec"><strong>R5 — Disentangle FDB v1 vs FDB v1.5 tension.</strong> Strong FDB v1 ckpts often lag FDB v1.5; strong FDB v1.5 ckpts are mid-tier on FDB v1. Design an experiment that explicitly targets both.</div>',
        '<div class="rec"><strong>R6 — Ablate LR and init checkpoint independently.</strong> LR and init checkpoint are always confounded in the current experiment lattice. Add one experiment that crosses these to identify the actual driver.</div>',
        '<div class="rec"><strong>R7 — Investigate the BBA / Conv F1 leaders.</strong> Understanding what differs in their training (data blend, training duration, architecture) could guide future experiments toward higher reasoning capability.</div>',
    ])


def _detailed_sections() -> str:
    """Render the D1-D12 Detailed sections with TOC + D12 numeric summary."""
    return "\n".join([
        _REPORT_BACK_TO_TOP,
        '<h2 id="detailed">Detailed Metrics — All Benchmarks</h2>',
        '<div class="toc"><h3>Table of Contents</h3>'
        '<ul style="columns:2;column-gap:30px;list-style:none">'
        '<li><a href="#d1">D1 — VoiceBench: All Sub-scores</a></li>'
        '<li><a href="#d2">D2 — FDB v1: Backchannel Detail</a></li>'
        '<li><a href="#d3">D3 — FDB v1: Pause Handling Detail</a></li>'
        '<li><a href="#d4">D4 — FDB v1: Interruption Detail</a></li>'
        '<li><a href="#d5">D5 — FDB v1: Turn-Taking Detail</a></li>'
        '<li><a href="#d6">D6 — FDB v1.5: Background Speech Detail</a></li>'
        '<li><a href="#d7">D7 — FDB v1.5: Talking-to-Other Detail</a></li>'
        '<li><a href="#d8">D8 — FDB v1.5: Backchannel + Interruption Detail</a></li>'
        '<li><a href="#d9">D9 — BFCL: All Categories</a></li>'
        '<li><a href="#d10">D10 — BBA: All Sub-tasks</a></li>'
        '<li><a href="#d11">D11 — Conv Behavior: Precision / Recall / Cutoff</a></li>'
        '<li><a href="#d12">D12 — Full Numeric Summary Table</a></li>'
        '</ul></div>',
        '<section id="d1"><h3>D1 — VoiceBench: All Sub-scores</h3>'
        '<div class="grid2"><div class="card"><h4>AlpacaEval Full GPT (/5 → ×20)</h4><div class="cw"><canvas id="d1_al"></canvas></div></div>'
        '<div class="card"><h4>IFEval Final Accuracy</h4><div class="cw"><canvas id="d1_if"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>CommonEval GPT (/5 → ×20)</h4><div class="cw"><canvas id="d1_ce"></canvas></div></div>'
        '<div class="card"><h4>WildVoice GPT (/5 → ×20)</h4><div class="cw"><canvas id="d1_wv"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>SDQA avg(PANDA, GPT)</h4><div class="cw"><canvas id="d1_sdqa"></canvas></div></div>'
        '<div class="card"><h4>AlpacaEval-Speaker GPT (/5 → ×20)</h4><div class="cw"><canvas id="d1_spk"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>AdvBench Refusal Rate</h4><div class="cw"><canvas id="d1_adv"></canvas></div></div>'
        '<div class="card"><h4>BBH Accuracy</h4><div class="cw"><canvas id="d1_bbh"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>OpenBookQA Accuracy</h4><div class="cw"><canvas id="d1_opbk"></canvas></div></div>'
        '<div class="card"><h4>MMSU Accuracy</h4><div class="cw"><canvas id="d1_mmsu"></canvas></div></div></div>'
        '</section>',
        '<section id="d2"><h3>D2 — FDB v1: Backchannel Detail</h3>'
        '<div class="grid2"><div class="card"><h4>Backchannel TOR % (↓ better)</h4><div class="cw"><canvas id="d2_bc_tor"></canvas></div></div>'
        '<div class="card"><h4>Backchannel JSD (↑ better)</h4><div class="cw"><canvas id="d2_bc_jsd"></canvas></div></div></div></section>',
        '<section id="d3"><h3>D3 — FDB v1: Pause Handling (Candor + Synthetic)</h3>'
        '<div class="grid2"><div class="card"><h4>Candor Pause TOR % (↓ better)</h4><div class="cw"><canvas id="d3_cand"></canvas></div></div>'
        '<div class="card"><h4>Synthetic Pause TOR % (↓ better)</h4><div class="cw"><canvas id="d3_syn"></canvas></div></div></div></section>',
        '<section id="d4"><h3>D4 — FDB v1: Interruption</h3>'
        '<div class="grid2"><div class="card"><h4>Interruption TOR %</h4><div class="cw"><canvas id="d4_intr_tor"></canvas></div></div>'
        '<div class="card"><h4>Latency ms + GPT Rating</h4><div class="cw"><canvas id="d4_intr_lat"></canvas></div></div></div></section>',
        '<section id="d5"><h3>D5 — FDB v1: Turn-Taking</h3>'
        '<div class="grid2"><div class="card"><h4>Turn-Taking TOR %</h4><div class="cw"><canvas id="d5_tt_tor"></canvas></div></div>'
        '<div class="card"><h4>Turn-Taking Latency ms (↓ better)</h4><div class="cw"><canvas id="d5_tt_lat"></canvas></div></div></div></section>',
        '<section id="d6"><h3>D6 — FDB v1.5: Background Speech</h3>'
        '<div class="grid2"><div class="card"><h4>C_RESPOND (lower better)</h4><div class="cw"><canvas id="d6_bg_resp"></canvas></div></div>'
        '<div class="card"><h4>C_RESUME (higher better)</h4><div class="cw"><canvas id="d6_bg_res"></canvas></div></div></div>'
        '<div class="card" style="margin-top:12px;max-width:700px"><h4>Background Speech Stop Latency ms (↓ better)</h4><div class="cw"><canvas id="d6_bg_sl"></canvas></div></div></section>',
        '<section id="d7"><h3>D7 — FDB v1.5: Talking-to-Other</h3>'
        '<div class="grid2"><div class="card"><h4>C_RESPOND (higher better)</h4><div class="cw"><canvas id="d7_tto_resp"></canvas></div></div>'
        '<div class="card"><h4>C_RESUME (lower better)</h4><div class="cw"><canvas id="d7_tto_res"></canvas></div></div></div></section>',
        '<section id="d8"><h3>D8 — FDB v1.5: Backchannel + Interruption Behavioral Rates</h3>'
        '<div class="grid2"><div class="card"><h4>Backchannel C_RESUME</h4><div class="cw"><canvas id="d8_bc_res"></canvas></div></div>'
        '<div class="card"><h4>Interruption C_RESPOND</h4><div class="cw"><canvas id="d8_intr_resp"></canvas></div></div></div></section>',
        '<section id="d9"><h3>D9 — BFCL: All Categories</h3>'
        '<div class="grid2"><div class="card"><h4>BFCL Headline Average</h4><div class="cw"><canvas id="d9_avg"></canvas></div></div>'
        '<div class="card"><h4>BFCL Simple</h4><div class="cw"><canvas id="d9_simp"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>BFCL Multiple</h4><div class="cw"><canvas id="d9_mult"></canvas></div></div>'
        '<div class="card"><h4>BFCL Irrelevance</h4><div class="cw"><canvas id="d9_irr"></canvas></div></div></div></section>',
        '<section id="d10"><h3>D10 — Big Bench Audio: All Sub-tasks</h3>'
        '<div class="grid2"><div class="card"><h4>Formal Fallacies</h4><div class="cw"><canvas id="d10_ff"></canvas></div></div>'
        '<div class="card"><h4>Navigate</h4><div class="cw"><canvas id="d10_nav"></canvas></div></div></div>'
        '<div class="grid2" style="margin-top:12px"><div class="card"><h4>Object Counting</h4><div class="cw"><canvas id="d10_oc"></canvas></div></div>'
        '<div class="card"><h4>Web of Lies</h4><div class="cw"><canvas id="d10_wl"></canvas></div></div></div></section>',
        '<section id="d11"><h3>D11 — Conv Behavior: Precision, Recall, Cutoff Rate</h3>'
        '<div class="grid2"><div class="card"><h4>TT Precision</h4><div class="cw"><canvas id="d11_prec"></canvas></div></div>'
        '<div class="card"><h4>TT Recall</h4><div class="cw"><canvas id="d11_rec"></canvas></div></div></div>'
        '<div class="card" style="margin-top:12px;max-width:700px"><h4>Cutoff Rate (↓ better)</h4><div class="cw"><canvas id="d11_cut"></canvas></div></div></section>',
        '<section id="d12"><h3>D12 — Full Numeric Summary Table</h3>'
        '<div class="ov"><table><thead><tr>'
        '<th>Checkpoint</th><th>VB Avg</th><th>BFCL Prio</th><th>BFCL Simp</th><th>BFCL Mult</th>'
        '<th>BFCL Irr</th><th>FDB v1 TOR</th><th>FDB v1.5</th><th>FDB v3</th><th>BBA</th><th>Conv F1</th>'
        '<th>Cand TOR↓</th><th>Syn TOR↓</th><th>TT TOR↑</th>'
        '</tr></thead><tbody id="summary-tbody"></tbody></table></div></section>',
    ])


def _report_chart_js(metric_arrays: dict[str, list[Any]],
                     labels: list[str],
                     pick_idxs: list[int]) -> str:
    """JS block: chart helpers + all chart calls + D12 summary table builder."""
    arr_js_lines = [f"const LABELS={_report_js_array(labels)};"]
    keys_in_order = [
        "VB_AVG", "VB_ALPACA", "VB_ALPACA_FULL", "VB_IFEVAL", "VB_COMMON",
        "VB_WILD", "VB_SDQA", "VB_SPK", "VB_ADV", "VB_BBH", "VB_OPBK", "VB_MMSU",
        "BFCL_SIMP", "BFCL_MULT", "BFCL_PAR", "BFCL_PM", "BFCL_IRR", "BFCL_AVG",
        "BFCL_PRIO",
        "FDBV1_HL", "FDBV1_BC", "FDBV1_CAND", "FDBV1_SYN",
        "FDBV1_INTR_TOR", "FDBV1_INTR_LAT", "FDBV1_INTR_RAT",
        "FDBV1_TT_TOR", "FDBV1_TT_LAT", "FDBV1_JSD",
        "FDBV1_TOR_SUMM", "FDBV1_ADH", "FDBV1_LAT_AVG",
        "FDBV15_HL", "FDBV15_BG_RESP", "FDBV15_BG_RES", "FDBV15_BG_SL",
        "FDBV15_TTO_RESP", "FDBV15_TTO_RES", "FDBV15_BC_RES", "FDBV15_INTR_RESP",
        "FDBV3_HL",
        "BBA_HL", "BBA_FF", "BBA_NAV", "BBA_OC", "BBA_WL",
        "CONV_HL", "CONV_PREC", "CONV_REC", "CONV_F1", "CONV_CUT",
    ]
    for k in keys_in_order:
        arr_js_lines.append(f"const {k}={_report_js_array(metric_arrays[k])};")
    data_js = "\n".join(arr_js_lines)
    top_idx_js = "[" + ",".join(str(i) for i in pick_idxs if i >= 0) + "]"

    return f"""<script>
// -- DATA ------------------------------------------------------------------
{data_js}

const TOP_IDXS = {top_idx_js};

// -- CHART HELPERS ---------------------------------------------------------
Chart.defaults.color='#8b949e';
Chart.defaults.borderColor='#30363d';
Chart.defaults.font.family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
Chart.defaults.font.size=10;

function barColors(data, mainC='rgba(88,166,255,0.7)') {{
  const cols=['rgba(255,215,0,0.85)','rgba(192,192,192,0.85)','rgba(205,127,50,0.85)','rgba(88,166,255,0.85)','rgba(188,140,255,0.85)','rgba(57,197,207,0.85)'];
  return data.map((v,i) => {{
    if (v === null || v === undefined) return 'rgba(80,80,80,0.3)';
    const tidx = TOP_IDXS.indexOf(i);
    if (tidx >= 0) return cols[tidx] || mainC;
    return mainC;
  }});
}}

function makeBar(id, label, data, opts={{}}) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const {{yMin, yMax, yLabel, reverse=false}} = opts;
  new Chart(ctx, {{
    type:'bar',
    data:{{labels:LABELS, datasets:[{{label:label, data:data,
      backgroundColor:barColors(data), borderWidth:0, borderRadius:3}}]}},
    options:{{responsive:true, maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.raw===null?'N/A':c.raw}}`}}}}}},
      scales:{{
        x:{{ticks:{{maxRotation:45,font:{{size:9}}}},grid:{{color:'#21262d'}}}},
        y:{{min:yMin,max:yMax,reverse:reverse,
          title:{{display:!!yLabel,text:yLabel||'',font:{{size:9}}}},
          grid:{{color:'#21262d'}}}}
      }}
    }}
  }});
}}

function makeGrouped(id, datasets, opts={{}}) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  new Chart(ctx, {{
    type:'bar', data:{{labels:LABELS, datasets}},
    options:{{responsive:true, maintainAspectRatio:false,
      plugins:{{legend:{{position:'top',labels:{{font:{{size:9}},boxWidth:10}}}}}},
      scales:{{
        x:{{ticks:{{maxRotation:45,font:{{size:9}}}},grid:{{color:'#21262d'}}}},
        y:{{min:opts.yMin,max:opts.yMax,reverse:opts.reverse||false,
            grid:{{color:'#21262d'}},
            title:{{display:!!opts.yLabel,text:opts.yLabel||''}}}}
      }}
    }}
  }});
}}

const DSC = (label, data, color) => ({{label, data,
  backgroundColor: color+'b3', borderColor: color, borderWidth:1, borderRadius:2}});

// -- FIGURES ---------------------------------------------------------------
makeBar('f1_avg','VB Normalized Avg',VB_AVG,{{yLabel:'Score (0-100)'}});
makeBar('f1_lat_avg','Combined Latency ms (TT+Intr)/2',FDBV1_LAT_AVG,{{yLabel:'ms'}});
makeBar('f1_adh_sum','Adherence Score (GPT-4o /5)',FDBV1_ADH,{{yLabel:'/5'}});
makeGrouped('f1_sub',[
  DSC('AlpacaEval Full',VB_ALPACA_FULL,'#58a6ff'),
  DSC('IFEval',VB_IFEVAL,'#f85149'),
  DSC('CommonEval',VB_COMMON,'#3fb950'),
  DSC('WildVoice',VB_WILD,'#e3b341'),
  DSC('SDQA-avg',VB_SDQA,'#ff7b72'),
  DSC('AlpacaEval-Spk',VB_SPK,'#ffa657'),
  DSC('AdvBench',VB_ADV,'#7ee787'),
  DSC('BBH',VB_BBH,'#bc8cff'),
  DSC('OpenBookQA',VB_OPBK,'#39c5cf'),
  DSC('MMSU',VB_MMSU,'#d2a8ff'),
],{{yMin:0,yMax:105,yLabel:'Score (0-100)'}});

makeBar('f2_hl','FDB v1 TOR Summary % (tracker)',FDBV1_TOR_SUMM);
makeBar('f2_adh','Adherence Score (GPT-4o /5)',FDBV1_ADH,{{yLabel:'/5'}});
makeGrouped('f2_pause',[
  DSC('Candor TOR%',FDBV1_CAND,'#f85149'),
  DSC('Synthetic TOR%',FDBV1_SYN,'#d29922'),
],{{yLabel:'TOR %'}});
makeGrouped('f2_intr',[
  DSC('Interruption TOR%',FDBV1_INTR_TOR,'#3fb950'),
  {{label:'Latency ms/10',data:FDBV1_INTR_LAT.map(v=>v===null?null:v/10),
    backgroundColor:'rgba(88,166,255,0.6)',borderColor:'#58a6ff',borderWidth:1,borderRadius:2}},
],{{yLabel:'TOR% / Latency÷10'}});
makeGrouped('f2_tt',[
  DSC('TT TOR%',FDBV1_TT_TOR,'#3fb950'),
  {{label:'TT Latency ms/10',data:FDBV1_TT_LAT.map(v=>v===null?null:v/10),
    backgroundColor:'rgba(248,81,73,0.5)',borderColor:'#f85149',borderWidth:1,borderRadius:2}},
],{{yLabel:'TOR% / Latency÷10'}});

makeBar('f3_hl','FDB v1.5 Headline',FDBV15_HL);
makeGrouped('f3_bg',[
  DSC('C_RESPOND',FDBV15_BG_RESP,'#f85149'),
  DSC('C_RESUME',FDBV15_BG_RES,'#3fb950'),
]);
makeBar('f3_bc','Backchannel C_RESUME',FDBV15_BC_RES);
makeBar('f3_intr','Interruption C_RESPOND',FDBV15_INTR_RESP);

makeBar('f4_hl','FDB v3 Headline',FDBV3_HL);

makeBar('f5_prio','BFCL Priority',BFCL_PRIO);
makeGrouped('f5_sm',[
  DSC('Simple',BFCL_SIMP,'#3fb950'),
  DSC('Multiple',BFCL_MULT,'#58a6ff'),
]);
makeBar('f5_irr','BFCL Irrelevance',BFCL_IRR);
makeGrouped('f5_par',[
  DSC('Parallel',BFCL_PAR,'#bc8cff'),
  DSC('Par-Multiple',BFCL_PM,'#39c5cf'),
]);

makeGrouped('f6_all',[
  DSC('Headline',BBA_HL,'#e3b341'),
  DSC('Formal Fallacies',BBA_FF,'#f85149'),
  DSC('Navigate',BBA_NAV,'#3fb950'),
  DSC('Object Counting',BBA_OC,'#58a6ff'),
  DSC('Web of Lies',BBA_WL,'#bc8cff'),
]);
makeGrouped('f6_var',[
  DSC('Web of Lies',BBA_WL,'#bc8cff'),
  DSC('Object Counting',BBA_OC,'#39c5cf'),
]);

makeBar('f7_f1','Conv Behavior F1',CONV_F1);
makeGrouped('f7_sub',[
  DSC('Precision',CONV_PREC,'#3fb950'),
  DSC('Recall',CONV_REC,'#58a6ff'),
  {{label:'Cutoff×10',data:CONV_CUT.map(v=>v===null?null:v*10),
    backgroundColor:'rgba(248,81,73,0.7)',borderColor:'#f85149',borderWidth:1,borderRadius:2}},
]);

// -- DETAILS ---------------------------------------------------------------
makeBar('d1_al','AlpacaEval Full (norm)',VB_ALPACA_FULL);
makeBar('d1_if','IFEval',VB_IFEVAL);
makeBar('d1_ce','CommonEval (norm)',VB_COMMON);
makeBar('d1_wv','WildVoice (norm)',VB_WILD);
makeBar('d1_sdqa','SDQA avg(PANDA,GPT)',VB_SDQA);
makeBar('d1_spk','AlpacaEval-Spk (norm)',VB_SPK);
makeBar('d1_adv','AdvBench',VB_ADV);
makeBar('d1_bbh','BBH',VB_BBH);
makeBar('d1_opbk','OpenBookQA',VB_OPBK);
makeBar('d1_mmsu','MMSU',VB_MMSU);
makeBar('d2_bc_tor','Backchannel TOR %',FDBV1_BC);
makeBar('d2_bc_jsd','Backchannel JSD',FDBV1_JSD);
makeBar('d3_cand','Candor Pause TOR %',FDBV1_CAND);
makeBar('d3_syn','Synthetic Pause TOR %',FDBV1_SYN);
makeBar('d4_intr_tor','Interruption TOR %',FDBV1_INTR_TOR);
makeGrouped('d4_intr_lat',[
  DSC('Latency ms',FDBV1_INTR_LAT,'#f85149'),
  DSC('GPT Rating ×100',FDBV1_INTR_RAT.map(v=>v===null?null:v*100),'#3fb950'),
]);
makeBar('d5_tt_tor','Turn-Taking TOR %',FDBV1_TT_TOR);
makeBar('d5_tt_lat','Turn-Taking Latency ms',FDBV1_TT_LAT);
makeBar('d6_bg_resp','C_RESPOND bg',FDBV15_BG_RESP);
makeBar('d6_bg_res','C_RESUME bg',FDBV15_BG_RES);
makeBar('d6_bg_sl','BG stop latency ms',FDBV15_BG_SL);
makeBar('d7_tto_resp','TTO C_RESPOND',FDBV15_TTO_RESP);
makeBar('d7_tto_res','TTO C_RESUME',FDBV15_TTO_RES);
makeBar('d8_bc_res','Backchannel C_RESUME',FDBV15_BC_RES);
makeBar('d8_intr_resp','Interruption C_RESPOND',FDBV15_INTR_RESP);
makeBar('d9_avg','BFCL Headline Avg',BFCL_AVG);
makeBar('d9_simp','BFCL Simple %',BFCL_SIMP);
makeBar('d9_mult','BFCL Multiple %',BFCL_MULT);
makeBar('d9_irr','BFCL Irrelevance %',BFCL_IRR);
makeBar('d10_ff','BBA Formal Fallacies',BBA_FF);
makeBar('d10_nav','BBA Navigate',BBA_NAV);
makeBar('d10_oc','BBA Object Counting',BBA_OC);
makeBar('d10_wl','BBA Web of Lies',BBA_WL);
makeBar('d11_prec','TT Precision %',CONV_PREC);
makeBar('d11_rec','TT Recall %',CONV_REC);
makeBar('d11_cut','Cutoff Rate % (↓ better)',CONV_CUT);

// -- SUMMARY TABLE ---------------------------------------------------------
(function(){{
  const tbody = document.getElementById('summary-tbody');
  if (!tbody) return;
  const top = TOP_IDXS;
  const medals = ['\U0001F947','\U0001F948','\U0001F949','#4','#5','#6'];
  function cls(v, arr, rev=false){{
    if (v === null || v === undefined) return 'na';
    const sorted = [...arr].filter(x => x !== null && x !== undefined).sort((a,b) => rev?a-b:b-a);
    const rank = sorted.findIndex(x => x === v);
    if (rank < 3) return 'best';
    if (rank < 6) return 'good';
    if (rank < 12) return 'mid';
    return 'low';
  }}
  function fmt(v, dec=1) {{
    return v === null || v === undefined ? '<span class="na">N/A</span>' : `${{(+v).toFixed(dec)}}`;
  }}
  LABELS.forEach((lbl, i) => {{
    const tidx = top.indexOf(i);
    const tr = document.createElement('tr');
    const m = tidx >= 0 ? ` <span class="badge ${{['b1','b2','b3'][tidx] || 'badge'}}">${{medals[tidx] || ('#'+(tidx+1))}}</span>` : '';
    tr.innerHTML = `
      <td style="font-family:monospace;font-size:.8em">${{lbl}}${{m}}</td>
      <td class="num ${{cls(VB_AVG[i],VB_AVG)}}">${{fmt(VB_AVG[i],2)}}</td>
      <td class="num ${{cls(BFCL_PRIO[i],BFCL_PRIO)}}">${{fmt(BFCL_PRIO[i],1)}}</td>
      <td class="num ${{cls(BFCL_SIMP[i],BFCL_SIMP)}}">${{fmt(BFCL_SIMP[i],1)}}</td>
      <td class="num ${{cls(BFCL_MULT[i],BFCL_MULT)}}">${{fmt(BFCL_MULT[i],1)}}</td>
      <td class="num ${{cls(BFCL_IRR[i],BFCL_IRR)}}">${{fmt(BFCL_IRR[i],1)}}</td>
      <td class="num ${{cls(FDBV1_TOR_SUMM[i],FDBV1_TOR_SUMM)}}">${{fmt(FDBV1_TOR_SUMM[i],1)}}</td>
      <td class="num ${{cls(FDBV15_HL[i],FDBV15_HL.filter(x=>x!==null&&x<1))}}">${{fmt(FDBV15_HL[i],2)}}</td>
      <td class="num ${{cls(FDBV3_HL[i],FDBV3_HL.filter(x=>x!==null))}}">${{fmt(FDBV3_HL[i],1)}}</td>
      <td class="num ${{cls(BBA_HL[i],BBA_HL)}}">${{fmt(BBA_HL[i],1)}}</td>
      <td class="num ${{cls(CONV_F1[i],CONV_F1)}}">${{fmt(CONV_F1[i],1)}}</td>
      <td class="num ${{cls(FDBV1_CAND[i],FDBV1_CAND,true)}}">${{fmt(FDBV1_CAND[i],1)}}</td>
      <td class="num ${{cls(FDBV1_SYN[i],FDBV1_SYN,true)}}">${{fmt(FDBV1_SYN[i],1)}}</td>
      <td class="num ${{cls(FDBV1_TT_TOR[i],FDBV1_TT_TOR)}}">${{fmt(FDBV1_TT_TOR[i],1)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}})();
</script>
"""


FALLBACK_JUDGMENT_SCOPE = ("fdb_v1", "fdb_v1_5")


def apply_audio_judgments(ckpts: list[dict[str, Any]], picks_doc: dict[str, Any] | None) -> None:
    """Overlay LLM-as-judge labels onto the in-memory examples.

    Always stamps `judgment_source = "fallback"` on in-scope examples that
    don't already have one (back-fills older sidecars written before the
    load-time stamp). When `picks_doc.audio_judgments.by_ckpt` has a verdict
    for a given ckpt+example, the status is overridden and the source is
    upgraded to "llm". Examples outside scope are left untouched.
    """
    judgments: dict[str, Any] = {}
    if isinstance(picks_doc, dict):
        judgments = picks_doc.get("audio_judgments") or {}
    scope = set(judgments.get("scope") or FALLBACK_JUDGMENT_SCOPE)
    by_ckpt = judgments.get("by_ckpt") or {}
    for ckpt in ckpts:
        name = ckpt.get("name") or ""
        ck_judgments = by_ckpt.get(name) or {}
        for group, examples in (ckpt.get("audio_examples") or {}).items():
            bench, _, _ = group.partition(":")
            if bench not in scope:
                continue
            for ex in examples or []:
                key = ex.get("key")
                if not key:
                    ex.setdefault("judgment_source", "fallback")
                    continue
                joined = f"{bench}:{key}"
                verdict = ck_judgments.get(joined)
                if verdict and verdict.get("status") in {"good", "bad", "ok"}:
                    ex["status"] = verdict["status"]
                    ex["status_label"] = verdict["status"]
                    if verdict.get("reason"):
                        ex["judgment_reason"] = verdict["reason"]
                    ex["judgment_source"] = "llm"
                else:
                    ex.setdefault("judgment_source", "fallback")


def _render_report_html(ckpts: list[dict[str, Any]],
                        picks_doc: dict[str, Any],
                        out_html: Path,
                        top_N: int) -> str:
    """Assemble the full report HTML (mirrors asset/report_may20.html)."""
    apply_audio_judgments(ckpts, picks_doc)
    picks: list[dict[str, Any]] = picks_doc.get("picks", []) if isinstance(picks_doc, dict) else []
    labels = _report_short_labels_with_disambig([c.get("name", "?") for c in ckpts])
    metric_arrays = _report_derive_metrics(ckpts)

    name_to_idx = {c.get("name", ""): i for i, c in enumerate(ckpts)}
    pick_idxs: list[int] = [name_to_idx.get(p.get("full_name", ""), -1) for p in picks]

    top_cards = _top_n_cards(picks, metric_arrays, pick_idxs)

    n_have_fdbv3 = sum(1 for v in metric_arrays["FDBV3_HL"] if v is not None)
    n_total = len(ckpts)
    fdbv3_cov_text = f"{n_have_fdbv3} of {n_total}"

    fig1, fig2, fig3, fig4, fig5, fig6, fig7 = _fig_section(
        metric_arrays, labels, picks, fdbv3_cov_text)
    recs_html = _recommendations_section(picks, metric_arrays, labels, fdbv3_cov_text)
    detailed_html = _detailed_sections()
    audio_html = _curated_audio_section(ckpts, picks)
    js_block = _report_chart_js(metric_arrays, labels, pick_idxs)

    xlsx_name = out_html.with_suffix(".xlsx").name
    sub = f"{n_total} checkpoints across 38 benchmark tabs ({escape(xlsx_name)})"
    title = "Full-Duplex Speech — Checkpoint Comparison"

    parts: list[str] = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<title>{title}</title>',
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>',
        _REPORT_STYLE_BLOCK,
        '</head>',
        '<body>',
        '',
        '<h1 id="top">Full-Duplex Speech Model — Checkpoint Comparison</h1>',
        f'<div class="subtitle">{sub}</div>',
        '',
        f'<h2>Top {len(picks)} Recommended Checkpoints</h2>',
        '<div class="top3-grid">',
        top_cards,
        '</div>',
        '',
        _REPORT_NAV_BAR_HTML,
        '',
        '<div class="insight"><strong>Key cross-cutting finding:</strong> There is a persistent tension between FDB v1 and FDB v1.5. Experiments optimized for FDB v1 are often among the worst on FDB v1.5. Conversely, the strongest FDB v1.5 ckpts only rank mid-tier on FDB v1. These benchmarks may be measuring partially conflicting behaviors.</div>',
        '<div class="warn"><strong>BFCL Parallel categories remain a systemic weakness.</strong> Parallel and Parallel-Multiple are well below acceptable thresholds across all checkpoints in this set — a systemic gap requiring dedicated training data with parallel function calls.</div>',
        '',
        fig1, '', fig2, '', fig3, '', fig4, '', fig5, '', fig6, '', fig7,
        '',
        recs_html,
        '',
        detailed_html,
        js_block,
        '',
        audio_html,
        _REPORT_BACK_TO_TOP,
        '</body>',
        '</html>',
    ]
    return "\n".join(parts) + "\n"


def _expand_metrics_dirs(raw: list[str]) -> list[Path]:
    """Expand the --metrics_dirs argument list into a flat list of source paths.

    Each entry is either:
      - a directory path (or a sidecar JSON file) → used as-is
      - a regular text file → each non-empty, non-comment line read as a path

    Mixing both in one invocation is allowed, e.g.:
      --metrics_dirs list.txt /path/to/run_a /path/to/run_b
    """
    sources: list[Path] = []
    for entry in raw:
        p = Path(entry).expanduser()
        # If it's a directory or a JSON sidecar, use directly.
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".json"):
            sources.append(p)
            continue
        # If it's a regular non-JSON file, treat as a list file (one path per line).
        if p.is_file():
            for raw_line in p.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                sources.append(Path(line).expanduser())
            continue
        # Doesn't exist — keep it, let downstream error message be precise.
        sources.append(p)
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_dirs",
        nargs="+",
        default=None,
        help=(
            "One or more checkpoint metric directories OR a path to a text file "
            "with one metrics_dir per line (lines starting with # are comments). "
            "You can mix both forms in a single invocation. Existing sidecar JSON "
            "files are accepted for compatibility. "
            "If omitted, the script reuses sidecars already in asset/ that match "
            "the --name stem; pair with --force_rerun to override this caching."
        ),
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Output HTML name or path (REQUIRED). If no directory is given, writes under asset/. "
             "An XLSX with the same stem is also written next to the HTML.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore cached sidecars and re-derive every metrics_dir. Requires --metrics_dirs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes for `build_checkpoint(source_dir)`. "
             "Default 0 = min(os.cpu_count(), 16). Set 1 for serial execution.",
    )
    parser.add_argument(
        "--html_mode",
        choices=("default", "report"),
        default="default",
        help="HTML rendering mode. 'default' (current behavior, minus audio) "
             "renders the comparison-style scorecard. 'report' renders the "
             "report-style dashboard with Top-N cards, Figures 1-7, "
             "Recommendations, Detailed (D1-D12), and curated lone-winner "
             "Audio Examples; requires a picks JSON at "
             "asset/<name>_top<top_N>.json (use the /pick_topN skill to "
             "produce it).",
    )
    parser.add_argument(
        "--top_N",
        type=int,
        default=3,
        help="Number of top checkpoints to highlight (only meaningful in "
             "--html_mode report). Default 3.",
    )
    return parser.parse_args()


def build_xlsx(ckpts: list[dict[str, Any]], out_path: Path) -> None:
    """Write an .xlsx whose numbers exactly mirror the HTML summary/detailed views.

    Sheets (in order):
      1. vb_aggregate          — VoiceBench AJ-formula aggregate + 9 components.
      2. <bench>                — main per-bench sheet: ckpt | average | <split.metric...>
                                  where 'average' is ckpt.headlines[bench] (same value
                                  as the 'average' row in the HTML summary table).
                                  'bba' uses native aggregate; no average column.
      3. <bench>.<split>        — one per subtest for benches with ≥2 splits and no
                                  '_aggregate'-style native aggregate split.

    Numerical values are the raw metric leaves from ckpt['metrics']; the
    HTML's summary table presents `aggregate_metric_for_ckpt(...)` (mean across
    splits per metric) — for the Excel we surface per-split values directly,
    which is what the HTML's detailed tables show. The headline/average column
    matches the HTML's "average" row exactly.
    """
    try:
        import openpyxl  # type: ignore
        from openpyxl import Workbook  # type: ignore
    except ImportError:
        print("warning: openpyxl not installed — skipping xlsx; "
              "install with `pip install openpyxl`", file=sys.stderr)
        return

    SKIP_RE = re.compile(r"\.scenario_results\.\d+\.")

    def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for k, v_ in value.items():
                _flatten(f"{prefix}.{k}" if prefix else k, v_, out)
        elif isinstance(value, list):
            return
        else:
            if SKIP_RE.search(prefix):
                return
            out[prefix] = value

    # Discover bench → ordered split list across all ckpts; detect aggregate splits.
    bench_order = list(BENCHMARKS)
    splits_by_bench: dict[str, list[str]] = {}
    for ck in ckpts:
        for bench, mode_dict in (ck.get("metrics") or {}).items():
            if not isinstance(mode_dict, dict): continue
            seq = splits_by_bench.setdefault(bench, [])
            for _, sd in mode_dict.items():
                if isinstance(sd, dict):
                    for split in sd:
                        if split not in seq:
                            seq.append(split)
    has_aggregate = {b: any("aggregate" in s.lower() for s in seq)
                     for b, seq in splits_by_bench.items()}

    def _headline(ck: dict[str, Any], bench: str) -> Any:
        hd = (ck.get("headlines") or {}).get(bench)
        if not isinstance(hd, dict) or not hd:
            return ""
        for m in KNOWN_MODES:
            if m in hd and hd[m] is not None:
                return hd[m]
        for v_ in hd.values():
            if v_ is not None:
                return v_
        return ""

    def _add_sheet(wb, name: str, cols: list[str], rows: list[tuple[str, dict]],
                   average_first: bool = False, average_lookup=None):
        ws = wb.create_sheet(title=name[:31])
        header = ["ckpt"] + (["average"] if average_first else []) + cols
        ws.append(header)
        for ck_name, vals in rows:
            row = [ck_name]
            if average_first:
                row.append(average_lookup(ck_name) if average_lookup else "")
            row += [vals.get(c, "") for c in cols]
            ws.append(row)
        ws.freeze_panes = "C2" if average_first else "B2"
        ws.column_dimensions["A"].width = 60
        start = 3 if average_first else 2
        if average_first:
            ws.column_dimensions["B"].width = 12
        for i, c in enumerate(cols, start=start):
            cl = ws.cell(row=1, column=i).column_letter
            ws.column_dimensions[cl].width = min(max(12, len(c) + 2), 40)

    def _build_per_bench(bench: str):
        cols: list[str] = []; rows: list[tuple[str, dict]] = []
        for ck in ckpts:
            flat: dict[str, Any] = {}
            _flatten("headline", (ck.get("headlines") or {}).get(bench) or {}, flat)
            _flatten("", (ck.get("metrics") or {}).get(bench) or {}, flat)
            if not flat: continue
            for k in flat:
                if k not in cols: cols.append(k)
            rows.append((ck["name"], flat))
        return cols, rows

    def _build_per_subtest(bench: str, split: str):
        cols: list[str] = []; rows: list[tuple[str, dict]] = []
        for ck in ckpts:
            bm = (ck.get("metrics") or {}).get(bench) or {}
            flat: dict[str, Any] = {}
            for mode, splits in bm.items():
                sv = (splits or {}).get(split) or {}
                if sv: _flatten(mode if len(bm) > 1 else "", sv, flat)
            if not flat: continue
            for k in flat:
                if k not in cols: cols.append(k)
            rows.append((ck["name"], flat))
        return cols, rows

    wb = Workbook(); wb.remove(wb.active)

    # 1. vb_aggregate sheet (AJ formula + components).
    ws_agg = wb.create_sheet(title="vb_aggregate")
    ws_agg.append([
        "ckpt", "aggregate",
        "P (mcq.openbookqa.acc)", "R (mcq.mmsu.acc)", "AD (mcq.bbh.acc)",
        "Y (sd_qa.panda+gpt /2)", "AH (ifeval.final)", "AI (advbench.refusal_rate)",
        "T (commoneval.gpt)*20", "AB (alpacaeval_full.gpt)*20", "AF (wildvoice.gpt)*20",
    ])
    def _num(d, *keys):
        if not isinstance(d, dict): return None
        for k in keys:
            v_ = d.get(k)
            if v_ is None: continue
            try: return float(v_)
            except (TypeError, ValueError): pass
        return None
    def _pct(v_):
        return v_ * 100.0 if (v_ is not None and 0.0 <= v_ <= 1.0) else v_
    def _s(x, mul=1): return "" if x is None else round(x * mul, 4)
    for ck in ckpts:
        metrics = ck.get("metrics") or {}
        agg = compute_vb_aggregate(metrics)
        agg_val = agg.get("greedy") if "greedy" in agg \
                  else next((v_ for v_ in agg.values() if v_ is not None), None)
        vm = (metrics.get("vb_mcq")    or {}).get("greedy", {}) or {}
        vn = (metrics.get("vb_nonmcq") or {}).get("greedy", {}) or {}
        P  = _num(vm.get("openbookqa"), "acc", "accuracy")
        R  = _num(vm.get("mmsu"),       "acc", "accuracy")
        AD = _num(vm.get("bbh"),        "acc", "accuracy")
        AH = _pct(_num(vn.get("ifeval"),   "final", "strict-prompt", "loose-prompt"))
        AI = _pct(_num(vn.get("advbench"), "refusal_rate", "refusal"))
        sp = _num(vn.get("sd_qa"), "panda")
        sg = _num(vn.get("sd_qa"), "gpt")
        Y  = ((sp + sg) / 2) if (sp is not None and sg is not None) \
             else (sp if sp is not None else sg)
        T  = _num(vn.get("commoneval"),      "gpt")
        AB = _num(vn.get("alpacaeval_full"), "gpt")
        AF = _num(vn.get("wildvoice"),       "gpt")
        ws_agg.append([
            ck["name"], agg_val,
            _s(P), _s(R), _s(AD), _s(Y), _s(AH), _s(AI),
            _s(T, 20), _s(AB, 20), _s(AF, 20),
        ])
    ws_agg.freeze_panes = "C2"
    ws_agg.column_dimensions["A"].width = 60
    ws_agg.column_dimensions["B"].width = 12
    for i in range(3, 12):
        ws_agg.column_dimensions[ws_agg.cell(row=1, column=i).column_letter].width = 26

    # 2 & 3. Per-bench main sheets + per-subtest sheets.
    for bench in bench_order:
        if bench not in splits_by_bench:
            continue
        cols, rows = _build_per_bench(bench)
        if rows:
            if bench != "bba":
                hb = {ck["name"]: _headline(ck, bench) for ck in ckpts}
                _add_sheet(wb, bench, cols, rows, average_first=True,
                           average_lookup=lambda n, _h=hb: _h.get(n, ""))
            else:
                _add_sheet(wb, bench, cols, rows)
        splits = splits_by_bench[bench]
        if len(splits) >= 2 and not has_aggregate[bench]:
            for split in splits:
                sc_cols, sc_rows = _build_per_subtest(bench, split)
                if sc_rows:
                    _add_sheet(wb, f"{bench}.{split}", sc_cols, sc_rows)

    wb.save(str(out_path))
    print(f"XLSX: {out_path}")


def main() -> None:
    args = parse_args()
    out_html = output_html_path(args.name)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if args.force_rerun and not args.metrics_dirs:
        sys.exit("--force_rerun requires --metrics_dirs")

    ckpts: list[dict[str, Any] | None] = []

    if args.metrics_dirs:
        sources = _expand_metrics_dirs(args.metrics_dirs)
        sidecars = sidecar_paths(out_html, sources)
        # First pass: handle fast cases (JSON inputs + cache hits) inline.
        # Collect slow cases (raw-dir builds) into `to_build` and process them in
        # a process pool below. ckpts is positional so we preserve order.
        ckpts = [None] * len(sources)
        to_build: list[tuple[int, Path, Path]] = []
        for i, (source, sidecar) in enumerate(zip(sources, sidecars)):
            if source.suffix.lower() == ".json":
                if not source.exists():
                    sys.exit(f"sidecar json does not exist: {source}")
                print(f"sidecar input loaded {source}")
                ckpts[i] = build_checkpoint(source)
                continue
            if sidecar is not None and sidecar.exists() and not args.force_rerun:
                print(f"sidecar file found {sidecar}, skipping data retrieval from {source}")
                ckpts[i] = build_checkpoint(sidecar)
                continue
            if not source.exists():
                sys.exit(f"metrics_dir does not exist: {source}")
            to_build.append((i, source, sidecar))

        if to_build:
            import os as _os
            from concurrent.futures import ProcessPoolExecutor, as_completed
            n_workers = args.workers if args.workers > 0 else min((_os.cpu_count() or 4), 16)
            n_workers = max(1, min(n_workers, len(to_build)))
            print(f"building {len(to_build)} sidecar(s) with {n_workers} worker process(es)...")
            for _, _, sc in to_build:
                print(f"creating {sc}...")
            if n_workers == 1:
                results = [(i, sc, build_checkpoint(src)) for i, src, sc in to_build]
            else:
                results = []
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    fut_to_meta = {pool.submit(build_checkpoint, src): (i, sc) for i, src, sc in to_build}
                    for fut in as_completed(fut_to_meta):
                        i, sc = fut_to_meta[fut]
                        try:
                            ck = fut.result()
                        except Exception as e:
                            sys.exit(f"failed to build {sc}: {e}")
                        results.append((i, sc, ck))
            for i, sc, ck in results:
                ckpts[i] = ck
                sc.parent.mkdir(parents=True, exist_ok=True)
                with sc.open("w", encoding="utf-8") as f:
                    json.dump(ck, f, indent=2, ensure_ascii=False)
                print(f"done: {sc}")
        ckpts = [c for c in ckpts if c is not None]
    else:
        # No --metrics_dirs → reuse all sidecars in the asset dir whose filename
        # matches the --name stem convention: <stem>_<anything>.json.
        asset_dir = out_html.parent
        stem = out_html.stem
        candidates = sorted(asset_dir.glob(f"{stem}_*.json"))
        if not candidates:
            sys.exit(
                f"--metrics_dirs not provided and no sidecars matching "
                f"{asset_dir}/{stem}_*.json were found. Re-run with --metrics_dirs."
            )
        print(f"--metrics_dirs not provided; reusing sidecars from {asset_dir}:")
        for sc in candidates:
            print(f"  {sc}")
        for sc in candidates:
            ckpts.append(build_checkpoint(sc))

    # Always write the XLSX next to the HTML — both modes need it (default uses
    # it as a sidecar; report mode references it as input to /pick_topN).
    out_xlsx = out_html.with_suffix(".xlsx")
    build_xlsx(ckpts, out_xlsx)

    if args.html_mode == "default":
        html = page_html(ckpts, out_html)
        out_html.write_text(html, encoding="utf-8")
        print(f"HTML: {out_html}")
    else:  # report
        picks_path = out_html.with_name(f"{out_html.stem}_top{args.top_N}.json")
        if args.force_rerun or not picks_path.exists():
            msg = textwrap.dedent(f"""
                ================================================================================
                NEXT STEP — Top-{args.top_N} selection via /pick_topN skill
                ================================================================================
                The metrics spreadsheet has been written to:

                  {out_xlsx}

                To pick the Top-{args.top_N} checkpoints, invoke this Claude skill in a chat:

                  /pick_topN --data_file {out_xlsx} --N {args.top_N}

                The skill (see asset/pick_topN.md) will write the picks JSON to:

                  {picks_path}

                After that file exists, re-run this same command to render the report HTML:

                  python3 scripts/visualize_metrics.py --name {args.name} --html_mode report --top_N {args.top_N}

                ================================================================================
            """).strip()
            print(msg)
            return
        try:
            picks_doc = json.loads(picks_path.read_text(encoding="utf-8"))
        except Exception as e:
            sys.exit(f"failed to parse picks file {picks_path}: {e}")
        picks_list = picks_doc.get("picks", []) if isinstance(picks_doc, dict) else []
        top_n_field = picks_doc.get("top_N") if isinstance(picks_doc, dict) else None
        if top_n_field != args.top_N or len(picks_list) != args.top_N:
            sys.exit(
                f"picks file {picks_path} doesn't match --top_N={args.top_N} "
                f"(top_N={top_n_field}, len(picks)={len(picks_list)})"
            )
        html = _render_report_html(ckpts, picks_doc, out_html, args.top_N)
        out_html.write_text(html, encoding="utf-8")
        print(f"HTML: {out_html}")

    print(f"View / download with:  bash scripts/serve_scorecard.sh")
    print(f"  (the printed URL list includes every .html and .xlsx in asset/)")


if __name__ == "__main__":
    main()
