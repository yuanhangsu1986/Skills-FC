#!/usr/bin/env python3
"""Lightweight source/export validation for Megatron Duplex orchestration."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

ARTIFACT_TYPE = "megatron_duplex_hybrid"


def resolve_iteration_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if (checkpoint / ".metadata").is_file():
        return checkpoint
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"No latest_checkpointed_iteration.txt under {checkpoint}")
    value = tracker.read_text().strip()
    candidate = checkpoint / ("release" if value.lower() == "release" else f"iter_{int(value):07d}")
    if not (candidate / ".metadata").is_file():
        raise FileNotFoundError(f"No Torch-DCP metadata at {candidate / '.metadata'}")
    return candidate.resolve()


def probe_directory_writable(directory: Path) -> tuple[bool, str | None]:
    """Test actual create/remove access without colliding with user files."""

    try:
        with tempfile.TemporaryDirectory(prefix=".nemo_skills_write_test.", dir=directory):
            pass
    except OSError as exc:
        return False, str(exc)
    return True, None


def _load_manifest(export_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = export_dir / "export_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read manifest: {exc}"]
    if not isinstance(payload, dict):
        return None, ["manifest is not a JSON object"]
    return payload, []


def validate_export(export_dir: Path, checkpoint: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        source_iteration = resolve_iteration_dir(checkpoint)
    except (OSError, ValueError) as exc:
        return {"ok": False, "export_dir": str(export_dir), "errors": [str(exc)]}

    manifest, manifest_errors = _load_manifest(export_dir)
    errors.extend(manifest_errors)
    if manifest is None:
        return {
            "ok": False,
            "export_dir": str(export_dir.resolve()),
            "source_iteration_dir": str(source_iteration),
            "errors": errors,
        }

    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        errors.append(f"artifact_type is not {ARTIFACT_TYPE!r}")

    recorded_source = manifest.get("source_checkpoint")
    if not recorded_source or Path(recorded_source).resolve() != checkpoint.resolve():
        errors.append("manifest source_checkpoint does not match the requested checkpoint")
    recorded_iteration = manifest.get("source_iteration_dir")
    if not recorded_iteration or Path(recorded_iteration).resolve() != source_iteration:
        errors.append("manifest source_iteration_dir does not match the requested iteration")

    frontend = manifest.get("frontend") or {}
    vllm = manifest.get("vllm_llm") or {}
    relative_paths = (
        Path("config.json"),
        Path("vllm_llm/config.json"),
        Path(frontend.get("path", "model.safetensors")),
        Path(vllm.get("path", "vllm_llm/model.safetensors")),
    )
    for relative_path in relative_paths:
        path = export_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {path}")

    tts_checkpoint = manifest.get("tts_checkpoint")
    if not tts_checkpoint or not (Path(tts_checkpoint) / "config.json").is_file():
        errors.append("manifest tts_checkpoint is missing or has no config.json")

    # Opening the safetensors headers catches truncated/invalid files without
    # reading or hashing the multi-gigabyte tensor payloads.
    if not errors:
        try:
            from safetensors import safe_open

            expected = (
                (export_dir / relative_paths[2], {"stt_model.embed_tokens.weight", "stt_model.lm_head.weight"}),
                (export_dir / relative_paths[3], {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}),
            )
            for path, required_keys in expected:
                with safe_open(str(path), framework="numpy") as handle:
                    missing = required_keys - set(handle.keys())
                if missing:
                    errors.append(f"{path} is missing tensor keys: {sorted(missing)}")
        except (ImportError, OSError, ValueError) as exc:
            errors.append(f"cannot validate safetensors headers: {exc}")

    return {
        "ok": not errors,
        "export_dir": str(export_dir.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "source_iteration_dir": str(source_iteration),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-export")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--export-dir", type=Path, required=True)
    validate.add_argument("--quiet", action="store_true")

    probe = subparsers.add_parser("probe-writable")
    probe.add_argument("--directory", type=Path, required=True)
    probe.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-export":
        report = validate_export(args.export_dir, args.checkpoint)
    else:
        ok, error = probe_directory_writable(args.directory)
        report = {"ok": ok, "directory": str(args.directory.resolve()), "error": error}
    if not args.quiet:
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
