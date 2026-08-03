#!/usr/bin/env python3
"""Validate a Duplex hybrid export and compare native/hybrid result dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(export_dir: Path, verify_hashes: bool) -> int:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("validate-export requires safetensors") from exc

    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frontend_path = export_dir / manifest["frontend"]["path"]
    vllm_path = export_dir / manifest["vllm_llm"]["path"]
    required = (
        export_dir / "config.json",
        export_dir / "vllm_llm" / "config.json",
        frontend_path,
        vllm_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"ok": False, "missing_files": missing}, indent=2))
        return 1

    with safe_open(str(frontend_path), framework="pt", device="cpu") as handle:
        frontend_keys = set(handle.keys())
    with safe_open(str(vllm_path), framework="pt", device="cpu") as handle:
        vllm_keys = set(handle.keys())

    expected_frontend = {"stt_model.embed_tokens.weight", "stt_model.lm_head.weight"}
    expected_vllm = {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}
    if manifest.get("has_function_head"):
        expected_frontend.add("stt_model.function_head.weight")
        expected_vllm.add("stt_model.function_head.weight")
    errors: list[str] = []
    for label, expected, actual in (
        ("frontend", expected_frontend, frontend_keys),
        ("vllm_llm", expected_vllm, vllm_keys),
    ):
        absent = sorted(expected - actual)
        if absent:
            errors.append(f"{label} missing keys: {absent}")

    if verify_hashes:
        for label, path in (("frontend", frontend_path), ("vllm_llm", vllm_path)):
            actual = _sha256(path)
            expected = manifest[label]["sha256"]
            if actual != expected:
                errors.append(f"{label} SHA256 mismatch: {actual} != {expected}")

    report = {
        "ok": not errors,
        "export_dir": str(export_dir.resolve()),
        "frontend_tensor_count": len(frontend_keys),
        "vllm_tensor_count": len(vllm_keys),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def _load_tensor_dump(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("compare-tensors requires torch") from exc
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("tensors"), dict):
        payload = payload["tensors"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a tensor dictionary in {path}")
    return payload


def compare_tensors(native_path: Path, hybrid_path: Path, atol: float, rtol: float, min_cosine: float) -> int:
    import torch

    native = _load_tensor_dump(native_path)
    hybrid = _load_tensor_dump(hybrid_path)
    common = sorted(set(native) & set(hybrid))
    if not common:
        raise RuntimeError("The dumps have no common tensor names")
    rows: list[dict[str, Any]] = []
    ok = True
    for key in common:
        left, right = native[key], hybrid[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            continue
        if tuple(left.shape) != tuple(right.shape):
            rows.append(
                {
                    "name": key,
                    "ok": False,
                    "native_shape": list(left.shape),
                    "hybrid_shape": list(right.shape),
                }
            )
            ok = False
            continue
        left_f = left.float().reshape(-1)
        right_f = right.float().reshape(-1)
        max_abs = float((left_f - right_f).abs().max()) if left_f.numel() else 0.0
        cosine = float(torch.nn.functional.cosine_similarity(left_f, right_f, dim=0)) if left_f.numel() else 1.0
        allclose = bool(torch.allclose(left_f, right_f, atol=atol, rtol=rtol))
        row_ok = allclose or cosine >= min_cosine
        rows.append({"name": key, "ok": row_ok, "max_abs": max_abs, "cosine": cosine, "allclose": allclose})
        ok &= row_ok and not math.isnan(cosine)
    print(json.dumps({"ok": ok, "common_names": len(common), "comparisons": rows}, indent=2))
    return 0 if ok else 1


def _read_jsonl(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if id_field not in row:
                raise KeyError(f"{path}:{line_number} has no {id_field!r}")
            records[str(row[id_field])] = row
    return records


def compare_jsonl(native_path: Path, hybrid_path: Path, id_field: str, fields: list[str]) -> int:
    native = _read_jsonl(native_path, id_field)
    hybrid = _read_jsonl(hybrid_path, id_field)
    common = sorted(set(native) & set(hybrid))
    mismatches: list[dict[str, Any]] = []
    for sample_id in common:
        for field in fields:
            if native[sample_id].get(field) != hybrid[sample_id].get(field):
                mismatches.append(
                    {
                        "id": sample_id,
                        "field": field,
                        "native": native[sample_id].get(field),
                        "hybrid": hybrid[sample_id].get(field),
                    }
                )
    report = {
        "ok": not mismatches and set(native) == set(hybrid),
        "native_samples": len(native),
        "hybrid_samples": len(hybrid),
        "common_samples": len(common),
        "native_only": sorted(set(native) - set(hybrid)),
        "hybrid_only": sorted(set(hybrid) - set(native)),
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-export")
    validate.add_argument("--export-dir", type=Path, required=True)
    validate.add_argument("--verify-hashes", action="store_true")

    tensors = subparsers.add_parser("compare-tensors")
    tensors.add_argument("--native", type=Path, required=True)
    tensors.add_argument("--hybrid", type=Path, required=True)
    tensors.add_argument("--atol", type=float, default=2e-2)
    tensors.add_argument("--rtol", type=float, default=2e-2)
    tensors.add_argument("--min-cosine", type=float, default=0.999)

    results = subparsers.add_parser("compare-jsonl")
    results.add_argument("--native", type=Path, required=True)
    results.add_argument("--hybrid", type=Path, required=True)
    results.add_argument("--id-field", default="id")
    results.add_argument("--fields", nargs="+", default=["pred_text", "function_channel_text"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-export":
        code = validate_export(args.export_dir, args.verify_hashes)
    elif args.command == "compare-tensors":
        code = compare_tensors(args.native, args.hybrid, args.atol, args.rtol, args.min_cosine)
    else:
        code = compare_jsonl(args.native, args.hybrid, args.id_field, args.fields)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
