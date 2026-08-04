"""Artifact-format helpers for converted Megatron Duplex checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_TYPE = "megatron_duplex_hybrid"


def read_export_manifest(export_dir: str | Path) -> dict[str, Any] | None:
    """Read a marked Megatron Duplex manifest, or return ``None`` otherwise."""

    export_dir = Path(export_dir)
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Megatron Duplex manifest is not a JSON object: {manifest_path}")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        return None
    return manifest


def load_megatron_duplex_export(export_dir: str | Path) -> dict[str, Any] | None:
    """Return a runtime-validated manifest, or ``None`` for a normal checkpoint."""

    export_dir = Path(export_dir)
    manifest = read_export_manifest(export_dir)
    if manifest is None:
        return None

    frontend = export_dir / "model.safetensors"
    engine_path = export_dir / manifest.get("vllm_llm", {}).get("directory", "vllm_llm")
    if not frontend.is_file():
        raise FileNotFoundError(f"Megatron Duplex export is missing {frontend}")
    if not (engine_path / "config.json").is_file():
        raise FileNotFoundError(f"Megatron Duplex vLLM engine is missing config.json: {engine_path}")
    if not (engine_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Megatron Duplex vLLM engine is missing model.safetensors: {engine_path}")

    tts_checkpoint = manifest.get("tts_checkpoint")
    if not tts_checkpoint or not (Path(tts_checkpoint) / "config.json").is_file():
        raise FileNotFoundError(f"Megatron Duplex export has an invalid TTS checkpoint: {tts_checkpoint}")

    manifest["_engine_path"] = str(engine_path)
    return manifest
