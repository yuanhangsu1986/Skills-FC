import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from scripts.megatron.duplex_checkpoint_manager import (
    probe_directory_writable,
    resolve_iteration_dir,
    validate_export,
)
from scripts.megatron.duplex_export_hybrid_checkpoint import _copy_template_assets


def _source_checkpoint(root: Path, iteration: int = 2400) -> tuple[Path, Path]:
    checkpoint = root / "checkpoints"
    iteration_dir = checkpoint / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / ".metadata").write_bytes(b"dcp")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    return checkpoint, iteration_dir


def _valid_export(root: Path, checkpoint: Path, iteration_dir: Path) -> Path:
    export_dir = root / "nemo_skills_converted"
    vllm_dir = export_dir / "vllm_llm"
    tts_dir = root / "tts"
    vllm_dir.mkdir(parents=True)
    tts_dir.mkdir()
    (export_dir / "config.json").write_text("{}\n")
    (vllm_dir / "config.json").write_text("{}\n")
    (tts_dir / "config.json").write_text("{}\n")
    save_file(
        {
            "stt_model.embed_tokens.weight": np.zeros((1, 1), dtype=np.float32),
            "stt_model.lm_head.weight": np.zeros((1, 1), dtype=np.float32),
        },
        export_dir / "model.safetensors",
    )
    save_file(
        {
            "backbone.embeddings.weight": np.zeros((1, 1), dtype=np.float32),
            "backbone.norm_f.weight": np.zeros((1,), dtype=np.float32),
            "lm_head.weight": np.zeros((1, 1), dtype=np.float32),
        },
        vllm_dir / "model.safetensors",
    )
    manifest = {
        "artifact_type": "megatron_duplex_hybrid",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_iteration_dir": str(iteration_dir.resolve()),
        "tts_checkpoint": str(tts_dir.resolve()),
        "frontend": {"path": "model.safetensors"},
        "vllm_llm": {"path": "vllm_llm/model.safetensors"},
    }
    (export_dir / "export_manifest.json").write_text(json.dumps(manifest))
    return export_dir


def test_resolve_iteration_and_probe_writable(tmp_path):
    checkpoint, iteration_dir = _source_checkpoint(tmp_path)
    assert resolve_iteration_dir(checkpoint) == iteration_dir.resolve()
    assert resolve_iteration_dir(iteration_dir) == iteration_dir.resolve()
    assert probe_directory_writable(checkpoint) == (True, None)
    assert not list(checkpoint.glob(".nemo_skills_write_test.*"))


def test_validate_export_matches_exact_source_iteration(tmp_path):
    checkpoint, iteration_dir = _source_checkpoint(tmp_path)
    export_dir = _valid_export(tmp_path, checkpoint, iteration_dir)

    report = validate_export(export_dir, checkpoint)

    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_export_rejects_new_source_iteration(tmp_path):
    checkpoint, iteration_dir = _source_checkpoint(tmp_path)
    export_dir = _valid_export(tmp_path, checkpoint, iteration_dir)
    next_iteration = checkpoint / "iter_0002401"
    next_iteration.mkdir()
    (next_iteration / ".metadata").write_bytes(b"dcp")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2401\n")

    report = validate_export(export_dir, checkpoint)

    assert report["ok"] is False
    assert "source_iteration_dir" in " ".join(report["errors"])


def test_validate_export_rejects_missing_artifact(tmp_path):
    checkpoint, iteration_dir = _source_checkpoint(tmp_path)
    export_dir = _valid_export(tmp_path, checkpoint, iteration_dir)
    (export_dir / "vllm_llm" / "model.safetensors").unlink()

    report = validate_export(export_dir, checkpoint)

    assert report["ok"] is False
    assert "missing or empty artifact" in " ".join(report["errors"])


def test_template_copy_excludes_conversion_bookkeeping(tmp_path):
    source = tmp_path / "template"
    destination = tmp_path / "export"
    source.mkdir()
    (source / "config.json").write_text("{}\n")
    (source / ".conversion_lock").touch()
    (source / ".conversion_done").touch()
    (source / "conversion_logs").mkdir()

    _copy_template_assets(source, destination)

    assert (destination / "config.json").is_file()
    assert not (destination / ".conversion_lock").exists()
    assert not (destination / ".conversion_done").exists()
    assert not (destination / "conversion_logs").exists()
