import json

import pytest

from nemo_skills.inference.server.serve_unified import _load_megatron_duplex_export


def test_normal_checkpoint_does_not_enter_megatron_path(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert _load_megatron_duplex_export(str(tmp_path)) is None


def test_unrelated_manifest_does_not_enter_megatron_path(tmp_path):
    (tmp_path / "export_manifest.json").write_text(json.dumps({"artifact_type": "something_else"}))
    assert _load_megatron_duplex_export(str(tmp_path)) is None


def test_megatron_manifest_resolves_split_artifacts(tmp_path):
    tts = tmp_path / "tts"
    tts.mkdir()
    (tts / "config.json").write_text("{}")
    engine = tmp_path / "vllm_llm"
    engine.mkdir()
    (engine / "config.json").write_text("{}")
    (engine / "model.safetensors").touch()
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "megatron_duplex_hybrid",
                "tts_checkpoint": str(tts),
                "vllm_llm": {"directory": "vllm_llm"},
            }
        )
    )

    manifest = _load_megatron_duplex_export(str(tmp_path))
    assert manifest is not None
    assert manifest["_engine_path"] == str(engine)


def test_megatron_manifest_fails_closed_when_incomplete(tmp_path):
    (tmp_path / "export_manifest.json").write_text(
        json.dumps({"artifact_type": "megatron_duplex_hybrid", "tts_checkpoint": "/missing"})
    )
    with pytest.raises(FileNotFoundError):
        _load_megatron_duplex_export(str(tmp_path))
