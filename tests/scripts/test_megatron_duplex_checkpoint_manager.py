import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from scripts.megatron.duplex_checkpoint_manager import (
    probe_directory_writable,
    resolve_iteration_dir,
    validate_export,
)
from scripts.megatron.duplex_export_hybrid_checkpoint import _copy_template_assets, _prepare_configs


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
    (export_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "stt": {
                        "model": {
                            "pretrained_llm": "nvidia/test-model",
                            "pretrained_weights": False,
                            "perception": {},
                        }
                    }
                }
            }
        )
    )
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


def test_validate_export_rejects_flat_voicechat_config(tmp_path):
    checkpoint, iteration_dir = _source_checkpoint(tmp_path)
    export_dir = _valid_export(tmp_path, checkpoint, iteration_dir)
    (export_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "pretrained_llm": "nvidia/test-model",
                    "perception": {},
                    "stt": {"model": {"use_function_head": True}},
                }
            }
        )
    )

    report = validate_export(export_dir, checkpoint)

    assert report["ok"] is False
    assert "model.stt.model.pretrained_llm is missing" in report["errors"]
    assert "model.stt.model.perception is missing" in report["errors"]


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


def test_prepare_configs_builds_drirf_wrapper_schema(tmp_path):
    voicechat = tmp_path / "voicechat"
    hf_template = tmp_path / "hf"
    tts = tmp_path / "tts"
    output = tmp_path / "output"
    voicechat.mkdir()
    hf_template.mkdir()
    tts.mkdir()
    (voicechat / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "pretrained_llm": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                    "pretrained_weights": True,
                    "perception": {"modality_adapter": {"d_model": 1024}},
                }
            }
        )
    )
    (hf_template / "config.json").write_text(json.dumps({"hidden_size": 4096}))
    (tts / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "stt": {
                        "data": {"source_sample_rate": 16000},
                        "model": {"incremental_loading": True, "pretrained_llm": "old/model"},
                    },
                    "speech_generation": {"model": {}},
                },
                "data": {"target_sample_rate": 22050},
            }
        )
    )

    _prepare_configs(
        voicechat,
        hf_template,
        tts,
        output,
        hidden_size=4096,
        custom_input_dtype="bfloat16",
        has_asr_head=False,
        has_function_head=True,
    )

    config = json.loads((output / "config.json").read_text())
    stt = config["model"]["stt"]["model"]
    assert stt["pretrained_llm"] == "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    assert stt["pretrained_weights"] is False
    assert stt["predict_user_text"] is False
    assert stt["use_function_head"] is True
    assert stt["perception"]["modality_adapter"]["d_model"] == 1024
    assert stt["incremental_loading"] is True
    assert config["model"]["speech_generation"] == {"model": {}}
