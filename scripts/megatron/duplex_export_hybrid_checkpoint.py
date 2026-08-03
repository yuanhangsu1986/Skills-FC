#!/usr/bin/env python3
"""Export a Megatron DuplexALM Torch-DCP checkpoint for DRIRF hybrid inference.

The exporter reads model tensors selectively from a distributed checkpoint. It
does not restore optimizer state and does not instantiate the training model.
Run it in an environment containing PyTorch, safetensors, and PyYAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.megatron.duplex_checkpoint_mapping import (
    expected_custom_outputs,
    frontend_target_key,
    is_frontend_key,
    is_llm_key,
    llm_target_tensors,
)

LOG = logging.getLogger("duplex_export")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".distcp")
DEFAULT_CONVERTED_ROOT = Path(
    "/lustre/fsw/portfolios/llmservice/users/nsrihari/full_duplex/avlm/init/e001/results/"
    "nemo_converted_megatron_voc_ckpts2/s2s_Jan_21_2026/result/"
    "PreT_PT0.95_SFT0.0_QA0.05_TEXT0.0_tloss0.0_MCQ0.0_ASR0.00_sysp0.0_se0.0_sm0.0_id0.0_"
    "its10.0_its20.0_wc0.0_fc0.0_fcv3_0.0_asr_dtc0_dst0_lt1.0_bos10.0_eos10.0_pad0.5_"
    "fcp0.3_fcs6.0_fce6.0_fcr3.0_fcc64.0/checkpoints/step-99015_hf"
)
DEFAULT_VOICECHAT_TEMPLATE = Path(
    "/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/s2s_Jan_21_2026/result/"
    "PreT_PT0.95_SFT0.0_QA0.05_TEXT0.0_tloss0.0_MCQ0.0_ASR0.00_sysp0.0_se0.0_sm0.0_id0.0_"
    "its10.0_its20.0_wc0.0_fc0.0_fcv3_0.0_asr_dtc0_dst0_lt1.0_bos10.0_eos10.0_pad0.5_"
    "fcp0.3_fcs6.0_fce6.0_fcr3.0_fcc64.0/checkpoints/step-99015_hf"
)
DEFAULT_TTS_CHECKPOINT = Path(
    "/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/function_calling_share/"
    "pretrained_s2s_rnnt_ckpt/Apr_02_2026/e70/"
    "e70-step28008-tts-eartts-34014_asr_cand3_0.6b-PK_ep0_01988_300"
)


def resolve_iteration_dir(checkpoint: Path, iteration: int | None) -> Path:
    """Resolve a checkpoint root or iteration directory to ``iter_*``."""

    checkpoint = checkpoint.resolve()
    if (checkpoint / ".metadata").is_file():
        return checkpoint
    if iteration is None:
        tracker = checkpoint / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            value = tracker.read_text().strip()
            if value.lower() == "release":
                candidate = checkpoint / "release"
            else:
                candidate = checkpoint / f"iter_{int(value):07d}"
            if candidate.is_dir():
                return candidate
        candidates = sorted(checkpoint.glob("iter_*"))
        if not candidates:
            raise FileNotFoundError(f"No Torch-DCP iteration found under {checkpoint}")
        return candidates[-1]
    candidate = checkpoint / f"iter_{iteration:07d}"
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def _import_runtime():
    try:
        import torch
        import torch.distributed.checkpoint as dcp
        from safetensors.torch import save_file
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    except ImportError as exc:
        raise SystemExit(
            "This command requires torch and safetensors. Run it in the Megatron training environment."
        ) from exc
    return torch, dcp, DefaultLoadPlanner, save_file


def _tensor_metadata(reader: Any) -> dict[str, Any]:
    metadata = reader.read_metadata()
    return {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("model.") and hasattr(value, "size") and hasattr(value, "properties")
    }


def load_selected_tensors(iter_dir: Path, predicate: Callable[[str], bool]) -> tuple[dict[str, Any], list[str]]:
    """Materialize only selected model tensors from a Torch-DCP checkpoint."""

    torch, dcp, DefaultLoadPlanner, _ = _import_runtime()
    reader = dcp.FileSystemReader(str(iter_dir))
    metadata = _tensor_metadata(reader)
    selected = sorted(key for key in metadata if predicate(key))
    if not selected:
        raise RuntimeError(f"No matching tensors in {iter_dir}")

    model_state: dict[str, Any] = {}
    for flat_key in selected:
        item = metadata[flat_key]
        model_key = flat_key[len("model.") :]
        model_state[model_key] = torch.empty(tuple(item.size), dtype=item.properties.dtype, device="cpu")

    LOG.info("Loading %d selected tensors from %s", len(selected), iter_dir)
    state = {"model": model_state}
    planner = DefaultLoadPlanner(allow_partial_load=True)
    dcp.load(state, storage_reader=reader, planner=planner)
    return {f"model.{key}": value for key, value in state["model"].items()}, selected


def _copy_template_assets(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name in {
            ".conversion.lock",
            ".conversion_lock",
            ".conversion_done",
            "conversion_logs",
        }:
            continue
        if entry.name.startswith("model") and entry.name.endswith(WEIGHT_SUFFIXES):
            continue
        if entry.name.endswith(WEIGHT_SUFFIXES) or entry.name.endswith(".index.json"):
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _prepare_configs(
    voicechat_template: Path,
    hf_template: Path,
    output_dir: Path,
    *,
    hidden_size: int,
    custom_input_dtype: str,
    has_asr_head: bool,
    has_function_head: bool,
) -> dict[str, Any]:
    _copy_template_assets(voicechat_template, output_dir)
    frontend_config_path = output_dir / "config.json"
    if not frontend_config_path.is_file():
        raise FileNotFoundError(f"VoiceChat template has no config.json: {voicechat_template}")
    frontend_config = _read_json(frontend_config_path)
    stt_config = frontend_config.setdefault("model", {}).setdefault("stt", {}).setdefault("model", {})
    stt_config["predict_user_text"] = has_asr_head
    stt_config["use_function_head"] = has_function_head
    stt_config["pretrained_rnnt_asr"] = None
    frontend_config.pop("_rnnt_merge_info", None)
    _write_json(frontend_config_path, frontend_config)

    vllm_dir = output_dir / "vllm_llm"
    _copy_template_assets(hf_template, vllm_dir)
    vllm_config_path = vllm_dir / "config.json"
    if not vllm_config_path.is_file():
        raise FileNotFoundError(f"HF LLM template has no config.json: {hf_template}")
    vllm_config = _read_json(vllm_config_path)
    if int(vllm_config.get("hidden_size", -1)) != hidden_size:
        raise ValueError(
            f"HF template hidden_size={vllm_config.get('hidden_size')} does not match checkpoint {hidden_size}"
        )
    vllm_config.update(
        {
            "custom_input_specs": [
                {"name": "combined_embeds", "dtype": custom_input_dtype, "dim": hidden_size}
            ],
            "custom_outputs": expected_custom_outputs(has_function_head=has_function_head),
            "has_asr_head": has_asr_head,
            "use_function_head": has_function_head,
        }
    )
    _write_json(vllm_config_path, vllm_config)
    return vllm_config


def _save_safetensors(path: Path, tensors: dict[str, Any], save_file: Any) -> None:
    if path.exists():
        path.unlink()
    contiguous = {key: tensor.detach().cpu().contiguous() for key, tensor in sorted(tensors.items())}
    save_file(contiguous, str(path), metadata={"format": "pt"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> None:
    _, _, _, save_file = _import_runtime()
    iter_dir = resolve_iteration_dir(args.checkpoint, args.iteration)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend_source, frontend_selected = load_selected_tensors(iter_dir, is_frontend_key)
    frontend: dict[str, Any] = {}
    frontend_unmapped: list[str] = []
    for source_key, tensor in frontend_source.items():
        target = frontend_target_key(source_key)
        if target is None:
            frontend_unmapped.append(source_key)
        else:
            frontend[target] = tensor

    required_frontend = {
        "stt_model.embed_tokens.weight",
        "stt_model.lm_head.weight",
    }
    missing_frontend = sorted(required_frontend - frontend.keys())
    if missing_frontend:
        raise RuntimeError(f"Missing required frontend tensors: {missing_frontend}")
    hidden_size = int(frontend["stt_model.embed_tokens.weight"].shape[1])
    has_asr_head = "stt_model.asr_head.weight" in frontend
    has_function_head = "stt_model.function_head.weight" in frontend

    vllm_config = _prepare_configs(
        args.voicechat_template,
        args.hf_llm_template,
        output_dir,
        hidden_size=hidden_size,
        custom_input_dtype=args.custom_input_dtype,
        has_asr_head=has_asr_head,
        has_function_head=has_function_head,
    )
    frontend_path = output_dir / "model.safetensors"
    _save_safetensors(frontend_path, frontend, save_file)
    del frontend_source, frontend

    llm_source, llm_selected = load_selected_tensors(iter_dir, is_llm_key)
    vllm_tensors: dict[str, Any] = {}
    llm_unmapped: list[str] = []
    for source_key, tensor in llm_source.items():
        mapped = llm_target_tensors(source_key, tensor, vllm_config)
        if not mapped:
            llm_unmapped.append(source_key)
        for target, value in mapped.items():
            if target in vllm_tensors:
                raise RuntimeError(f"Multiple source tensors map to {target}")
            vllm_tensors[target] = value

    required_llm = {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}
    if has_function_head:
        required_llm.add("stt_model.function_head.weight")
    missing_llm = sorted(required_llm - vllm_tensors.keys())
    if missing_llm:
        raise RuntimeError(f"Missing required vLLM tensors: {missing_llm}")
    if (frontend_unmapped or llm_unmapped) and not args.allow_unmapped:
        raise RuntimeError(
            "Unmapped selected tensors detected. Re-run with --allow-unmapped only after auditing: "
            f"frontend={frontend_unmapped}, llm={llm_unmapped}"
        )

    vllm_path = output_dir / "vllm_llm" / "model.safetensors"
    _save_safetensors(vllm_path, vllm_tensors, save_file)

    manifest = {
        "artifact_type": "megatron_duplex_hybrid",
        "format_version": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_iteration_dir": str(iter_dir),
        "voicechat_template": str(args.voicechat_template.resolve()),
        "hf_llm_template": str(args.hf_llm_template.resolve()),
        "hidden_size": hidden_size,
        "has_asr_head": has_asr_head,
        "has_function_head": has_function_head,
        "tts_checkpoint": str(args.tts_checkpoint.resolve()),
        "frontend": {
            "path": frontend_path.name,
            "selected_source_tensors": len(frontend_selected),
            "exported_tensors": len(frontend_selected) - len(frontend_unmapped),
            "unmapped": frontend_unmapped,
            "sha256": _file_sha256(frontend_path),
        },
        "vllm_llm": {
            "directory": "vllm_llm",
            "path": str(vllm_path.relative_to(output_dir)),
            "selected_source_tensors": len(llm_selected),
            "exported_tensors": len(vllm_tensors),
            "unmapped": llm_unmapped,
            "sha256": _file_sha256(vllm_path),
        },
    }
    _write_json(output_dir / "export_manifest.json", manifest)
    LOG.info("Export complete: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint root or iter_* directory")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration when --checkpoint is a run root")
    parser.add_argument(
        "--voicechat-template",
        type=Path,
        default=DEFAULT_VOICECHAT_TEMPLATE,
        help=f"NeMo VoiceChat directory providing config/assets (default: {DEFAULT_VOICECHAT_TEMPLATE})",
    )
    parser.add_argument(
        "--hf-llm-template",
        type=Path,
        default=DEFAULT_CONVERTED_ROOT / "llm_hf",
        help="Stock Nemotron-Nano-v2 HF directory",
    )
    parser.add_argument(
        "--tts-checkpoint",
        type=Path,
        default=DEFAULT_TTS_CHECKPOINT,
        help="EAR-TTS checkpoint recorded in the export manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custom-input-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--allow-unmapped", action="store_true", help="Record rather than reject selected unmapped tensors"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace generated files in an existing output directory"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    export(args)


if __name__ == "__main__":
    main()
