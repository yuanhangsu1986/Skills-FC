#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run conv_behav through the DRIRF offline Lightning validation loop, but load the
current DRIRF/DSFTS HuggingFace checkpoints with NemotronVoicechatInferenceWrapper.

The old DRIRF script directly instantiated DuplexSTTModel from a Hydra config.
That only works when the Hydra config itself fully describes the checkpoint
architecture. Current combined HF checkpoints require the wrapper's config merge
and manual state-dict loading path, so this adapter keeps the validation output
contract while reusing the tested loader.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any


def _set_cache_env(args) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", args.hf_home)
    if args.torch_home:
        os.environ.setdefault("TORCH_HOME", args.torch_home)


def _maybe_init_distributed():
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    launched_by_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if launched_by_torchrun and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return local_rank


def _wrapper_config(args, local_rank: int):
    from omegaconf import OmegaConf

    if "vllm" in args.engine_type:
        raise ValueError(
            "drirf_offline uses Lightning validation/offline_inference. "
            "Use pipeline_mode=drirf_incremental or pipeline_mode=nemo_eval for vLLM engine_type."
        )

    speaker_reference = args.speaker_reference or _speaker_reference_from_checkpoint(args.tts_checkpoint_path or args.model_path)
    cfg: dict[str, Any] = {
        "model_path": args.tts_checkpoint_path or args.model_path,
        "llm_checkpoint_path": args.llm_checkpoint_path or args.model_path,
        "speaker_reference": speaker_reference,
        "engine_type": args.engine_type,
        "decode_audio": True,
        "compute_dtype": args.dtype,
        "device": "cuda" if args.use_cuda_device else "cpu",
        "device_id": local_rank if args.use_cuda_device else None,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "force_turn_taking": args.force_turn_taking,
        "force_turn_taking_threshold": args.force_turn_taking_threshold,
        "force_turn_taking_pad_window": args.force_turn_taking_pad_window,
    }
    for key in (
        "inference_pad_boost",
        "inference_bos_boost",
        "inference_eos_boost",
        "inference_user_pad_boost",
        "inference_user_bos_boost",
        "inference_user_eos_boost",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return OmegaConf.create(cfg)


def _speaker_reference_from_checkpoint(model_path: str | None) -> str | None:
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fin:
            cfg = json.load(fin)
    except Exception:
        return None
    model_cfg = cfg.get("model", {}) or {}
    return model_cfg.get("inference_speaker_reference")


def _validation_config(model, args):
    from omegaconf import OmegaConf

    metadata_path = Path(args.output_dir) / "validation_logs" / "metadatas" / f"{args.dataset_name}.json"
    metadata_path.with_suffix(metadata_path.suffix + ".done").unlink(missing_ok=True)

    cfg = OmegaConf.create(OmegaConf.to_container(model.full_cfg, resolve=True))
    cfg.exp_manager.explicit_log_dir = str(Path(args.output_dir))
    cfg.exp_manager.create_tensorboard_logger = False
    cfg.exp_manager.create_checkpoint_callback = False
    cfg.exp_manager.create_wandb_logger = False
    cfg.exp_manager.use_datetime_version = False
    cfg.exp_manager.resume_if_exists = False
    cfg.exp_manager.resume_ignore_no_checkpoint = True

    cfg.trainer.num_nodes = args.num_nodes
    cfg.trainer.devices = args.num_gpus
    cfg.trainer.logger = False
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.use_distributed_sampler = False
    cfg.trainer.num_sanity_val_steps = 0
    if args.max_samples is not None:
        cfg.trainer.limit_val_batches = args.max_samples

    cfg.data.validation_ds.datasets = {args.dataset_name: {"shar_path": args.shar_input_dir}}
    cfg.data.validation_ds.batch_size = 1
    cfg.data.validation_ds.seed = 42
    cfg.data.validation_ds.shard_seed = "randomized"
    return cfg


def _apply_runtime_overrides(model, args) -> None:
    from omegaconf import OmegaConf

    stt_cfg = getattr(getattr(model, "stt_model", None), "cfg", None)
    if stt_cfg is None:
        return
    for key in (
        "force_turn_taking",
        "force_turn_taking_threshold",
        "force_turn_taking_pad_window",
        "inference_pad_boost",
        "inference_bos_boost",
        "inference_eos_boost",
        "inference_user_pad_boost",
        "inference_user_bos_boost",
        "inference_user_eos_boost",
    ):
        value = getattr(args, key, None)
        if value is not None:
            OmegaConf.update(stt_cfg, key, value, force_add=True)


def _attach_results_logger_flush(model) -> None:
    original_on_validation_epoch_end = model.on_validation_epoch_end

    def patched_on_validation_epoch_end(self, prefix="val"):
        result = original_on_validation_epoch_end(prefix=prefix)
        if hasattr(self, "results_logger"):
            self.results_logger.compute_and_save()
            metadata_path = Path(self.validation_save_path) / "metadatas" / f"{self._conv_behav_dataset_name}.json"
            if metadata_path.exists() and metadata_path.stat().st_size > 0:
                metadata_path.with_suffix(metadata_path.suffix + ".done").write_text("done\n", encoding="utf-8")
        return result

    model.on_validation_epoch_end = types.MethodType(patched_on_validation_epoch_end, model)


def run(args) -> None:
    _set_cache_env(args)
    local_rank = _maybe_init_distributed()

    import torch
    from lightning.pytorch import Trainer
    from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset
    from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
        NemotronVoicechatInferenceWrapper,
    )
    from nemo.utils.exp_manager import exp_manager
    from nemo.utils.trainer_utils import resolve_trainer_cfg
    from omegaconf import OmegaConf

    args.use_cuda_device = args.use_cuda_device and torch.cuda.is_available()
    if args.use_cuda_device:
        torch.set_float32_matmul_precision(args.matmul_precision)

    wrapper = NemotronVoicechatInferenceWrapper(model_cfg=_wrapper_config(args, local_rank))
    model = wrapper.model
    model._conv_behav_dataset_name = args.dataset_name
    _apply_runtime_overrides(model, args)
    _attach_results_logger_flush(model)

    if args.disable_rnnt_decoder_cuda_graphs:
        from recipes.multimodal.server.backends.s2s_incremental_backend_v2 import (
            disable_rnnt_decoder_cuda_graphs_for_model,
        )

        disable_rnnt_decoder_cuda_graphs_for_model(model, log_prefix="[conv_behav_drirf_offline]")

    cfg = _validation_config(model, args)
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = Path(exp_manager(trainer, cfg.get("exp_manager", None)))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    validation_save_path = str(log_dir / "validation_logs")
    model.validation_save_path = validation_save_path
    if hasattr(model, "stt_model"):
        model.stt_model.validation_save_path = validation_save_path

    tokenizer = model.stt_model.tokenizer if hasattr(model, "stt_model") else model.tokenizer
    dataset = DuplexS2SDataset(
        tokenizer=tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        include_turn_metadata=True,
    )
    datamodule = DataModule(cfg.data, tokenizer=tokenizer, dataset=dataset)
    trainer.validate(model, datamodule)


def main():
    parser = argparse.ArgumentParser(description="Run conv_behav DRIRF offline validation with wrapper-backed checkpoint loading")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--llm_checkpoint_path")
    parser.add_argument("--tts_checkpoint_path")
    parser.add_argument("--speaker_reference")
    parser.add_argument("--shar_input_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True, help="Path to eval-results directory")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--engine_type", default="native")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--matmul_precision", default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--force_turn_taking", action="store_true")
    parser.add_argument("--force_turn_taking_threshold", type=int, default=40)
    parser.add_argument("--force_turn_taking_pad_window", type=int, default=25)
    parser.add_argument("--disable_rnnt_decoder_cuda_graphs", action="store_true")
    parser.add_argument("--inference_pad_boost", type=float)
    parser.add_argument("--inference_bos_boost", type=float)
    parser.add_argument("--inference_eos_boost", type=float)
    parser.add_argument("--inference_user_pad_boost", type=float)
    parser.add_argument("--inference_user_bos_boost", type=float)
    parser.add_argument("--inference_user_eos_boost", type=float)
    parser.add_argument("--hf_home")
    parser.add_argument("--torch_home")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--no_cuda", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[conv_behav_drirf_offline] Warning: ignoring unsupported args: {' '.join(unknown)}")
    args.use_cuda_device = not args.no_cuda
    run(args)


if __name__ == "__main__":
    main()
