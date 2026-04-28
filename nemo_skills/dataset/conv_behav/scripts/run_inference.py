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
Offline inference for conv_behav evaluation.

Loads a HuggingFace-format DuplexS2SSpeechDecoderModel checkpoint and runs
offline inference (trainer.validate) on a lhotse shar dataset.

Produces under <output_dir>/:
  validation_logs/pred_wavs/{dataset_name}_{sample_id}.wav   (stereo: user L, agent R)
  validation_logs/metadatas/{dataset_name}.json              (pred_text, pred_src_text, ...)

Usage:
    python run_inference.py \
        --model_path /path/to/hf_checkpoint_dir \
        --shar_input_dir /path/to/lhotse_shar \
        --output_dir /path/to/output \
        --nemo_code_path /path/to/NeMo \
        [--dataset_name team_20251124] \
        [--num_gpus 1] \
        [--batch_size 1] \
        [--force_turn_taking] \
        [--precision bf16-true]
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


def _apply_inference_overrides(cfg, extra_args: list) -> None:
    """Apply --key [value] pairs from extra_args as flat OmegaConf overrides on cfg.

    Flag-style args (no following value, or next token starts with '--') are set
    to True.  All values are coerced: 'true'/'false' → bool, integers → int,
    floats → float, everything else stays a string.
    """
    def _coerce(v: str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token.lstrip("-")
        # peek at next token to decide if it's the value or the next flag
        if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
            value = _coerce(extra_args[i + 1])
            i += 2
        else:
            value = True
            i += 1
        OmegaConf.update(cfg, key, value, merge=True)
        print(f"[inference] model.cfg override: {key} = {value!r}")


def main():
    parser = argparse.ArgumentParser(description="conv_behav offline inference")
    parser.add_argument("--model_path", required=True, help="HF-format checkpoint directory")
    parser.add_argument("--shar_input_dir", required=True, help="Lhotse shar directory with user audio")
    parser.add_argument("--output_dir", required=True, help="Root output dir; validation_logs/ written here")
    parser.add_argument("--nemo_code_path", required=True, help="Path to NeMo codebase (prepended to sys.path so the right speechlm2 version is used)")
    parser.add_argument("--dataset_name", default="conv_behav", help="Name used for output files / metadatas key")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", default="bf16-true")
    # All remaining --key [value] pairs are applied as flat OmegaConf overrides
    # on model.cfg after loading the HF checkpoint.  This lets the config YAML
    # pass inference knobs (temperature, top_p, force_turn_taking, use_codec_cache,
    # inference_*_boost, etc.) the same way BBA/VB/FDB pass them via server_args.
    args, extra_args = parser.parse_known_args()

    # Prepend Kevin's NeMo codebase so nemo.collections.speechlm2 resolves
    # to the right version regardless of what is installed in the container.
    sys.path.insert(0, args.nemo_code_path)

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from lightning.pytorch import Trainer
    from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
    from nemo.utils.exp_manager import exp_manager
    from nemo.utils.trainer_utils import resolve_trainer_cfg

    # Load model from HF-format checkpoint (config.json + weights).
    # from_pretrained sets pretrained_weights=False so component checkpoints
    # inside the config are not re-fetched; the trained weights are loaded directly.
    model = DuplexS2SSpeechDecoderModel.from_pretrained(args.model_path)

    # Apply flat OmegaConf overrides from extra_args (e.g. --temperature 0.0
    # --force_turn_taking --use_codec_cache).  Flag args (no value) are set to True.
    _apply_inference_overrides(model.cfg, extra_args)

    # Use DDP only when more than one GPU is requested.
    if args.num_gpus > 1:
        strategy = {
            "_target_": "lightning.pytorch.strategies.DDPStrategy",
            "gradient_as_bucket_view": True,
            "find_unused_parameters": True,
        }
        torch.distributed.init_process_group(backend="nccl")
    else:
        strategy = "auto"

    trainer_cfg = OmegaConf.create({
        "devices": args.num_gpus,
        "num_nodes": args.num_nodes,
        "accelerator": "gpu",
        "precision": args.precision,
        "logger": False,
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "strategy": strategy,
    })
    trainer = Trainer(**resolve_trainer_cfg(trainer_cfg))

    # exp_manager sets trainer.log_dir = output_dir, which ResultsLogger uses
    # to write pred_wavs/ and metadatas/ under validation_logs/.
    exp_mgr_cfg = OmegaConf.create({
        "explicit_log_dir": args.output_dir,
        "resume_if_exists": False,
        "resume_ignore_no_checkpoint": True,
        "create_checkpoint_callback": False,
        "create_tensorboard_logger": False,
    })
    exp_manager(trainer, exp_mgr_cfg)

    # Data config matching the structure expected by DataModule / DuplexS2SDataset.
    data_cfg = OmegaConf.create({
        "frame_length": 0.08,
        "source_sample_rate": 16000,
        "target_sample_rate": 22050,
        "input_roles": ["user", "User"],
        "output_roles": ["agent", "Assistant"],
        "validation_ds": {
            "datasets": {
                args.dataset_name: {
                    "shar_path": args.shar_input_dir,
                }
            },
            "sample_rate": 22050,
            "batch_size": args.batch_size,
            "seed": 42,
            "shard_seed": "randomized",
        },
    })

    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=data_cfg.frame_length,
        source_sample_rate=data_cfg.source_sample_rate,
        target_sample_rate=data_cfg.target_sample_rate,
        input_roles=data_cfg.input_roles,
        output_roles=data_cfg.output_roles,
    )
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    main()
