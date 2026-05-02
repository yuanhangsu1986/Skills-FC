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
Run conv_behav (conversational behavior) evaluation.

Pipeline stages:
  1. inference  — runs s2s_duplex_stt_infer.py (PyTorch Lightning + Hydra)
                  from nemo_code_path; produces
                    <output_dir>/eval-results/validation_logs/pred_wavs/
                    <output_dir>/eval-results/validation_logs/metadatas/<dataset_name>.json
  2. scoring    — calls eval_conversation_behavior.py from
                  <inference_nemo_code_path>/scripts/speech_eval/,
                  parses its output, writes metrics.json

Both stages are submitted as Slurm jobs via run_cmd.  Stage 2 runs after
stage 1 via run_after dependency.

Usage:
    python nemo_skills/dataset/conv_behav/scripts/run_eval.py \
        --config nemo_skills/dataset/conv_behav/scripts/conv_behav_config.yaml \
        [--output_dir /override/path] \
        [--model /override/path] \
        [--dry_run]
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_SCRIPTS_DIR = Path(__file__).resolve().parent


def build_inference_command(config: dict) -> str:
    nemo_code_path = config["nemo_code_path"]
    infer_script = f"{nemo_code_path}/examples/speechlm2/s2s_duplex_stt_infer.py"

    config_path = config.get("inference_config_path", "")
    config_name = config.get("inference_config_name", "")
    if config_path and config_name and (Path(config_path) / f"{config_name}.yaml").exists():
        pass  # use designated config
    else:
        if config_path:
            print(f"[warn] {config_path}/{config_name}.yaml not found, falling back to conv_behav_infer.yaml")
        config_path = str(_SCRIPTS_DIR)
        config_name = "conv_behav_infer"

    dataset_name = config["dataset_name"]
    force_turn_taking = str(config.get("force_turn_taking", False)).lower()
    num_nodes = config.get("num_nodes", 1)
    num_gpus = config.get("num_gpus", 1)
    launcher = f"torchrun --nproc_per_node={num_gpus}" if num_gpus > 1 else "python"

    hf_home = config.get("hf_home", "")
    env_prefix = f"export PYTHONPATH={nemo_code_path}:${{PYTHONPATH:-}}"
    if hf_home:
        env_prefix += f" && export HF_HOME={hf_home}"

    cmd = (
        f"{env_prefix} && "
        f"{launcher} {infer_script}"
        f" --config-path={config_path}"
        f" --config-name={config_name}"
        f" trainer.num_nodes={num_nodes}"
        f" ++ckpt_path=null"
        f" ++model.pretrained_s2s_model={config['model']}"
        f" ++model.pretrained_llm={config['pretrained_llm']}"
        f" ++exp_manager.explicit_log_dir={config['output_dir']}/eval-results"
        f" exp_manager.create_wandb_logger=false"
        f" '++data.validation_ds.datasets.{dataset_name}.shar_path={config['shar_input_dir']}'"
        f" ++model.force_turn_taking={force_turn_taking}"
        f" trainer.devices={num_gpus}"
    )
    if config.get("inference_args"):
        cmd += f" {config['inference_args']}"
    return cmd


def build_scoring_command(config: dict) -> str:
    eval_script = f"{config['nemo_code_path']}/scripts/speech_eval/eval_conversation_behavior.py"
    decoding_mode = "greedy" if config.get("force_turn_taking", False) else "sampling"
    cmd = (
        f"python nemo_skills/dataset/conv_behav/scripts/run_scoring.py"
        f" --output_dir {config['output_dir']}/eval-results"
        f" --shar_input_dir {config['shar_input_dir']}"
        f" --dataset_name {config['dataset_name']}"
        f" --eval_script_path {eval_script}"
        f" --decoding_mode {decoding_mode}"
        f" --barge_in_threshold_sec {config.get('barge_in_threshold_sec', 1.5)}"
        f" --tt_latency_threshold_sec {config.get('tt_latency_threshold_sec', 1.5)}"
        f" --tt_precision_buffer_sec {config.get('tt_precision_buffer_sec', 1.0)}"
        f" --tt_recall_buffer_sec {config.get('tt_recall_buffer_sec', 20.0)}"
        f" --vad_min_silence_duration_ms {config.get('vad_min_silence_duration_ms', 2000)}"
    )
    if config.get("scoring_force", False):
        cmd += " --force"
    torch_home = config.get("torch_home", "")
    if torch_home:
        cmd += f" --torch_home {torch_home}"
    return cmd


def run_inference_stage(config: dict, expname: str, dry_run: bool) -> bool:
    """Submit inference job. Returns True if submitted (False if skipped)."""
    output_json = Path(
        f"{config['output_dir']}/eval-results/validation_logs/metadatas/{config['dataset_name']}.json"
    )
    if output_json.exists() and not config.get("scoring_force", False):
        print(f"\n--- Stage 1: Skipping inference (found {output_json}) ---")
        return False

    print("\n--- Stage 1: Running inference (s2s_duplex_stt_infer.py) ---")
    num_gpus = config.get("num_gpus", 1)
    log_dir = str(Path(config["output_dir"]) / "eval-results" / "summarized-results")
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_inference_command(config),
        container=config.get("inference_container"),
        num_gpus=num_gpus,
        num_nodes=config.get("num_nodes", 1),
        partition=config.get("partition"),
        expname=expname,
        installation_command=config.get("installation_command"),
        log_dir=log_dir,
        reuse_code=False,
        exclusive=True if num_gpus >= 8 else None,
        dry_run=dry_run,
    )
    return True


def run_scoring_stage(config: dict, expname: str, run_after, dry_run: bool):
    """Submit scoring job."""
    print("\n--- Stage 2: Running scoring ---")
    log_dir = str(Path(config["output_dir"]) / "eval-results" / "summarized-results")
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_scoring_command(config),
        container=config.get("scoring_container") or config.get("inference_container"),
        num_gpus=config.get("scoring_gpus", 1),
        partition=config.get("partition"),
        run_after=run_after,
        expname=f"{expname}_score",
        installation_command=config.get("scoring_installation_command"),
        log_dir=log_dir,
        reuse_code=False,
        dry_run=dry_run,
    )


def run_conv_behav_eval(config: dict):
    inference_only = config.get("inference_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)
    expname = config.get("expname", "conv_behav")

    print(f"Dataset:    {config['dataset_name']}")
    print(f"Shar dir:   {config['shar_input_dir']}")
    print(f"Model:      {config['model']}")
    print(f"Output dir: {config['output_dir']}")

    infer_submitted = False
    if not scoring_only:
        infer_submitted = run_inference_stage(config, expname, dry_run)
        if inference_only:
            return

    run_after = [expname] if infer_submitted else None
    run_scoring_stage(config, expname, run_after, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Run conv_behav evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--shar_input_dir", help="Override shar input directory")
    parser.add_argument("--dataset_name", help="Override dataset name")
    parser.add_argument("--nemo_code_path", help="Override NeMo path (inference + scoring script)")
    parser.add_argument("--pretrained_llm", help="Override LLM backbone HF model ID or path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--scoring_only", action="store_true")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run scoring even if metrics.json exists")
    args = parser.parse_args()

    config = load_config(args.config)

    for key in ["model", "output_dir", "shar_input_dir", "dataset_name", "nemo_code_path", "pretrained_llm"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.inference_only:
        config["inference_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True
    if args.scoring_force:
        config["scoring_force"] = True

    output_dir = config.get("output_dir", "")
    if output_dir:
        commit_hash = get_git_commit_hash()
        if not output_dir.endswith(f"_{commit_hash}"):
            config["output_dir"] = f"{output_dir}_{commit_hash}"

    isolate_job_dir(config)
    run_conv_behav_eval(config)


if __name__ == "__main__":
    main()
