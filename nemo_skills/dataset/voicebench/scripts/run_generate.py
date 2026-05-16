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
Run generation on an arbitrary input.jsonl, parallelized into num_chunks Slurm jobs.

Uses the s2s_voicechat backend via nemo-skills pipeline. Server/cluster settings
are read from a YAML config (same format as voicebench eval configs).

Usage:
    python run_generate.py \
        --config voicebench_s2s_voicechat_offline_sound_config.yaml \
        --input_file /path/to/input.jsonl \
        --output_dir /path/to/output \
        --num_chunks 48

Full run command:
cd <nemo-skills-path>
. ./.venv/bin/activate && NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1  \
    python nemo_skills/dataset/voicebench/scripts/run_generate.py \
        --config nemo_skills/dataset/voicebench/scripts/run_generate_test_sdqa10.yaml \
        --input_file /lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir/voicebench/sd_qa/test_10.jsonl
"""

import argparse

import yaml

from nemo_skills.pipeline.generate import generate as nemo_generate


def wrap_arguments(arguments: str):
    """Returns a mock context object to allow using the cli entrypoints as functions."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    return MockContext(args=arguments.split(" "))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run generation on an arbitrary input.jsonl")
    parser.add_argument("--config", required=True, help="Path to YAML config file with server/cluster settings")
    parser.add_argument("--input_file", required=True, help="Path to input.jsonl file")
    parser.add_argument("--output_dir", help="Override output directory from config")
    parser.add_argument("--num_chunks", type=int, help="Override num_chunks from config")
    parser.add_argument("--model", help="Override model path from config")
    parser.add_argument("--partition", help="Override partition from config")
    parser.add_argument("--expname", help="Override expname from config")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_chunks is not None:
        config["num_chunks"] = args.num_chunks
    if args.model:
        config["model"] = args.model
    if args.partition:
        config["partition"] = args.partition
    if args.expname:
        config["expname"] = args.expname
    if args.dry_run:
        config["dry_run"] = True

    if not config.get("output_dir"):
        raise ValueError("output_dir must be specified in config or via --output_dir")

    # Build hydra extra args
    extra_args = ["++prompt_format=openai"]
    if config.get("server_server_type"):
        extra_args.append(f"++server.server_type={config['server_server_type']}")
    if config.get("data_dir"):
        extra_args.append(f"++eval_config.data_dir={config['data_dir']}")
    if config.get("max_samples"):
        extra_args.append(f"++max_samples={config['max_samples']}")

    # Build mount_paths: data_dir is mounted as /dataset so absolute audio paths
    # in jsonl (e.g. /dataset/voicebench/data/foo.wav) resolve correctly.
    mount_paths = None
    if config.get("data_dir"):
        mount_paths = f"{config['data_dir']}:/dataset"
    extra_args_str = " ".join(extra_args)

    print(f"Input file: {args.input_file}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Num chunks: {config.get('num_chunks', 1)}")
    print(f"Dry run: {config.get('dry_run', False)}")

    nemo_generate(
        ctx=wrap_arguments(extra_args_str),
        cluster=config["cluster"],
        input_file=args.input_file,
        output_dir=config["output_dir"],
        model=config["model"],
        server_type=config.get("server_type", "vllm"),
        server_gpus=config.get("server_gpus", 1),
        num_chunks=config.get("num_chunks", 1),
        server_container=config.get("server_container"),
        server_entrypoint=config.get("server_entrypoint"),
        server_args=config.get("server_args", ""),
        installation_command=config.get("installation_command"),
        partition=config.get("partition"),
        mount_paths=mount_paths,
        expname=config.get("expname", "generate"),
        dry_run=config.get("dry_run", False),
    )

    print("Done!")


if __name__ == "__main__":
    main()
