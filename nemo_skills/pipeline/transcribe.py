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
Pipeline stage for audio transcription using a vLLM-hosted ASR model.

Two SLURM jobs are submitted per call:
  1. {expname}_model_dl  — CPU job that downloads the ASR model to HF_HOME if not cached
  2. {expname}           — GPU het job (vLLM server + transcription client), depends on job 1

Can be used as:
  - A CLI command: ns pipeline transcribe-audio --cluster ... --input-jsonl ... --output-jsonl ...
  - A callable from other pipeline scripts (e.g. run_bba_eval.py)
"""

import logging
import shlex
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_mounts,
    configure_client,
    get_exp,
    run_exp,
    set_python_path_and_wait_for_server,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def _submit_model_download_job(
    cluster_config: dict,
    model: str,
    expname: str,
    partition: str,
    log_dir: str,
    run_after,
    dry_run: bool,
):
    """Submit a CPU job that downloads model to HF_HOME if not already cached."""
    download_cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python nemo_skills/inference/download_model.py --model {model}"
    )
    with get_exp(expname, cluster_config) as exp:
        add_task(
            exp,
            cmd=download_cmd,
            task_name=expname,
            log_dir=log_dir,
            container=cluster_config["containers"].get("nemo-skills"),
            cluster_config=cluster_config,
            partition=partition,
            run_after=run_after,
        )
        run_exp(exp, cluster_config, dry_run=dry_run)
    return exp


@app.command(name="transcribe-audio")
@typer_unpacker
def transcribe_audio(
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs.",
    ),
    input_jsonl: str = typer.Option(..., help="Path to input JSONL file (e.g. output.jsonl from generation)"),
    output_jsonl: str = typer.Option(..., help="Path to output JSONL file to write ASR transcripts"),
    asr_model: str = typer.Option(..., help="ASR model to serve via vLLM (e.g. openai/whisper-large-v3)"),
    asr_server_args: str = typer.Option("", help="Extra arguments for the vLLM ASR server"),
    data_dir: str = typer.Option(None, help="Root dir for input question audio (writes question_asr field)"),
    server_gpus: int = typer.Option(1, help="Number of GPUs for the vLLM ASR server"),
    server_nodes: int = typer.Option(1, help="Number of nodes for the vLLM ASR server"),
    server_container: str = typer.Option(None, help="Container image for the vLLM server"),
    partition: str = typer.Option(None, help="GPU partition for the transcription het job"),
    cpu_partition: str = typer.Option(None, help="CPU partition for the model download job (defaults to partition)"),
    expname: str = typer.Option("transcribe_audio", help="Experiment name"),
    run_after: List[str] = typer.Option(None, help="Experiment names that must complete before this one starts"),
    installation_command: str = typer.Option(None, help="Installation command to run before the transcription job"),
    log_dir: str = typer.Option(None, help="Directory for Slurm logs"),
    force: bool = typer.Option(False, help="Overwrite output_jsonl if it already exists"),
    reuse_code: bool = typer.Option(True, help="If True, reuse code from the last submitted experiment in this session."),
    dry_run: bool = typer.Option(False, help="Print job config without submitting"),
    config_dir: str = typer.Option(None, help="Custom location for cluster configs"),
):
    """Transcribe audio files in a JSONL using a vLLM-hosted ASR model.

    Submits two jobs: a CPU download job to cache the model in HF_HOME, followed by
    a GPU het job (vLLM server + transcription client) that depends on the download.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)

    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    log_dir = check_mounts(cluster_config, log_dir)
    download_partition = cpu_partition or cluster_config.get("cpu_partition") or partition

    # Job 1: download model to HF_HOME on a CPU node
    download_expname = f"{expname}_model_dl"
    _submit_model_download_job(
        cluster_config=cluster_config,
        model=asr_model,
        expname=download_expname,
        partition=download_partition,
        log_dir=log_dir,
        run_after=run_after,
        dry_run=dry_run,
    )

    # Job 2: vLLM server + transcription client, depends on download
    server_config, server_address, _ = configure_client(
        model=asr_model,
        server_type="vllm",
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_address=None,
        server_args=asr_server_args,
        server_entrypoint=None,
        server_container=server_container,
        get_random_port=True,
        extra_arguments="",
    )

    transcribe_cmd = (
        f"python nemo_skills/inference/transcribe_audio.py"
        f" --server_url http://{server_address}/v1"
        f" --asr_model {asr_model}"
        f" --input_jsonl {input_jsonl}"
        f" --output_jsonl {output_jsonl}"
    )
    if data_dir:
        transcribe_cmd += f" --data_dir {shlex.quote(data_dir)}"
    if force:
        transcribe_cmd += " --force"

    cmd = set_python_path_and_wait_for_server(server_address, transcribe_cmd)

    with get_exp(expname, cluster_config) as exp:
        add_task(
            exp,
            cmd=cmd,
            task_name=expname,
            log_dir=log_dir,
            container=cluster_config["containers"].get("nemo-skills"),
            cluster_config=cluster_config,
            partition=partition,
            server_config=server_config,
            run_after=[download_expname],
            installation_command=installation_command,
            reuse_code=reuse_code,
        )
        run_exp(exp, cluster_config, dry_run=dry_run)

    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
