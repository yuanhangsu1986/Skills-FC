# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import typer

# isort: off
from nemo_skills.pipeline.app import app

# isort: on

# need the imports to make sure the commands are registered
from nemo_skills.pipeline.convert import convert
from nemo_skills.pipeline.eval import eval
from nemo_skills.pipeline.generate import generate
from nemo_skills.pipeline.megatron_lm.train import train_megatron_lm
from nemo_skills.pipeline.nemo_evaluator import nemo_evaluator
from nemo_skills.pipeline.nemo_rl.grpo import grpo_nemo_rl
from nemo_skills.pipeline.nemo_rl.sft import sft_nemo_rl
from nemo_skills.pipeline.prepare_data import prepare_data
from nemo_skills.pipeline.robust_eval import robust_eval
from nemo_skills.pipeline.run_cmd import run_cmd
from nemo_skills.pipeline.setup import setup
from nemo_skills.pipeline.start_server import start_server
from nemo_skills.pipeline.transcribe import transcribe_audio
from nemo_skills.pipeline.summarize_results import summarize_results
from nemo_skills.pipeline.summarize_robustness import summarize_robustness
from nemo_skills.pipeline.verl.ppo import ppo_verl

typer.main.get_command_name = lambda name: name


def wrap_arguments(arguments: str | list):
    """Returns a mock context object to allow using the cli entrypoints as functions."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    args = arguments if isinstance(arguments, list) else arguments.split(" ")
    return MockContext(args=args)


def maybe_merge_before_scoring(
    config: dict,
    eval_results_path: str,
    expname: str,
    run_after=None,
    dry_run: bool = False,
):
    """Submit a CPU merge job if chunks exist but the merged output.jsonl is missing.

    Returns the run_after value to use for the scoring job: either [merge_expname]
    when a merge job was submitted, or the original run_after otherwise.
    Call this before every scoring run_cmd when num_chunks > 1.
    """
    from pathlib import Path

    from nemo_skills.pipeline.utils import get_merge_cmd

    num_chunks = int(config.get("num_chunks", 1) or 1)
    if num_chunks <= 1:
        return run_after
    # When run_after is provided, generation was just submitted and eval.py already
    # added a merge job inside that pipeline — no need for a second one.
    if run_after is not None:
        return run_after
    output_jsonl_done = Path(eval_results_path) / "output.jsonl.done"
    if output_jsonl_done.exists():
        return run_after
    merge_expname = f"{expname}_merge"
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=get_merge_cmd(eval_results_path, num_chunks, random_seed=None),
        partition=config.get("cpu_partition") or config.get("partition"),
        run_after=run_after,
        expname=merge_expname,
        installation_command=config.get("installation_command"),
        log_dir=f"{eval_results_path}/summarized-results",
        dry_run=dry_run,
    )
    return [merge_expname]


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    app()
