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

Supported pipeline modes:
  dsfts_offline
      Original DSFTS Hydra/Lightning validation pipeline used by the deleted
      conv_behav_config_greedy.yaml. This keeps inference and scoring in the
      DSFTS tree and preserves the validation_logs output contract.

  drirf_offline
      Offline Lightning validation/scoring contract, but checkpoint loading is
      delegated to NemotronVoicechatInferenceWrapper so current DRIRF/DSFTS HF
      checkpoints use the same config merge and state-dict loading path as the
      shared backend.

  drirf_incremental
      Keeps the original conv_behav output contract
      (<output_dir>/eval-results/validation_logs/...) and scoring script, but
      runs inference through the existing DRIRF s2s_incremental_v2 backend.

  nemo_eval
      Converts the fixed conv_behav SHAR audio into a NeMo Skills benchmark,
      runs serve_unified/s2s_incremental_v2 through nemo_eval, and scores the
      resulting output.jsonl with eval_conversation_behavior_v2.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shlex
import shutil
import tarfile
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import maybe_merge_before_scoring, run_cmd, wrap_arguments
from nemo_skills.pipeline.utils.cluster import get_git_commit_hash, isolate_job_dir

_DRIRF_OFFLINE_SCRIPT = (
    "/nemo_run/code/nemo_skills/dataset/conv_behav/scripts/conv_behav_drirf_offline_infer.py"
)
_DRIRF_INCREMENTAL_SCRIPT = (
    "/nemo_run/code/nemo_skills/dataset/conv_behav/scripts/conv_behav_drirf_incremental_infer.py"
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _q(value) -> str:
    return shlex.quote(str(value))


def _config_get(config: dict, nested: str, key: str, default=None):
    nested_cfg = config.get(nested, {}) or {}
    if key in nested_cfg:
        return nested_cfg[key]
    return config.get(key, default)


def _benchmark_name(config: dict) -> str:
    return config.get("benchmark") or f"conv_behav.{config['dataset_name']}"


def _safe_audio_name(sample_id: str) -> str:
    return sample_id.replace("/", "_")


def _cut_text(cut: dict, roles: tuple[str, ...] | None = None) -> str:
    texts = []
    for supervision in sorted(cut.get("supervisions", []), key=lambda x: x.get("start", 0.0)):
        if roles and supervision.get("speaker") not in roles:
            continue
        text = (supervision.get("text") or "").strip()
        if text:
            texts.append(text)
    return " | ".join(texts)


def _write_init(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")


def _iter_shar_cuts(shar_input_dir: Path):
    cut_files = sorted(shar_input_dir.glob("cuts.*.jsonl.gz"))
    if not cut_files:
        raise FileNotFoundError(f"No cuts.*.jsonl.gz files found under {shar_input_dir}")
    for cut_file in cut_files:
        with gzip.open(cut_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _extract_shar_recordings(shar_input_dir: Path, audio_dir: Path, cut_ids: set[str]) -> dict[str, str]:
    """Extract fixed input recordings from SHAR and return cut id -> relative audio path."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = {}
    for tar_path in sorted(shar_input_dir.glob("recording.*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                basename = Path(member.name).name
                candidates = [basename]
                if "." in basename:
                    candidates.append(basename.rsplit(".", 1)[0])
                cut_id = next((candidate for candidate in candidates if candidate in cut_ids), None)
                if cut_id is None:
                    continue

                dst_name = _safe_audio_name(basename)
                dst_path = audio_dir / dst_name
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, open(dst_path, "wb") as fout:
                    shutil.copyfileobj(src, fout)
                audio_paths[cut_id] = f"data/{dst_name}"
    return audio_paths


def prepare_nemo_eval_data(config: dict) -> None:
    """Convert conv_behav SHAR into a NeMo Skills benchmark directory."""
    data_dir = Path(config["data_dir"])
    dataset_name = config["dataset_name"]
    benchmark = _benchmark_name(config)
    if not benchmark.startswith("conv_behav."):
        raise ValueError(f"conv_behav nemo_eval benchmark must look like conv_behav.<name>, got {benchmark}")

    shar_input_dir = Path(config["shar_input_dir"])
    benchmark_dir = data_dir / "conv_behav" / benchmark.split(".", 1)[1]
    audio_dir = benchmark_dir / "data"
    test_jsonl = benchmark_dir / "test.jsonl"

    force = bool(config.get("prepare_force", False))
    if test_jsonl.exists() and not force:
        print(f"[prepare] Reusing prepared conv_behav data: {test_jsonl}")
        return

    cuts = list(_iter_shar_cuts(shar_input_dir))
    if not cuts:
        raise RuntimeError(f"No cuts found in {shar_input_dir}")

    cut_ids = {cut["id"] for cut in cuts}
    audio_paths = _extract_shar_recordings(shar_input_dir, audio_dir, cut_ids)
    missing = sorted(cut_ids - set(audio_paths))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Could not find SHAR recording payloads for {len(missing)} conv_behav cuts. "
            f"Examples: {preview}"
        )

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    with open(test_jsonl, "w", encoding="utf-8") as fout:
        for cut in cuts:
            rel_audio_path = audio_paths[cut["id"]]
            problem = _cut_text(cut) or cut["id"]
            entry = {
                "id": cut["id"],
                "dataset": dataset_name,
                "sample_id": cut["id"],
                "problem": problem,
                "audio_path": rel_audio_path,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "",
                        "audio": {"path": f"conv_behav/{benchmark.split('.', 1)[1]}/{rel_audio_path}"},
                    },
                ],
            }
            user_text = _cut_text(cut, roles=("user",))
            if user_text:
                entry["messages_text"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
                entry["messages_text_audio"] = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_text,
                        "audio": {"path": f"conv_behav/{benchmark.split('.', 1)[1]}/{rel_audio_path}"},
                    },
                ]
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _write_init(
        data_dir / "conv_behav" / "__init__.py",
        (
            "DATASET_GROUP = 'speechlm'\n"
            "IS_BENCHMARK_GROUP = True\n"
            f"BENCHMARKS = {{{benchmark!r}: {{}}}}\n"
            "GENERATION_ARGS = '++prompt_format=openai ++eval_type=null'\n"
            "EVAL_ARGS = '++eval_type=null'\n"
            "SCORE_MODULE = None\n"
        ),
        force=True,
    )
    _write_init(
        benchmark_dir / "__init__.py",
        "GENERATION_ARGS = '++prompt_format=openai ++eval_type=null'\nEVAL_ARGS = '++eval_type=null'\n",
        force=True,
    )
    print(f"[prepare] Wrote {test_jsonl} with {len(cuts)} samples")


def _base_env_prefix(config: dict, code_path_key: str = "nemo_code_path") -> str:
    code_path = config.get(code_path_key)
    if code_path_key != "nemo_code_path" and not code_path:
        code_path = config.get("nemo_code_path")
    env_parts = []
    if code_path:
        env_parts.append(f"export PYTHONPATH={code_path}:/nemo_run/code:${{PYTHONPATH:-}}")
    hf_home = config.get("hf_home")
    if hf_home:
        env_parts.append(f"export HF_HOME={_q(hf_home)}")
    torch_home = config.get("torch_home")
    if torch_home:
        env_parts.append(f"export TORCH_HOME={_q(torch_home)}")
    return " && ".join(env_parts)


def build_dsfts_offline_inference_command(config: dict) -> str:
    nemo_code_path = config["nemo_code_path"]
    script = config.get(
        "dsfts_offline_inference_script_path",
        f"{nemo_code_path}/examples/speechlm2/s2s_duplex_stt_infer.py",
    )

    config_path = config.get("inference_config_path", "")
    config_name = config.get("inference_config_name", "")
    if not (config_path and config_name and (Path(config_path) / f"{config_name}.yaml").exists()):
        if config_path:
            print(f"[warn] {config_path}/{config_name}.yaml not found, falling back to conv_behav_infer.yaml")
        config_path = str(Path(__file__).resolve().parent)
        config_name = "conv_behav_infer"

    dataset_name = config["dataset_name"]
    num_nodes = config.get("num_nodes", 1)
    num_gpus = config.get("num_gpus", 1)
    python_exec = config.get("inference_container_python_exec", "python3")
    if num_nodes == 1 and num_gpus > 1:
        launcher = f"{python_exec} -m torch.distributed.run --standalone --nproc_per_node={num_gpus}"
    elif num_nodes > 1:
        launcher = f"{python_exec} -m torch.distributed.run --nnodes={num_nodes} --nproc_per_node={num_gpus}"
    else:
        launcher = python_exec

    force_turn_taking = str(bool(config.get("force_turn_taking", False))).lower()
    cmd_parts = [
        launcher,
        _q(script),
        f"--config-path={_q(config_path)}",
        f"--config-name={_q(config_name)}",
        f"trainer.num_nodes={num_nodes}",
        "++ckpt_path=null",
        f"++model.pretrained_s2s_model={_q(config['model'])}",
        f"++model.pretrained_llm={_q(config['pretrained_llm'])}",
        f"++exp_manager.explicit_log_dir={_q(Path(config['output_dir']) / 'eval-results')}",
        "exp_manager.create_wandb_logger=false",
        _q(f"++data.validation_ds.datasets.{dataset_name}.shar_path={config['shar_input_dir']}"),
        f"++model.force_turn_taking={force_turn_taking}",
        f"trainer.devices={num_gpus}",
    ]

    for key in (
        "force_turn_taking_threshold",
        "force_turn_taking_pad_window",
        "temperature",
        "top_p",
        "repetition_penalty",
    ):
        if config.get(key) is not None:
            cmd_parts.append(f"++model.{key}={config[key]}")

    if config.get("inference_args"):
        cmd_parts.append(config["inference_args"])

    env_prefix = _base_env_prefix(config)
    output_json = Path(config["output_dir"]) / "eval-results" / "validation_logs" / "metadatas" / f"{dataset_name}.json"
    done_marker = output_json.with_suffix(output_json.suffix + ".done")
    command = (
        " ".join(cmd_parts)
        + f" && test -s {_q(output_json)}"
        + f" && printf 'done\\n' > {_q(done_marker)}"
    )
    return f"{env_prefix} && {command}" if env_prefix else command


def build_drirf_offline_inference_command(config: dict) -> str:
    num_nodes = config.get("num_nodes", 1)
    num_gpus = config.get("num_gpus", 1)
    python_exec = config.get("inference_container_python_exec", "python3")
    script = config.get("drirf_offline_inference_script_path", _DRIRF_OFFLINE_SCRIPT)
    if num_nodes == 1:
        launcher = f"{python_exec} -m torch.distributed.run --standalone --nproc_per_node={num_gpus}"
    else:
        launcher = f"{python_exec} -m torch.distributed.run --nnodes={num_nodes} --nproc_per_node={num_gpus}"

    cmd_parts = [
        launcher,
        _q(script),
        "--model_path",
        _q(config["model"]),
        "--shar_input_dir",
        _q(config["shar_input_dir"]),
        "--dataset_name",
        _q(config["dataset_name"]),
        "--output_dir",
        _q(Path(config["output_dir"]) / "eval-results"),
        "--num_gpus",
        str(num_gpus),
        "--num_nodes",
        str(num_nodes),
        "--engine_type",
        _q(config.get("drirf_offline_engine_type", "native")),
        "--matmul_precision",
        _q(config.get("matmul_precision", "medium")),
        "--temperature",
        str(config.get("temperature", 0.0)),
        "--top_p",
        str(config.get("top_p", 1.0)),
        "--repetition_penalty",
        str(config.get("repetition_penalty", 1.0)),
    ]

    optional_values = {
        "llm_checkpoint_path": config.get("llm_checkpoint_path") or config.get("model"),
        "tts_checkpoint_path": config.get("tts_checkpoint_path"),
        "speaker_reference": config.get("speaker_reference"),
        "dtype": config.get("dtype"),
        "inference_pad_boost": config.get("inference_pad_boost"),
        "inference_bos_boost": config.get("inference_bos_boost"),
        "inference_eos_boost": config.get("inference_eos_boost"),
        "inference_user_pad_boost": config.get("inference_user_pad_boost"),
        "inference_user_bos_boost": config.get("inference_user_bos_boost"),
        "inference_user_eos_boost": config.get("inference_user_eos_boost"),
        "force_turn_taking_threshold": config.get("force_turn_taking_threshold"),
        "force_turn_taking_pad_window": config.get("force_turn_taking_pad_window"),
        "hf_home": config.get("hf_home"),
        "torch_home": config.get("torch_home"),
        "max_samples": config.get("max_samples"),
    }
    for key, value in optional_values.items():
        if value is not None:
            cmd_parts.extend([f"--{key}", _q(value)])

    for flag in ("force_turn_taking", "disable_rnnt_decoder_cuda_graphs"):
        if config.get(flag, False):
            cmd_parts.append(f"--{flag}")

    if config.get("inference_args"):
        cmd_parts.append(config["inference_args"])

    env_prefix = _base_env_prefix(config)
    command = " ".join(cmd_parts)
    return f"{env_prefix} && {command}" if env_prefix else command


def build_drirf_incremental_inference_command(config: dict) -> str:
    python_exec = config.get("inference_container_python_exec", "python3")
    script = config.get("drirf_incremental_inference_script_path", _DRIRF_INCREMENTAL_SCRIPT)
    output_dir = Path(config["output_dir"]) / "eval-results"

    cmd_parts = [
        python_exec,
        _q(script),
        "--model_path",
        _q(config["model"]),
        "--shar_input_dir",
        _q(config["shar_input_dir"]),
        "--dataset_name",
        _q(config["dataset_name"]),
        "--output_dir",
        _q(output_dir),
        "--num_frames_per_inference",
        str(config.get("num_frames_per_inference", 3)),
        "--buffer_size_frames",
        str(config.get("buffer_size_frames", 21)),
        "--codec_token_history_size",
        str(config.get("codec_token_history_size", 60)),
        "--engine_type",
        _q(config.get("engine_type", "vllm_llm_vllm_eartts")),
        "--vllm_gpu_memory_utilization",
        str(config.get("vllm_gpu_memory_utilization", 0.35)),
        "--vllm_max_model_len",
        str(config.get("vllm_max_model_len", 8192)),
        "--matmul_precision",
        _q(config.get("matmul_precision", "medium")),
        "--silence_padding_sec",
        str(config.get("silence_padding_sec", 0)),
        "--max_new_tokens",
        str(config.get("max_new_tokens", 4500)),
        "--temperature",
        str(config.get("temperature", 0.0)),
        "--top_p",
        str(config.get("top_p", 1.0)),
        "--repetition_penalty",
        str(config.get("repetition_penalty", 1.0)),
    ]

    optional_values = {
        "llm_checkpoint_path": config.get("llm_checkpoint_path") or config.get("model"),
        "tts_checkpoint_path": config.get("tts_checkpoint_path"),
        "speaker_reference": config.get("speaker_reference"),
        "system_prompt": config.get("system_prompt"),
        "tts_system_prompt": config.get("tts_system_prompt"),
        "inference_pad_boost": config.get("inference_pad_boost"),
        "inference_bos_boost": config.get("inference_bos_boost"),
        "inference_eos_boost": config.get("inference_eos_boost"),
        "inference_user_pad_boost": config.get("inference_user_pad_boost"),
        "inference_user_bos_boost": config.get("inference_user_bos_boost"),
        "inference_user_eos_boost": config.get("inference_user_eos_boost"),
        "force_turn_taking_threshold": config.get("force_turn_taking_threshold"),
        "force_turn_taking_pad_window": config.get("force_turn_taking_pad_window"),
        "pad_to_duration_secs": config.get("pad_to_duration_secs"),
        "max_samples": config.get("max_samples"),
        "inference_guidance_scale": config.get("inference_guidance_scale"),
        "inference_top_p_or_k": config.get("inference_top_p_or_k"),
        "inference_noise_scale": config.get("inference_noise_scale"),
        "tts_sliding_window": config.get("tts_sliding_window"),
    }
    for key, value in optional_values.items():
        if value is not None:
            cmd_parts.extend([f"--{key}", _q(value)])

    if config.get("use_codec_cache", True):
        cmd_parts.append("--use_codec_cache")
    else:
        cmd_parts.append("--no_use_codec_cache")

    for flag in (
        "force_turn_taking",
        "use_perception_cache",
        "use_perception_cudagraph",
        "disable_rnnt_decoder_cuda_graphs",
        "output_frame_alignment",
        "no_save_session_artifacts",
        "no_inference_guidance_enabled",
    ):
        if config.get(flag, False):
            cmd_parts.append(f"--{flag}")

    env_prefix = _base_env_prefix(config)
    command = " ".join(cmd_parts)
    return f"{env_prefix} && {command}" if env_prefix else command


def build_validation_logs_scoring_command(config: dict) -> str:
    scoring_nemo_code_path = config.get("scoring_nemo_code_path") or config["nemo_code_path"]
    eval_script = f"{scoring_nemo_code_path}/scripts/speech_eval/eval_conversation_behavior.py"
    python_exec = config.get("scoring_container_python_exec") or config.get("inference_container_python_exec") or "python3"
    cmd = (
        f"{python_exec} nemo_skills/dataset/conv_behav/scripts/run_scoring.py"
        f" --output_dir {_q(Path(config['output_dir']) / 'eval-results')}"
        f" --shar_input_dir {_q(config['shar_input_dir'])}"
        f" --dataset_name {_q(config['dataset_name'])}"
        f" --eval_script_path {_q(eval_script)}"
        f" --barge_in_threshold_sec {config.get('barge_in_threshold_sec', 1.5)}"
        f" --tt_latency_threshold_sec {config.get('tt_latency_threshold_sec', 1.5)}"
        f" --tt_precision_buffer_sec {config.get('tt_precision_buffer_sec', 1.0)}"
        f" --tt_recall_buffer_sec {config.get('tt_recall_buffer_sec', 20.0)}"
        f" --vad_min_silence_duration_ms {config.get('vad_min_silence_duration_ms', 2000)}"
    )
    if config.get("scoring_force", False):
        cmd += " --force"
    if config.get("torch_home"):
        cmd += f" --torch_home {_q(config['torch_home'])}"
    env_prefix = _base_env_prefix(config, code_path_key="scoring_nemo_code_path")
    return f"{env_prefix} && {cmd}" if env_prefix else cmd


def build_nemo_eval_scoring_command(config: dict, benchmark: str) -> str:
    eval_results_dir = Path(config["output_dir"]) / "eval-results" / benchmark
    script_path = "nemo_skills/dataset/s2s_demo/scripts/eval_conversation_behavior_v2.py"
    python_exec = config.get("scoring_container_python_exec") or config.get("inference_container_python_exec") or "python3"

    cmd_args = [
        f"{python_exec} {script_path}",
        f"--results_dir {_q(eval_results_dir)}",
        f"--barge_in_threshold_sec {_config_get(config, 'scoring', 'barge_in_threshold_sec', 1.5)}",
        f"--tt_latency_threshold_sec {_config_get(config, 'scoring', 'tt_latency_threshold_sec', 1.5)}",
        f"--tt_precision_buffer_sec {_config_get(config, 'scoring', 'tt_precision_buffer_sec', 1.0)}",
        f"--tt_recall_buffer_sec {_config_get(config, 'scoring', 'tt_recall_buffer_sec', 20.0)}",
        f"--vad_min_silence_duration_ms {_config_get(config, 'scoring', 'vad_min_silence_duration_ms', 2000)}",
        f"--segment_buffer_sec {_config_get(config, 'scoring', 'segment_buffer_sec', 0.5)}",
        "--output_file metrics.json",
    ]
    if _config_get(config, "scoring", "verbose", True):
        cmd_args.append("--verbose")
    if _config_get(config, "scoring", "disable_transcription", False):
        cmd_args.append("--disable_transcription")
    if _config_get(config, "scoring", "save_per_sample_results", True):
        cmd_args.append("--save_per_sample_results")
    if config.get("scoring_force", False) or _config_get(config, "scoring", "force_recompute", False):
        cmd_args.append("--force_recompute")

    env_prefix = _base_env_prefix(config, code_path_key="scoring_nemo_code_path")
    command = " ".join(cmd_args)
    return f"{env_prefix} && {command}" if env_prefix else command


def run_validation_logs_inference_stage(config: dict, expname: str, dry_run: bool, command_builder, label: str) -> bool:
    output_json = Path(config["output_dir"]) / "eval-results" / "validation_logs" / "metadatas" / f"{config['dataset_name']}.json"
    done_marker = output_json.with_suffix(output_json.suffix + ".done")
    if (
        output_json.exists()
        and output_json.stat().st_size > 0
        and done_marker.exists()
        and not config.get("scoring_force", False)
    ):
        print(f"\n--- Stage 1: Skipping inference (found {output_json}) ---")
        return False
    if output_json.exists() and output_json.stat().st_size == 0:
        print(f"\n--- Stage 1: Re-running inference (found empty metadata file {output_json}) ---")
    elif output_json.exists() and not done_marker.exists():
        print(f"\n--- Stage 1: Re-running inference (metadata exists without done marker {done_marker}) ---")
    if not dry_run:
        done_marker.unlink(missing_ok=True)

    print(f"\n--- Stage 1: Running inference ({label}) ---")
    num_gpus = config.get("num_gpus", 1)
    log_dir = str(Path(config["output_dir"]) / "eval-results" / "summarized-results")
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=command_builder(config),
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


def run_validation_logs_scoring_stage(config: dict, expname: str, run_after, dry_run: bool, force: bool = False):
    print("\n--- Stage 2: Running conv_behav validation_logs scoring ---")
    scoring_config = dict(config)
    if force:
        scoring_config["scoring_force"] = True
    log_dir = str(Path(config["output_dir"]) / "eval-results" / "summarized-results")
    cluster = config["cluster"]
    if run_after and isinstance(cluster, dict):
        cluster = dict(cluster)
        cluster["dependency_type"] = "afterok"
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=cluster,
        command=build_validation_logs_scoring_command(scoring_config),
        container=config.get("scoring_container") or config.get("inference_container"),
        num_gpus=config.get("scoring_gpus", 1),
        partition=config.get("scoring_partition") or config.get("partition"),
        run_after=run_after,
        expname=f"{expname}_score",
        installation_command=config.get("scoring_installation_command"),
        log_dir=log_dir,
        reuse_code=False,
        dry_run=dry_run,
    )


def run_validation_logs_pipeline(config: dict, mode: str) -> None:
    inference_only = config.get("inference_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)
    expname = config.get("expname", "conv_behav")

    if mode == "drirf_incremental":
        command_builder = build_drirf_incremental_inference_command
        label = "DRIRF incremental s2s_incremental_v2 adapter"
    elif mode == "dsfts_offline":
        command_builder = build_dsfts_offline_inference_command
        label = "DSFTS offline Hydra Lightning validation"
    else:
        command_builder = build_drirf_offline_inference_command
        label = "DRIRF offline wrapper-backed Lightning validation"

    infer_submitted = False
    if not scoring_only:
        infer_submitted = run_validation_logs_inference_stage(config, expname, dry_run, command_builder, label)
        if inference_only:
            return

    run_after = [expname] if infer_submitted else None
    run_validation_logs_scoring_stage(config, expname, run_after, dry_run, force=infer_submitted)


def _nemo_eval_generation_complete(eval_results_path: Path, num_chunks: int) -> bool:
    output_jsonl = eval_results_path / "output.jsonl"
    if num_chunks > 1:
        chunk_done_ok = all((eval_results_path / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
    else:
        chunk_done_ok = (eval_results_path / "output_chunk_0.jsonl.done").exists() or (
            eval_results_path / "output.jsonl.done"
        ).exists()
    return output_jsonl.exists() and chunk_done_ok


def run_nemo_eval_pipeline(config: dict) -> None:
    benchmark = _benchmark_name(config)
    expname = config.get("expname", "conv_behav")
    dry_run = config.get("dry_run", False)
    generation_only = config.get("generation_only", config.get("inference_only", False))
    scoring_only = config.get("scoring_only", False)
    eval_results_path = Path(config["output_dir"]) / "eval-results" / benchmark
    generation_submitted = False

    print(f"Benchmark:  {benchmark}")
    if not scoring_only:
        prepare_nemo_eval_data(config)
        num_chunks = int(config.get("num_chunks", 1) or 1)
        if _nemo_eval_generation_complete(eval_results_path, num_chunks) and not config.get("scoring_force", False):
            print(f"\n--- Stage 1: Skipping generation (found {eval_results_path / 'output.jsonl'}) ---")
        else:
            print("\n--- Stage 1: Running nemo_eval generation ---")
            extra_args = ["++eval_type=null"]
            if config.get("max_samples"):
                extra_args.append(f"++max_samples={config['max_samples']}")
            if config.get("server_server_type"):
                extra_args.append(f"++server.server_type={config['server_server_type']}")
            if config.get("api_key_env_var"):
                extra_args.append(f"++server.api_key_env_var={config['api_key_env_var']}")
            if config.get("inference_overrides"):
                extra_args.extend(config["inference_overrides"].strip().split())
            server_gpus = config.get("server_gpus", config.get("num_gpus", 1))
            partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
            nemo_eval(
                ctx=wrap_arguments(" ".join(extra_args)),
                cluster=config["cluster"],
                output_dir=config["output_dir"],
                benchmarks=benchmark,
                model=config["model"],
                server_type=config.get("server_type", "vllm"),
                server_gpus=server_gpus,
                server_nodes=config.get("server_nodes", config.get("num_nodes", 1)),
                server_address=config.get("server_address"),
                num_chunks=config.get("num_chunks", 1),
                server_container=config.get("server_container") or config.get("inference_container"),
                server_entrypoint=config.get("server_entrypoint"),
                data_dir=config.get("data_dir"),
                server_args=config.get("server_args", ""),
                installation_command=config.get("installation_command"),
                partition=partition,
                expname=expname,
                auto_summarize_results=False,
                reuse_code=False,
                rerun_done=config.get("scoring_force", False),
                dry_run=dry_run,
            )
            generation_submitted = True

    if generation_only:
        return

    score_run_after = maybe_merge_before_scoring(
        config,
        str(eval_results_path),
        expname,
        run_after=[expname] if generation_submitted else None,
        dry_run=dry_run,
    )
    print("\n--- Stage 2: Running nemo_eval conversation behavior scoring ---")
    run_cmd(
        ctx=wrap_arguments(""),
        cluster=config["cluster"],
        command=build_nemo_eval_scoring_command(config, benchmark),
        container=config.get("scoring_container") or config.get("server_container") or config.get("inference_container"),
        num_gpus=config.get("scoring_gpus", 1),
        partition=config.get("scoring_partition") or config.get("partition"),
        run_after=score_run_after,
        expname=f"{expname}_score",
        installation_command=config.get("scoring_installation_command"),
        log_dir=str(eval_results_path / "summarized-results"),
        reuse_code=False,
        dry_run=dry_run,
    )


def run_conv_behav_eval(config: dict):
    mode = config.get("pipeline_mode", "drirf_offline")
    valid_modes = {"dsfts_offline", "drirf_offline", "drirf_incremental", "nemo_eval"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown conv_behav pipeline_mode={mode!r}; expected one of {sorted(valid_modes)}")

    print(f"Pipeline:   {mode}")
    print(f"Dataset:    {config['dataset_name']}")
    print(f"Shar dir:   {config['shar_input_dir']}")
    print(f"Model:      {config['model']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"Infer src:  {config['nemo_code_path']}")
    print(f"Score src:  {config.get('scoring_nemo_code_path') or config['nemo_code_path']}")
    print(f"Infer ctr:  {config.get('inference_container') or config.get('server_container')}")
    print(f"Score ctr:  {config.get('scoring_container') or config.get('inference_container')}")
    print(f"Infer py:   {config.get('inference_container_python_exec') or 'python3'}")
    print(
        "Score py:   "
        f"{config.get('scoring_container_python_exec') or config.get('inference_container_python_exec') or 'python3'}"
    )

    if mode == "nemo_eval":
        run_nemo_eval_pipeline(config)
    else:
        run_validation_logs_pipeline(config, mode)


def main():
    parser = argparse.ArgumentParser(description="Run conv_behav evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--shar_input_dir", help="Override shar input directory")
    parser.add_argument("--dataset_name", help="Override dataset name")
    parser.add_argument("--benchmark", help="Override nemo_eval benchmark name, e.g. conv_behav.team_20251124")
    parser.add_argument("--pipeline_mode", choices=["dsfts_offline", "drirf_offline", "drirf_incremental", "nemo_eval"])
    parser.add_argument("--nemo_code_path", help="Override NeMo path for inference")
    parser.add_argument("--scoring_nemo_code_path", help="Override NeMo path for conv_behav scoring script")
    parser.add_argument("--pretrained_llm", help="Override LLM backbone HF model ID or path")
    parser.add_argument("--dsfts_offline_inference_script_path", help="Override DSFTS offline Hydra inference script path")
    parser.add_argument("--drirf_offline_inference_script_path", help="Override DRIRF offline adapter path")
    parser.add_argument("--drirf_incremental_inference_script_path", help="Override DRIRF incremental adapter path")
    parser.add_argument("--inference_container", help="Override container used for inference")
    parser.add_argument("--server_container", help="Override container used for nemo_eval server")
    parser.add_argument("--scoring_container", help="Override container used for scoring")
    parser.add_argument("--inference_container_python_exec", help="Python executable used for inference container")
    parser.add_argument("--scoring_container_python_exec", help="Python executable used for scoring container")
    parser.add_argument("--num_chunks", type=int, help="Override nemo_eval chunk count")
    parser.add_argument("--max_samples", type=int, help="Limit nemo_eval samples")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--generation_only", action="store_true")
    parser.add_argument("--scoring_only", action="store_true")
    parser.add_argument("--scoring_force", action="store_true", help="Re-run inference/generation and scoring")
    args = parser.parse_args()

    config = load_config(args.config)

    override_keys = [
        "model",
        "output_dir",
        "shar_input_dir",
        "dataset_name",
        "benchmark",
        "pipeline_mode",
        "nemo_code_path",
        "scoring_nemo_code_path",
        "pretrained_llm",
        "dsfts_offline_inference_script_path",
        "drirf_offline_inference_script_path",
        "drirf_incremental_inference_script_path",
        "inference_container",
        "server_container",
        "scoring_container",
        "inference_container_python_exec",
        "scoring_container_python_exec",
        "num_chunks",
        "max_samples",
    ]
    for key in override_keys:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.inference_only:
        config["inference_only"] = True
    if args.generation_only:
        config["generation_only"] = True
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
