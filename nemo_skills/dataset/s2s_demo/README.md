# End2end Evaluation of the Duplex Speech2Speech Model Based on Nemo-Skills

The suggested recipe performs inference with a duplex s2s model including TTS. It's based on the incremental decoding scripts. Currently the [demo_20251124](demo_20251124/) test set and Voicebench are supported.

For demo the recipe runs scoring based on Kevin's script which includes:
- Turn-taking evaluation.
- User ASR quality evaluation.
- Agent TTS quality evaluation.
- Special symbol balance.
- Agent content quality evaluation (LLM based).

For Voicebench it runs original Voicebench evaluation. The only change made there is support for nvidia inference API endpoints.

In addition to average metrics the scoring recipe saves all kinds of alignments and error scores for each sample.

## Getting started

Currently different scripts are used to run demo and Voicebench. The configs are similar but not identical. Of course those should be unified going forward.

1. Go to https://inference.nvidia.com/ to get API key.
2. Clone this branch.
```bash
git clone git@github.com:NVIDIA-NeMo/Skills.git nemo-skills
git checkout vmendelev/2512_s2s_eval
```
3. Create a `.venv` and install nemo-skills:

```bash
cd /path/to/nemo-skills
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

4. Decide which cluster you want to work on and setup the corresponding cluster configuration. The example configuration for draco_oci (oci_iad) is provided [here](../../../cluster_configs/). You can get more configurations [here](https://gitlab-master.nvidia.com/igitman/nemo-skills-configs/-/tree/main/cluster_configs/v0.7.1?ref_type=heads). Don't forget to update user name, folder names and add `/lustre` to the mounts list.

### Dataset preparation
#### Option 1
Just use the data directory which is already present on draco_oci or copy it to another cluster if necessary. `/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir`

#### Option 2
You can prepare the datasets again. The instructions are present later in the document.

### Running the tests

5. Look into the [demo config](scripts/s2s_demo_eval_config.yaml) or [voicebench config](../voicebench/scripts/voicebench_s2s_session_full_config.yaml) and make sure that the required artifacts are in the specified places.
6. Check the `data_dir` parameter. It should point to a folder on the cluster with `s2s_demo` and `voicebench` folders from `nemo_skills/datasets`. Demo wav files should be in `s2s_demo/demo_20251124/data`. On draco everything is in `/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir`
7. Adjust `output_dir`.
8. Set the `num_chunks` to be e.g. 8 to make the thing run faster. This is applied to each sub-benchmark. For demo 8 is enough, for voicebench I would recommend 32 or more because of big test sets like MMSU. In this case runtime will be about 3 hrs.
9. Set `max_samples` to e.g. 2 if you want a fast run.
10. Make sure that `$NVIDIA_API_KEY` is set to a correct value.
11. Run the below command:

```bash
# DEMO
cd /path/to/nemo-skills && \
source .venv/bin/activate && \
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/s2s_demo/scripts/run_s2s_demo_eval.py \
  --config nemo_skills/dataset/s2s_demo/scripts/s2s_demo_eval_config.yaml
```

or

```bash
# VOICEBENCH
cd /path/to/nemo-skills && \
source .venv/bin/activate && \
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python nemo_skills/dataset/voicebench/scripts/generate_from_api_and_score_official.py --config nemo_skills/dataset/voicebench/scripts/voicebench_s2s_session_full_config.yaml
```

## Running the Server Only (for Manual Testing)

If you want to run just the unified server for manual testing or integration with external clients (e.g., AU-Harness), you can start it directly without installing nemo-skills. Just copy the code folder to the cluster.

**On draco_oci:**

### Option 1: Text-only mode (no TTS)

Create `run_server.sbatch`:
```bash
#!/bin/bash
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=convai_convaird_nemo-speech
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=server_%j.log

srun --container-image=/lustre/fsw/portfolios/convai/users/ecasanova/docker_images/nemo_duplex_november_eartts.sqsh \
     --container-mounts="/lustre:/lustre,/path/to/your/workspace:/workspace" \
     bash -c 'cd /workspace && \
     export PYTHONPATH="/workspace/ns_eval:$PYTHONPATH" && \
     export HF_HOME=/lustre/fsw/portfolios/llmservice/users/YOUR_USER/.cache/huggingface && \
     export INCLUDE_DEBUG_INFO=true && \
     python -m nemo_skills.inference.server.serve_unified \
       --backend s2s_session \
       --model /lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-2mim_sw_et_eos_dp_eos_dup_fp32-stt-3-december_stt_edresson_model_R_digits_norm_eip_0.1_EA_model_step_9005 \
       --config_path /lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/configs/inference/nanov2_demo_model_eartts_updated.yaml \
       --code_path /lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased \
       --ignore_system_prompt \
       --num_frames_per_inference 2 \
       --silence_padding_sec 0.0 \
       --session_artifacts_dir /lustre/fsw/portfolios/llmservice/users/YOUR_USER/tmp/s2s_artifacts \
       --no_decode_audio \
       --response_end_detection_mode eos \
       --eos_detection_window 10 \
       --port 8000'
```

### Option 2: With audio output (TTS enabled)

Create `run_server_sound.sbatch`:
```bash
#!/bin/bash
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=convai_convaird_nemo-speech
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=server_%j.log

srun --container-image=/lustre/fsw/portfolios/convai/users/ecasanova/docker_images/nemo_duplex_november_eartts.sqsh \
     --container-mounts="/lustre:/lustre,/path/to/your/workspace:/workspace" \
     bash -c 'cd /workspace && \
     export PYTHONPATH="/workspace/ns_eval:$PYTHONPATH" && \
     export HF_HOME=/lustre/fsw/portfolios/llmservice/users/YOUR_USER/.cache/huggingface && \
     export INCLUDE_DEBUG_INFO=true && \
     python -m nemo_skills.inference.server.serve_unified \
       --backend s2s_session \
       --model /lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Nemotron-VoiceChat-november/duplex-eartts-2mim_sw_et_eos_dp_eos_dup_fp32-stt-3-december_stt_edresson_model_R_digits_norm_eip_0.1_EA_model_step_9005 \
       --config_path /lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/scripts/configs/inference/nanov2_demo_model_eartts_updated.yaml \
       --code_path /lustre/fsw/portfolios/convai/users/kevinhu/S2S-Duplex-new-codebase/branches/NeMo-release_not_rebased \
       --ignore_system_prompt \
       --num_frames_per_inference 2 \
       --silence_padding_sec 0.0 \
       --session_artifacts_dir /lustre/fsw/portfolios/llmservice/users/YOUR_USER/tmp/s2s_artifacts \
       --port 8001'
```

Submit with `sbatch run_server.sbatch` or `sbatch run_server_sound.sbatch`.

**Notes:**
- Replace `YOUR_USER` with your username and `/path/to/your/workspace` with your workspace path containing the code
- `--session_artifacts_dir` - where session audio artifacts are saved
- `INCLUDE_DEBUG_INFO=true` - includes debug info (with ASR transcription) in responses; set to `false` to disable
- Text-only mode uses `--no_decode_audio`, `--response_end_detection_mode eos` (stops after consecutive PAD tokens)
- Audio mode uses default energy-based response end detection (TTS silence)
- The server exposes OpenAI-compatible `/v1/chat/completions` endpoint
- The server supports multi-turn conversations with automatic session management based on conversation history hashing

Check `server_<jobid>.log` for server output. Once running, the log will show which node it's on (e.g., `batch-block1-3196:8000`). You can then send requests from the login node or any machine that can reach the compute node.

## Incremental and Session Backends

The demo test is run with incremental backend which assumes silences already in the test audio or you can use `silence_padding_sec` to add trailing pause automatically.

**Session backend.** This backend is used for Voicebench. The difference from the incremental one is that here we feed the user turn (no added pause), wait for the model to respond, then feed the second one and so on. We can also feed 2 turns in one go and preprogram feeding the second one at a predefined point while the system is responding to the first turn instead of 0s. With this one can enable a dialog between e.g Gemini and our model.

### NeMo Integration

The incremental backend interfaces with NeMo's `speechlm2` collection. The main model class is `NemotronVoiceChat` from `nemo.collections.speechlm2.models.nemotron_voicechat`. The frame-by-frame inference uses the following key interfaces:

1. **Perception** (`stt_model.perception`) - Encodes raw audio waveform into frame-level embeddings. Called once per inference step with a sliding window audio buffer.

2. **LLM Forward** (`stt_model.__call__`) - Generates agent response and ASR tokens from audio embeddings. Supports two modes:
   - `DynamicCache` mode (for most models) - uses HuggingFace's dynamic KV cache
   - `input_embeds_history` mode (for Mamba models) - accumulates all input embeddings

3. **TTS Code Generation** (`tts_model.infer_codes_one_step`) - Generates audio codec tokens frame-by-frame from text tokens. Maintains its own `past_key_values` cache.

4. **Audio Codec Decoding** (`tts_model.audio_codec.decode`) - Converts codec tokens to waveform audio. Called with a sliding window of codec tokens.

The backend implementation is in `recipes/multimodal/server/backends/s2s_incremental_backend.py`. The session backend (`s2s_session_backend.py`) extends the incremental backend to persist state (LLM cache, audio buffers, frame index) across multiple turns.

This should be based on the Niva code going forward.

In case you want to use a different nemo branch -- just replace the path to nemo code in the config.

## Preparing Tests

Standard nemo-skills `test.jsonl` and corresponding set of wav files are what we need per benchmark. The current demo test set can be found at `/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_test/data_dir/s2s_demo` and it was obtained from the available lhotse shars via this [script](convert_lhotse_to_eval.py). If you want to add more samples you need to add them to the `test.jsonl` and copy wav to the data folder. Or you can use the [script](convert_lhotse_to_eval.py) to convert another dataset. No reference transcription or segmentation is needed.

With Voicebench you can use the standard nemo-skills preparation command:

```bash
cd /path/to/nemo-skills && \
source .venv/bin/activate && \
python -m nemo_skills.dataset.prepare voicebench
```

### VoiceBench Scoring Setup

VoiceBench scoring uses the GPT judge (`api_judge.py`) for certain subtests. The original VoiceBench only supports OpenAI API, so modifications are needed for NVIDIA API support.

**On draco_oci:** A pre-modified VoiceBench repository is already available. Just set `voicebench_repo_path` in your config to:
```
/lustre/fsw/portfolios/llmservice/users/vmendelev/code/VoiceBench
```

The key modified file is `api_judge.py` (full path: `/lustre/fsw/portfolios/llmservice/users/vmendelev/code/VoiceBench/api_judge.py`) which adds:
- `--api_type nvidia` argument to use NVIDIA inference API
- `--nvidia_model` argument to specify the model (e.g., `meta/llama-3.1-70b-instruct`)
- Initializes OpenAI client with `https://inference-api.nvidia.com/v1` base URL

**On other clusters:** Clone VoiceBench, copy the modified `api_judge.py` from draco, and update `voicebench_repo_path` in your config.

## Comparing Multiple Evaluation Results

Use the `compare_eval_results.py` script to compare metrics from multiple model evaluations and generate a Markdown report:

```bash
cd /path/to/nemo-skills  && \
source .venv/bin/activate && \
python nemo_skills/dataset/s2s_demo/scripts/compare_eval_results.py \
    --eval_folders \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/vmendelev/tmp/s2s_demo_eval_t4:November Model" \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_s2s_session_full/runs/voicebench_20251230_041530:November Model" \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/vmendelev/tmp/s2s_demo_eval_t4_c3:December Model" \
        "draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/vmendelev/experiments/voicebench_s2s_session_full/runs/voicebench_20251229_113456:December Model" \
    --output /tmp/comparison_report.md
```

The script supports:
- **Remote folders via SSH**: `hostname:/path/to/folder:DisplayName`
- **Local folders**: `/path/to/folder:DisplayName`
- **Mixed**: compare models from different clusters or local/remote

See [example report](scripts/comparison_report.md) for output format.

## TODOs
0. Refactor the backends to directly call Niva.
1. Integrate vLLM and Triton.
2. Add WandB integration.
4. Refactor configs and scripts used to run the tests.
5. Refactor decmo scoding script.
6. Add batching support.
