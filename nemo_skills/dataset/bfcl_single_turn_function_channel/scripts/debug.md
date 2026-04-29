# BFCL Interactive Debug Setup

Use this when you need to debug the server + inference client interactively
instead of submitting via `run_eval.py`.

## Bug fixed (2026-04-29)

`del output` in `s2s_incremental_backend_v2.py` was placed before
`_decode_function_channel(output)`, causing a `NameError` on every request
when `--decode_function_channel` is enabled. Result: all `generation` fields
in `output.jsonl` were empty and acc was 0. Fixed by moving the decode call
before the `del`.

## Setup commands

```bash
# ── 1. Paths (adjust SKILLS_FC_PATH to your checkout) ────────────────────────
export SKILLS_FC_PATH=/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/Skills-FC
export NEMO_PATH=/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/NeMo/NeMo_Voice_Chat_FC

# ── 2. Symlink Skills-FC repo to /nemo_run/code ───────────────────────────────
mkdir -p /nemo_run
ln -s $SKILLS_FC_PATH /nemo_run/code

# ── 3. Environment ────────────────────────────────────────────────────────────
export PYTHONPATH=/nemo_run/code:$NEMO_PATH:$PYTHONPATH
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yuanhangs/.cache/huggingface

# ── 4. Pip installs (patches the container) ───────────────────────────────────
pip install hf-xet==1.1.9 huggingface-hub==0.34.4 nvidia-modelopt==0.33.1 \
  nvidia-modelopt-core==0.33.1 tokenizers==0.22.0 transformers==4.56.0 \
  lhotse==1.32.2 nv-one-logger-core==2.1.0 \
  nv-one-logger-pytorch-lightning-integration==2.1.0 \
  nv-one-logger-training-telemetry==2.1.0 kaldialign==0.9.1

# ── 5. Installation command (copies vllm model file) ─────────────────────────
mkdir -p /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models
cp /nemo_run/code/asset/nemotron_h.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nemotron_h.py
ln -sf $(which python3) /usr/local/bin/python

# ── 6. Start server (sampling config) ────────────────────────────────────────
cd /nemo_run/code
nohup python -m nemo_skills.inference.server.serve_unified \
  --model /lustre/fsw/portfolios/llmservice/users/vtrinh/projects/function_calling_share/pretrained_s2s_rnnt_ckpt/Apr_02_2026/e70/e70-step28008-tts-eartts-34014_asr_cand3_0.6b-PK_ep0_01988_300 \
  --port 8000 \
  --backend s2s_incremental_v2 \
  --speaker_reference /lustre/fsw/portfolios/convai/users/ecasanova/Checkpoints/Mg_a_00759.wav \
  --num_frames_per_inference 3 \
  --engine_type vllm_llm_vllm_eartts \
  --use_perception_cache \
  --use_perception_cudagraph \
  --use_codec_cache \
  --buffer_size_frames 21 \
  --codec_token_history_size 60 \
  --matmul_precision medium \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_max_model_len 8192 \
  --inference_pad_boost 0 \
  --inference_bos_boost 0 \
  --inference_eos_boost 0 \
  --inference_user_pad_boost 0 \
  --inference_user_bos_boost 0 \
  --inference_user_eos_boost 0 \
  --inference_guidance_scale 0.2 \
  --inference_top_p_or_k 0.95 \
  --inference_noise_scale 0.001 \
  --tts_sliding_window 7500 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --temperature 0.8 \
  --decode_function_channel \
  --tool_call_parser /nemo_run/code/recipes/multimodal/server/tool_calling/nemotron_v2_voicechat_toolcall_parser.py \
  --use_function_channel_for_tool_calls \
  --output_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat_sampling/bfcl_fc_artifacts \
  --no_save_session_artifacts \
  --batch_size 1 > /tmp/server.log 2>&1 &

# ── 7. Watch server logs (separate terminal) ─────────────────────────────────
# tail -f /tmp/server.log

# ── 8. Run inference client (unbuffered output) ───────────────────────────────
python -u /nemo_run/code/nemo_skills/dataset/bfcl_single_turn_function_channel/scripts/run_bfcl_fc_inference.py \
  --server_url http://localhost:8000 \
  --input_jsonl /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/bfcl_fc_data/simple/input.jsonl \
  --output_jsonl /tmp/debug_output.jsonl \
  --max_workers 1 \
  --poll_interval 10 \
  --max_poll_attempts 60
```

## Notes

- Container: `/lustre/fsw/portfolios/llmservice/users/erastorgueva/code/containers/triton25.05_s2svllm26.02.12.sqsh`
- `SKILLS_FC_PATH` is the only variable that needs updating per user
- `/nemo_run/code` must point to the Skills-FC repo (not NeMo source)
- NeMo source (`NeMo_Voice_Chat_FC`) is NOT pre-installed in the container — must be on `PYTHONPATH`
- The container's lhotse version is wrong; `pip install lhotse==1.32.2` fixes it
- `--batch_size 1` is used here instead of 2 for easier debugging
- `HF_HOME` is not required for the server (model loads from Lustre) but avoids cache writes to `~/.cache`
