# Megatron Duplex checkpoint → NeMo Skills hybrid inference

These scripts export a `DuplexALMModel` Torch-DCP checkpoint into the split
layout consumed by the DRIRF-backed `s2s_incremental_v2` server:

```text
<export>/
├── config.json                 # Native VoiceChat frontend configuration
├── model.safetensors           # Perception/projector/fusion/embeddings/heads
├── export_manifest.json
└── vllm_llm/
    ├── config.json             # combined_embeds + custom output declarations
    ├── model.safetensors       # Nemotron trunk + agent/function heads
    └── tokenizer/code assets
```

The exporter reads selected model tensors directly from Torch DCP. It does not
restore optimizer state and does not instantiate the training model.

## Phase 1: export iteration 2400

Run in the Megatron training environment, from the NeMo Skills repository root.
The two templates below are the original NeMo VoiceChat checkpoint and the HF
Nemotron directory used to warm-start this training run.

```bash
CKPT=/lustre/fsw/portfolios/llmservice/users/nsrihari/full_duplex/avlm/init/e001/results/voc_2/output/voicechat_nano_v2_stage2_sft_from_nemo_pt99k/checkpoints
CONVERTED_ROOT=/lustre/fsw/portfolios/llmservice/users/nsrihari/full_duplex/avlm/init/e001/results/nemo_converted_megatron_voc_ckpts2/s2s_Jan_21_2026/result/PreT_PT0.95_SFT0.0_QA0.05_TEXT0.0_tloss0.0_MCQ0.0_ASR0.00_sysp0.0_se0.0_sm0.0_id0.0_its10.0_its20.0_wc0.0_fc0.0_fcv3_0.0_asr_dtc0_dst0_lt1.0_bos10.0_eos10.0_pad0.5_fcp0.3_fcs6.0_fce6.0_fcr3.0_fcc64.0/checkpoints/step-99015_hf
VOICECHAT_TEMPLATE=/lustre/fsw/portfolios/llmservice/users/vtrinh/projects/s2s_Jan_21_2026/result/PreT_PT0.95_SFT0.0_QA0.05_TEXT0.0_tloss0.0_MCQ0.0_ASR0.00_sysp0.0_se0.0_sm0.0_id0.0_its10.0_its20.0_wc0.0_fc0.0_fcv3_0.0_asr_dtc0_dst0_lt1.0_bos10.0_eos10.0_pad0.5_fcp0.3_fcs6.0_fce6.0_fcr3.0_fcc64.0/checkpoints/step-99015_hf
EXPORT=/lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/megatron_duplex_exports/voicechat_nano_v2_stage2_pt99k_iter2400

python scripts/megatron/duplex_export_hybrid_checkpoint.py \
  --checkpoint "$CKPT" \
  --iteration 2400 \
  --voicechat-template "$VOICECHAT_TEMPLATE" \
  --hf-llm-template "$CONVERTED_ROOT/llm_hf" \
  --output-dir "$EXPORT"
```

The template arguments are optional for this checkpoint family; their defaults
are the paths shown above. The export manifest also records the established
EAR-TTS checkpoint used by the current incremental benchmark configurations.

Do not use `--allow-unmapped` unless every reported tensor has been audited.
`--overwrite` replaces generated files but does not recursively delete the
output directory.

Validate structure and hashes:

```bash
python scripts/megatron/duplex_verify_hybrid_checkpoint.py validate-export \
  --export-dir "$EXPORT" \
  --verify-hashes
```

## Phase 2: native reference and end-to-end comparison

The Megatron repository already contains the native full-AR trace driver. Run
it in its training container on a short parity sample:

```bash
MEGATRON=/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/megatron-lm-omcat/megatron-lm-voicechat

cd "$MEGATRON"
python examples/multimodal/duplex/parity/trace_megatron.py \
  --config examples/multimodal/duplex/configs/voicechat_nano_v2_stage2_sft_from_nemo_pt99k.yaml \
  --ckpt "$CKPT" \
  --sample /tmp/parity_sample.pt \
  --out /tmp/megatron_native_trace.pt \
  --full-ar --all-stages --ar-frames 64 --precision bf16
```

For named tensor dumps produced by native and hybrid instrumentation:

```bash
cd /nemo_run/code
python scripts/megatron/duplex_verify_hybrid_checkpoint.py compare-tensors \
  --native /tmp/megatron_native_tensors.pt \
  --hybrid /tmp/drirf_hybrid_tensors.pt \
  --atol 0.02 --rtol 0.02 --min-cosine 0.999
```

For end-to-end JSONL outputs, compare deterministic text and function channels:

```bash
python scripts/megatron/duplex_verify_hybrid_checkpoint.py compare-jsonl \
  --native /tmp/native_output.jsonl \
  --hybrid /tmp/hybrid_output.jsonl \
  --fields pred_text function_channel_text
```

## Phase 3/4: BFCL hybrid smoke test

The runtime implementation remains DRIRF; NeMo Skills supplies the server/API
adapter and passes the preconverted engine path. The checkpoint has no RNNT or
ASR head, so keep forced turn-taking disabled and let the trained agent channel
emit its own BOS/EOS tokens.

`run_all_benchmarks.sh` detects Torch-DCP checkpoints passed through `--model`,
converts them once, caches the export under the user's shared Lustre workspace,
and substitutes the export transparently. Ordinary checkpoints do not enter
this path. The conversion Python must contain PyTorch and safetensors; set it
once in the environment when the launcher's lightweight virtualenv lacks them.

```bash
cd /nemo_run/code
export NEMO_SKILLS_MEGATRON_CONVERSION_PYTHON=/path/to/megatron-training-python

bash scripts/run_all_benchmarks.sh \
  --benchmarks bfcl \
  --model "$CKPT" \
  --output_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/megatron_duplex_eval/iter2400 \
  --dry_run
```

Inspect the dry-run command, then remove `--dry_run`. For the first real smoke
test, set `max_samples` in a copied BFCL config and pass it with
`--config_bfcl`; expand to the stock config only after parity succeeds.

## Phases 5/6

Use the same Megatron checkpoint as `--model` with any incremental benchmark or
the full suite. Audio-output benchmarks obtain EAR-TTS from the export manifest;
BFCL continues to use `--no_decode_audio` from its existing configuration.

```bash
bash scripts/run_all_benchmarks.sh \
  --model "$CKPT" \
  --output_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/megatron_duplex_eval/iter2400_full
```
