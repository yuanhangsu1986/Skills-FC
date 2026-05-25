# BigBench Audio (BBA)

Speech language model evaluation on reasoning tasks from [Big-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), delivered as spoken audio questions.

- **Source**: [ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)
- **Size**: 1,000 samples across 4 categories (250 each)
- **Audio**: MP3, 4–60 seconds, synthetic voices (OpenAI / Azure / AWS Polly)
- **Scoring**: exact-match accuracy per category and overall

## Categories

| Category | Answer type | Example |
|---|---|---|
| `formal_fallacies` | valid / invalid | "invalid" |
| `navigate` | Yes / No | "Yes" |
| `object_counting` | integer | "5" |
| `web_of_lies` | Yes / No | "No" |

## Step 1 — Prepare the dataset

Run once on the cluster to download audio from HuggingFace and write per-category JSONL files:

```bash
python nemo_skills/dataset/bba/prepare.py \
    --output_dir /lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/bba_data/nemo_skills/dataset/bba
```

This creates:
```
<output_dir>/
├── data/               # WAV audio files (bba_0.wav ... bba_999.wav)
├── formal_fallacies/test.jsonl
├── navigate/test.jsonl
├── object_counting/test.jsonl
└── web_of_lies/test.jsonl
```

Set `data_dir` in the config to the **parent** of `nemo_skills/` in that path, i.e.:
```
/lustre/fsw/portfolios/llmservice/users/yuanhangs/workspace/voice_chat/bba_data
```

## Step 2 — Run evaluation

Edit `scripts/bba_config_fc.yaml` to set your `output_dir` and `data_dir`, then:

```bash
python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
    --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml
```

Run generation only (no scoring):
```bash
python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
    --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml \
    --generation_only
```

Run scoring only on existing output:
```bash
python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
    --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml \
    --scoring_only
```

Run a single category:
```bash
python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
    --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml \
    --categories formal_fallacies
```

Dry run (print commands without submitting):
```bash
python nemo_skills/dataset/bba/scripts/run_bba_eval.py \
    --config nemo_skills/dataset/bba/scripts/bba_config_fc.yaml \
    --dry_run
```

## Output

Results are written to `<output_dir>/eval-results/bba.<category>/metrics.json`:

```json
{
  "bba.formal_fallacies": {"greedy": {"accuracy": 72.4, "correct": 181, "total": 250}},
  "bba.navigate":          {"greedy": {"accuracy": 68.0, "correct": 170, "total": 250}},
  "bba.object_counting":   {"greedy": {"accuracy": 55.2, "correct": 138, "total": 250}},
  "bba.web_of_lies":       {"greedy": {"accuracy": 61.6, "correct": 154, "total": 250}}
}
```

## Methodology — alignment with the official BBA spec

Primary reference: [Artificial Analysis BBA release blog](https://huggingface.co/blog/big-bench-audio-release).

Secondary reference (for items the blog leaves unspecified): the standalone implementation at
`/lustre/fsw/portfolios/llmservice/users/yuanhangs/codes/bigbench-audio-standalone`
(`benchmark.py` + `scoring.py` + `clients/openai_realtime_client.py`).

**Convention used in this section**: when the official blog pins down a value, that value is the spec. When the blog is silent, **we treat the standalone repo's choice as the canonical reference**, because it's the most-direct upstream implementation we have visibility into. The table below uses both — the "Reference value" column shows whichever is authoritative for that item.

This section tracks the choices we made and the known deviations from those references, so we can revisit them later if we want closer alignment with the official leaderboard numbers.

### What matches the official methodology

- **Judge prompt template**: verbatim copy from the blog (`run_bba_scoring.py:JUDGE_PROMPT`).
- **3 evaluation runs averaged**: `NUM_EVAL_RUNS = 3` in `run_bba_scoring.py`. Matches blog's *"All results presented represent averages across three independent evaluation runs on each dataset."*
- **Audio answer → text via Whisper before judging**: blog says *"For audio responses, we first transcribe them to text using OpenAI's Whisper API"*. We do this via the ASR stage (`pipeline/transcribe.py`).
- **`{question}` substituted with real question text**: we Whisper-ASR the input audio and pass it as `entry["question_asr"]`. The blog's prompt template implies this substitution; we honor it.

### Known discrepancies (to revisit)

| # | Area | Reference value | Source | Our implementation | Why it matters |
|---|---|---|---|---|---|
| 1 | **Whisper service** | OpenAI's hosted Whisper API (`https://api.openai.com/v1`, `model=whisper-1`, internally Whisper Large v2 + proprietary tuning) | Official blog | Self-hosted `openai/whisper-large-v3` via vLLM (`asr_model: openai/whisper-large-v3` in YAML, `--task transcription`) | Different model version (v3 vs v2-derived `whisper-1`) and different serving stack (vLLM vs OpenAI's hosted infra) → candidate transcripts can differ on edge cases (proper nouns, numbers, noisy audio). |
| 2 | **Judge model** | `claude-3-5-sonnet-20241022` (Claude 3.5 Sonnet, Oct '24), Anthropic-hosted | Official blog | `aws/anthropic/bedrock-claude-sonnet-4-6` (Claude 4 Sonnet) via NVIDIA inference API → AWS Bedrock | Different model generation entirely. Single largest deviation from the blog's pinned spec. |
| 3 | **Judge temperature** | `0.2` | Standalone (`scoring.py:check_answer_llm`) | `temperature=1.0` in `run_bba_scoring.py:llm_judge` | Higher temperature → noisier judgments per call. Combined with the 3-run average it stays defensible, but the standalone is much more deterministic. |
| 4 | **Judge `max_tokens`** | `256` | Standalone (`scoring.py:check_answer_llm`) | `max_tokens=10` | Sufficient in practice for `"CORRECT"` / `"INCORRECT"`, but technically different from the reference. |
| 5 | **Judge system prompt** | `"You are a helpful assistant"` | Standalone (`scoring.py:check_answer_llm`) | None — only a user message | Probably negligible for instruction-tuned Claude/GPT judges, but worth aligning for cleanliness. |
| 6 | **SUT generation temperature** | `0.6` (sampling) | Standalone (`clients/openai_realtime_client.py:TEMPERATURE`) | `--temperature 0.0` in YAML (greedy) | Deterministic but different from the published reference. If we want to match the variability of the leaderboard numbers, set to `0.6`. |
| 7 | **SUT system prompt** | `"You are a helpful assistant"` (via OpenAI Realtime `session.instructions`) | Standalone (`clients/openai_realtime_client.py:SYSTEM_PROMPT`) | No system prompt — `prepare.py` writes only a user message. | Possibly affects model behavior on instruction-tuned checkpoints. Add a system message in `prepare.py` if we want alignment with the standalone. |
| 8 | **Audio input padding** | **No padding** by default; standalone has optional `--include-2s-silence` / `--include-10s-silence` flags for stress-test runs only | Standalone (`benchmark.py` flag defaults) | `--pad_audio_to_sec 40` in YAML — every input audio is padded to a fixed 40 s | Required for our `s2s_incremental_v2` wrapper's perception cache. Pure infra-level choice; does change what the SUT acoustically sees. Standalone passes audio at natural length. |

### To bring this benchmark fully in line

In rough priority order (biggest expected metric delta first):

1. **Switch judge to `claude-3-5-sonnet-20241022`** via Anthropic-hosted API (not Bedrock). [official blog spec]
2. **Switch ASR to OpenAI's hosted Whisper API** (`https://api.openai.com/v1`, `model=whisper-1`). ~1k samples × $0.006/min ≈ cheap. [official blog spec]
3. **Lower judge temperature** to `0.2` and **raise judge `max_tokens`** to `256`. [standalone]
4. **Set SUT generation temperature to `0.6`** in the YAML. [standalone]
5. **Add `"You are a helpful assistant"` system prompt** in `prepare.py` (SUT) and as the judge's system message (`run_bba_scoring.py`). [standalone]
6. **Drop or reduce `--pad_audio_to_sec 40`** if the wrapper can tolerate variable-length audio. [standalone — wrapper architecture choice; revisit only if 1–5 don't close the gap]

### Note: the standalone is also not a bit-for-bit reproduction

The standalone repo (`bigbench-audio-standalone`) is the BBA vendor implementation we use as the secondary reference, but it itself deviates from the official blog on two points:

- **Whisper service**: uses OpenAI Realtime's server-side `response.audio_transcript.done` field (the model's own transcription, served alongside its audio output) instead of the OpenAI Whisper API the blog explicitly names.
- **Judge model**: uses `claude-sonnet-4-20250514` (Claude 4) instead of `claude-3-5-sonnet-20241022` (Claude 3.5 Sonnet Oct '24).

So neither implementation is currently a faithful reproduction of the leaderboard methodology. When the official blog *is* explicit (judge model identity, 3-run averaging, Whisper API), the blog wins; when it's silent, we follow the standalone.
