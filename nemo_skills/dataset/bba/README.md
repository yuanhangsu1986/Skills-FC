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
