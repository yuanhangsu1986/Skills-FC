# Evaluation Comparison Report

Comparing 2 model(s):

- December Model
- November Model

## S2S Demo Evaluation

| Metric | November Model | December Model |
|---|---|---|
| Samples Evaluated | 60 | 60 |
| **Turn-Taking** | | |
|   Latency (ms) ↓ | 404.8 | **297.9** |
|   Precision (%) ↑ | **76.9** | 75.9 |
|   Recall (%) ↑ | 71.8 | **80.5** |
|   F1 (%) ↑ | 72.5 | **77.0** |
| **Barge-In** | | |
|   Success Rate (%) ↑ | 72.3 | **90.2** |
|   Latency (ms) ↓ | 535.8 | **413.9** |
| **Backchanneling** | | |
|   Accuracy (%) ↑ | **0.0** | 0.0 |
| **User Speech (ASR)** | | |
|   WER (%) ↓ | 34.5 | **24.4** |
|   OOB Ratio ↓ | 0.009 | **0.000** |
| **Agent Speech (TTS)** | | |
|   WER (%) ↓ | 10.2 | **4.1** |
|   CER (%) ↓ | 4.5 | **2.2** |
|   Hallucination (%) ↓ | 3.8 | **2.6** |
| **LLM Judge (1-5)** | | |
|   Overall Rating ↑ | 3.27 | **3.31** |
|   Full Response ↑ | **3.53** | 3.45 |
|   Sounded Response ↑ | 3.00 | **3.17** |

## VoiceBench Evaluation

### Summary

| Model | Total Samples | Subtests |
|---|---|---|
| November Model | 5001 | 5 |
| December Model | 5001 | 5 |

### Per-Subtest Metrics

#### advbench

| Metric | November Model | December Model |
|---|---|---|
| Samples | 520 | 520 |
| refusal_rate ↑ | 0.95 | **0.98** |

#### alpacaeval

| Metric | November Model | December Model |
|---|---|---|
| Samples | 199 | 199 |
| gpt ↑ | **3.64** | 3.32 |

#### commoneval

| Metric | November Model | December Model |
|---|---|---|
| Samples | 200 | 200 |
| gpt ↑ | **3.33** | 3.08 |

#### mmsu

| Metric | November Model | December Model |
|---|---|---|
| Samples | 3074 | 3074 |
| acc ↑ | 41.67 | **43.14** |
| fail ↓ | 4.59 | **1.66** |

#### openbookqa

| Metric | November Model | December Model |
|---|---|---|
| Samples | 455 | 455 |
| acc ↑ | 58.02 | **58.68** |
| fail ↓ | 1.32 | **0.44** |

---
*↑ = higher is better, ↓ = lower is better, **bold** = best value*
