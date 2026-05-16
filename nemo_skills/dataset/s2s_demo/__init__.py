# S2S Demo dataset - for testing speech-to-speech models

DATASET_GROUP = "speechlm"
IS_BENCHMARK_GROUP = True

BENCHMARKS = {
    "s2s_demo.demo_20251124": {},
}

GENERATION_ARGS = "++prompt_format=openai"
