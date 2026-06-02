"""NV-gateway judge routing for the vendored FDB-v3 official evaluators.

Upstream Full-Duplex-Bench v3 (DanielLin94144/Full-Duplex-Bench) calls the
gpt-4o judge directly against the public OpenAI API (api.openai.com). On the
eval cluster we route that *same* gpt-4o judge through NVIDIA's
OpenAI-compatible gateway instead, using the cluster's NVIDIA_API_KEY (the
scoring job sets it from NV_INFERENCE_KEY). Base URL and model are
env-overridable; the defaults preserve the official methodology (gpt-4o).

This is the ONLY behavioral patch applied to the vendored upstream scripts; see
PROVENANCE.md. Keeping it in one helper makes the per-script edits a two-line
substitution and keeps future upstream diffs clean.
"""

import os

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"


def _base_url() -> str:
    return (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("NVIDIA_BASE_URL")
        or DEFAULT_BASE_URL
    )


def _api_key():
    return os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")


def get_client():
    """Lazy OpenAI client pointed at the NV gateway (or env-provided base_url)."""
    from openai import OpenAI

    return OpenAI(base_url=_base_url(), api_key=_api_key())


def judge_model() -> str:
    """Resolve the judge model, mapping to the gateway's qualified names.

    Default is gpt-4o (the official benchmark's judge), mapped to the gateway's
    ``azure/openai/gpt-4o``. Override with FD3_JUDGE_MODEL / JUDGE_MODEL. Names
    already qualified (openai/ azure/ nvidia/) are passed through unchanged.
    """
    m = os.getenv("FD3_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or "gpt-4o"
    if m.startswith(("openai/", "azure/", "nvidia/")):
        return m
    if m == "gpt-4o":
        return "azure/openai/gpt-4o"
    if m == "gpt-4o-mini":
        return "azure/openai/gpt-4o-mini"
    if m.startswith("gpt-"):
        return f"openai/openai/{m}"
    return m
