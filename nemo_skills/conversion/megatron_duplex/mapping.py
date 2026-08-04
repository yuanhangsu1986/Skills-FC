"""Pure key/tensor transforms for Megatron DuplexALM hybrid exports.

The functions in this module deliberately do not import Megatron, NeMo, vLLM,
or torch.distributed.checkpoint.  Keeping the conversion rules isolated makes
them usable from a lightweight unit test and from the DCP export driver.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


MCORE_MODEL_PREFIX = "model."
MCORE_LLM_PREFIX = "model.backbone.mamba_model.mamba_model."


def strip_model_prefix(key: str) -> str:
    """Remove the DCP top-level ``model.`` namespace from a metadata key."""

    return key[len(MCORE_MODEL_PREFIX) :] if key.startswith(MCORE_MODEL_PREFIX) else key


def is_llm_key(key: str) -> bool:
    """Return whether a flattened DCP key belongs in the vLLM artifact."""

    if key.endswith("._extra_state"):
        return False
    return key.startswith(MCORE_LLM_PREFIX) or key in {
        "model.heads.function.linear.weight",
        "model.heads.asr.linear.weight",
        "model.text_encoder.embed_asr_tokens.weight",
    }


def is_frontend_key(key: str) -> bool:
    """Return whether a flattened DCP key belongs in the native artifact."""

    if key.endswith("._extra_state"):
        return False
    return key.startswith(
        (
            "model.audio_encoder.",
            "model.audio_projector.",
            "model.modality_adapter.",
            "model.fusion.",
        )
    ) or key in {
        "model.backbone._input_embeddings.weight",
        "model.backbone._lm_head.weight",
        "model.heads.function.linear.weight",
        "model.heads.asr.linear.weight",
        "model.text_encoder.embed_asr_tokens.weight",
    }


def frontend_target_key(source_key: str) -> str | None:
    """Map a DuplexALM state key to the DRIRF/NeMo VoiceChat namespace."""

    key = strip_model_prefix(source_key)
    prefix_rules = (
        ("audio_encoder.preprocessor.", "stt_model.perception.preprocessor."),
        ("audio_encoder.encoder.", "stt_model.perception.encoder."),
        ("modality_adapter.", "stt_model.perception.modality_adapter."),
        ("audio_projector.transform.linear.", "stt_model.perception.proj."),
        ("fusion.", "stt_model.fusion."),
        ("text_encoder.embed_asr_tokens.", "stt_model.embed_asr_tokens."),
        ("heads.asr.linear.", "stt_model.asr_head."),
        ("heads.function.linear.", "stt_model.function_head."),
    )
    for source, target in prefix_rules:
        if key.startswith(source):
            return target + key[len(source) :]

    aliases = {
        "backbone._input_embeddings.weight": "stt_model.embed_tokens.weight",
        "backbone._lm_head.weight": "stt_model.lm_head.weight",
    }
    return aliases.get(key)


def split_grouped_qkv(
    weight: Any,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[Any, Any, Any]:
    """Split Megatron grouped-QKV rows into HF Q, K, and V matrices.

    Megatron stores each query group as ``[Q heads..., K, V]``.  HF and the
    custom vLLM loader expect three separate matrices.  ``weight`` only needs
    the torch Tensor reshape/indexing interface, which keeps torch optional at
    module import time.
    """

    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by "
            f"num_key_value_heads={num_key_value_heads}"
        )
    heads_per_group = num_attention_heads // num_key_value_heads
    expected_rows = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    if weight.ndim != 2 or weight.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected QKV shape {tuple(weight.shape)}; expected first dimension {expected_rows}"
        )

    grouped = weight.reshape(num_key_value_heads, heads_per_group + 2, head_dim, weight.shape[1])
    query = grouped[:, :heads_per_group].reshape(num_attention_heads * head_dim, weight.shape[1])
    key = grouped[:, heads_per_group].reshape(num_key_value_heads * head_dim, weight.shape[1])
    value = grouped[:, heads_per_group + 1].reshape(num_key_value_heads * head_dim, weight.shape[1])
    return query.contiguous(), key.contiguous(), value.contiguous()


_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")


def llm_target_tensors(source_key: str, tensor: Any, hf_config: Mapping[str, Any]) -> dict[str, Any]:
    """Map one MCore MambaModel tensor to one or more custom-vLLM tensors."""

    key = strip_model_prefix(source_key)
    if key.startswith("backbone.mamba_model.mamba_model."):
        key = key[len("backbone.mamba_model.mamba_model.") :]

    top_level = {
        "embedding.word_embeddings.weight": "backbone.embeddings.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        "output_layer.weight": "lm_head.weight",
        "backbone._input_embeddings.weight": "backbone.embeddings.weight",
        "backbone._lm_head.weight": "lm_head.weight",
        "heads.function.linear.weight": "stt_model.function_head.weight",
        "heads.asr.linear.weight": "stt_model.asr_head.weight",
        "text_encoder.embed_asr_tokens.weight": "stt_model.embed_asr_tokens.weight",
    }
    if key in top_level:
        return {top_level[key]: tensor}

    match = _LAYER_RE.match(key)
    if not match:
        return {}
    layer, suffix = match.groups()
    target_prefix = f"backbone.layers.{layer}."

    direct_suffixes = {
        # Mamba layer.
        "mixer.in_proj.layer_norm_weight": "norm.weight",
        "mixer.dt_bias": "mixer.dt_bias",
        "mixer.A_log": "mixer.A_log",
        "mixer.D": "mixer.D",
        "mixer.in_proj.weight": "mixer.in_proj.weight",
        "mixer.conv1d.weight": "mixer.conv1d.weight",
        "mixer.conv1d.bias": "mixer.conv1d.bias",
        "mixer.norm.weight": "mixer.norm.weight",
        "mixer.out_proj.weight": "mixer.out_proj.weight",
        # Standalone MLP layer.
        "mlp.linear_fc1.layer_norm_weight": "norm.weight",
        "mlp.linear_fc1.weight": "mixer.up_proj.weight",
        "mlp.linear_fc2.weight": "mixer.down_proj.weight",
        # Attention layer.
        "self_attention.linear_qkv.layer_norm_weight": "norm.weight",
        "self_attention.linear_proj.weight": "mixer.o_proj.weight",
    }
    if suffix in direct_suffixes:
        return {target_prefix + direct_suffixes[suffix]: tensor}

    if suffix == "self_attention.linear_qkv.weight":
        num_heads = int(hf_config["num_attention_heads"])
        num_kv_heads = int(hf_config.get("num_key_value_heads", hf_config.get("num_query_groups", num_heads)))
        head_dim = int(hf_config.get("head_dim", int(hf_config["hidden_size"]) // num_heads))
        query, key_tensor, value = split_grouped_qkv(
            tensor,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return {
            target_prefix + "mixer.q_proj.weight": query,
            target_prefix + "mixer.k_proj.weight": key_tensor,
            target_prefix + "mixer.v_proj.weight": value,
        }

    return {}


def expected_custom_outputs(*, has_function_head: bool) -> list[str]:
    """Return outputs in the positional order required by patched vLLM."""

    outputs = ["text_logits", "asr_tokens", "asr_logits"]
    if has_function_head:
        outputs.extend(("function_tokens", "function_logits"))
    return outputs
