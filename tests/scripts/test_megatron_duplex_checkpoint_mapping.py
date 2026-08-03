import pytest

torch = pytest.importorskip("torch")

from scripts.megatron.duplex_checkpoint_mapping import (
    expected_custom_outputs,
    frontend_target_key,
    is_frontend_key,
    is_llm_key,
    llm_target_tensors,
    split_grouped_qkv,
)


def test_frontend_mapping_uses_voicechat_namespaces():
    assert (
        frontend_target_key("model.audio_projector.transform.linear.weight")
        == "stt_model.perception.proj.weight"
    )
    assert (
        frontend_target_key("model.backbone._input_embeddings.weight")
        == "stt_model.embed_tokens.weight"
    )
    assert (
        frontend_target_key("model.heads.function.linear.weight")
        == "stt_model.function_head.weight"
    )


def test_transformer_engine_extra_state_is_not_exported():
    llm_extra_state = (
        "model.backbone.mamba_model.mamba_model.decoder.layers.0.mixer.in_proj._extra_state"
    )
    frontend_extra_state = "model.audio_encoder.encoder.layers.0.self_attention._extra_state"
    assert is_llm_key(llm_extra_state) is False
    assert is_frontend_key(frontend_extra_state) is False


def test_split_grouped_qkv_preserves_group_order():
    # 4 Q heads, 2 KV heads, head_dim=1. Megatron rows are
    # group0=[q0,q1,k0,v0], group1=[q2,q3,k1,v1].
    packed = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    query, key, value = split_grouped_qkv(
        packed, num_attention_heads=4, num_key_value_heads=2, head_dim=1
    )
    assert query.flatten().tolist() == [0, 1, 4, 5]
    assert key.flatten().tolist() == [2, 6]
    assert value.flatten().tolist() == [3, 7]


def test_attention_mapping_splits_qkv_for_custom_vllm():
    packed = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    mapped = llm_target_tensors(
        "model.backbone.mamba_model.mamba_model.decoder.layers.14.self_attention.linear_qkv.weight",
        packed,
        {"hidden_size": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 1},
    )
    assert set(mapped) == {
        "backbone.layers.14.mixer.q_proj.weight",
        "backbone.layers.14.mixer.k_proj.weight",
        "backbone.layers.14.mixer.v_proj.weight",
    }


def test_custom_output_order_keeps_asr_placeholders_for_no_asr_model():
    assert expected_custom_outputs(has_function_head=True) == [
        "text_logits",
        "asr_tokens",
        "asr_logits",
        "function_tokens",
        "function_logits",
    ]
