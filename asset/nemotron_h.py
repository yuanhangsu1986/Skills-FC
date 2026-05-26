# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/vllm-project/vllm/blob/94d8ec8d2bcb4ec55e33022b313c7e978edf05e1/vllm/model_executor/models/bamba.py
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only NemotronH model."""

import os
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import NemotronHConfig

from vllm.logger import init_logger
logger = init_logger(__name__)


def _get_stt_model_cfg(config) -> dict:
    """Extract the stt.model sub-config from the nested NeMo training config."""
    model_field = getattr(config, 'model', None)
    if isinstance(model_field, dict):
        return model_field.get('stt', {}).get('model', {})
    return {}


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor):
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.mixer = NemotronHMLP(
            config,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
            layer_idx=layer_idx,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.ssm_state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.mamba_num_heads * config.mamba_head_dim,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.mamba_hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        return output, residual


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.mixer = NemotronHAttention(
            config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states=hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "M": NemotronHMambaDecoderLayer,
    "-": NemotronHMLPDecoderLayer,
    "*": NemotronHAttentionDecoderLayer,
}


@support_torch_compile
class NemotronHModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: NemotronHConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self._has_asr = getattr(config, 'has_asr_head', False)
        self.embed_asr_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
            if self._has_asr
            else None
        )

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.hybrid_override_pattern[layer_idx]
            ]
            return layer_class(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.hybrid_override_pattern), get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intmd_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor, input_asr_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(input_ids)
        if self.embed_asr_tokens is not None and input_asr_ids is not None:
            emb = emb + self.embed_asr_tokens(input_asr_ids)
        return emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_asr_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids, input_asr_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break

            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class NemotronHForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "backbone": "model",
            "stt_model.llm": "model",
            "stt_model.embed_tokens": "model.embed_tokens",
            "stt_model.embed_asr_tokens": "model.embed_asr_tokens",
            "stt_model.lm_head": "lm_head",
            "stt_model.asr_head": "asr_head",
            "stt_model.function_head": "function_head",
        },
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches."""
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_num_heads * hf_config.mamba_head_dim

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = NemotronHModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            if not lora_config
            else lora_config.lora_vocab_padding_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self._has_asr = getattr(config, 'has_asr_head', False)
        self.asr_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "asr_head"),
            )
            if self._has_asr
            else None
        )
        _custom_outputs = getattr(config, 'custom_outputs', None) or []
        self._has_fc = "function_tokens" in _custom_outputs or _get_stt_model_cfg(config).get('use_function_head', False)
        self.function_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "function_head"),
            )
            if self._has_fc
            else None
        )
        if self._has_fc:
            logger.info("NemotronH: function_head enabled and loaded")

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

        # Read logit boosts from environment variables (set by NeMo wrapper
        # before engine fork).  Format: VLLM_{ASR,TEXT}_BOOST_<TOKEN_ID>=<value>
        self.asr_logit_boosts: dict[int, float] = {}
        self.text_logit_boosts: dict[int, float] = {}
        for key, val in os.environ.items():
            if key.startswith("VLLM_ASR_BOOST_"):
                token_id = int(key[len("VLLM_ASR_BOOST_"):])
                boost = float(val)
                if boost != 0.0:
                    self.asr_logit_boosts[token_id] = boost
            elif key.startswith("VLLM_TEXT_BOOST_"):
                token_id = int(key[len("VLLM_TEXT_BOOST_"):])
                boost = float(val)
                if boost != 0.0:
                    self.text_logit_boosts[token_id] = boost

        self.make_empty_intmd_tensors = self.model.make_empty_intmd_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor, input_asr_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids, input_asr_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = kwargs.get("combined_embeds", None)
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, input_asr_ids=None
        )

        text_logits = self.compute_logits(hidden_states)
        for token_id, boost in self.text_logit_boosts.items():
            if boost != 0.0 and token_id < text_logits.shape[-1]:
                text_logits[:, token_id] += boost

        if self.asr_head is not None:
            asr_logits = self.logits_processor(self.asr_head, hidden_states)
            for token_id, boost in self.asr_logit_boosts.items():
                if boost != 0.0 and token_id < asr_logits.shape[-1]:
                    asr_logits[:, token_id] += boost
            asr_tokens = torch.argmax(asr_logits, dim=1)
        else:
            n = hidden_states.shape[0]
            asr_tokens = torch.zeros(n, dtype=torch.long, device=hidden_states.device)
            asr_logits = torch.zeros(n, self.config.vocab_size, dtype=hidden_states.dtype, device=hidden_states.device)

        results = [hidden_states, text_logits, asr_tokens, asr_logits]

        if self._has_fc and self.function_head is not None:
            function_logits = self.logits_processor(self.function_head, hidden_states)
            function_tokens = torch.argmax(function_logits, dim=1)
            results.extend([function_tokens, function_logits])

        return tuple(results)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Heads we can't load: when the model wasn't built with one of these
        # submodules (because config says it's absent for this checkpoint),
        # the corresponding tensors in the source state_dict have no
        # destination and must be DROPPED, not merely unmapped. Just removing
        # them from `hf_to_vllm_mapper` leaves raw `stt_model.<head>.*` names
        # in the iterator, which makes `AutoWeightsLoader` raise
        # `ValueError: There is no module or parameter named 'stt_model'`.
        absent_head_prefixes: list[str] = []
        if not self._has_asr:
            absent_head_prefixes.extend(("stt_model.asr_head.", "stt_model.embed_asr_tokens."))
        if not self._has_fc:
            absent_head_prefixes.append("stt_model.function_head.")

        if absent_head_prefixes:
            def _filter(it):
                for name, tensor in it:
                    if any(name.startswith(p) for p in absent_head_prefixes):
                        continue
                    yield name, tensor
            weights = _filter(weights)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

