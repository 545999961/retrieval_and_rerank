# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" PyTorch Gemma model."""
import pickle
import random
import sys
import time
from dataclasses import dataclass

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings, ModelOutput,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import (GemmaMLP, GemmaRMSNorm, GemmaRotaryEmbedding, rotate_half,
                                                      apply_rotary_pos_emb, repeat_kv, GemmaPreTrainedModel,
                                                      GEMMA_START_DOCSTRING, GEMMA_INPUTS_DOCSTRING)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)

@dataclass
class CustomModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_masks: Optional[torch.FloatTensor] = None

@dataclass
class CustomCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_masks: Optional[torch.FloatTensor] = None

class GemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Ignore copy
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->Gemma
class GemmaFlashAttention2(GemmaAttention):
    """
    Gemma flash attention module. This module inherits from `GemmaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (GemmaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in GemmaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Gemma
class GemmaSdpaAttention(GemmaAttention):
    """
    Gemma attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GemmaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "GemmaModel is using GemmaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,
    "flash_attention_2": GemmaFlashAttention2,
    "sdpa": GemmaSdpaAttention,
}


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LLAMA->GEMMA,Llama->Gemma
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def token_compress(average_num,
                   hidden_states,
                   attention_mask,
                   query_lengths,
                   prompt_lengths,
                   compress_method,
                   weights: torch.Tensor = None):
    # get some specific parameters
    passage_lengths = torch.sum(attention_mask, dim=1, dtype=torch.int) - query_lengths - prompt_lengths # the raw passage lengths
    retain_passage_lengths = (passage_lengths + average_num - 1) // average_num # the passage lengths need to be retained
    final_useful_lengths = query_lengths + prompt_lengths + retain_passage_lengths # the final useful length after compress
    max_passage_length = torch.max(passage_lengths) # the max passage lengths
    max_final_lengths = torch.max(final_useful_lengths) # the max useful lengths after compress
    # make new hidden states and new attention masks
    new_hidden_states = torch.zeros((hidden_states.shape[0], max_final_lengths,
                                     hidden_states.shape[-1]), dtype=hidden_states.dtype).to(hidden_states.device)
    new_attention_mask = torch.ones((hidden_states.shape[0], max_final_lengths), dtype=attention_mask.dtype).to(attention_mask.device)
    # get new attention mask
    mask_attention_index = torch.arange(max_final_lengths, device=hidden_states.device).unsqueeze(0) >= final_useful_lengths[:, None]
    new_attention_mask[mask_attention_index] = 0
    # get new hidden states
    # add query into new hidden states
    query_index = torch.arange(max_final_lengths, device=hidden_states.device).unsqueeze(0)
    mask_query_index = query_index < query_lengths[:, None]
    new_hidden_states[mask_query_index] = hidden_states[:, : max_final_lengths, :][mask_query_index]
    # add prompt into new hidden states
    # get the index of the prompt in new hidden states
    new_prompt_start_length = query_lengths + retain_passage_lengths
    new_prompt_end_length = new_prompt_start_length + prompt_lengths
    new_prompt_index = torch.arange(max_final_lengths, device=hidden_states.device).unsqueeze(0)
    new_mask_prompt_index_start = new_prompt_index >= new_prompt_start_length[:, None]
    new_mask_prompt_index_end = new_prompt_index < new_prompt_end_length[:, None]
    new_mask_prompt_index = new_mask_prompt_index_start & new_mask_prompt_index_end
    # get the index of the prompt in hidden states
    raw_prompt_start_length = query_lengths + passage_lengths
    raw_prompt_end_length = raw_prompt_start_length + prompt_lengths
    raw_prompt_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    raw_mask_prompt_index_start = raw_prompt_index >= raw_prompt_start_length[:, None]
    raw_mask_prompt_index_end = raw_prompt_index < raw_prompt_end_length[:, None]
    raw_mask_prompt_index = raw_mask_prompt_index_start & raw_mask_prompt_index_end
    # replace the prompt hidden states
    new_hidden_states[new_mask_prompt_index] = hidden_states[raw_mask_prompt_index]
    # 以上均没问题

    # print(new_hidden_states.view(len(new_hidden_states), -1))
    # print(new_attention_mask)

    # get the index of the passage in new hidden states
    new_passage_start_length = query_lengths
    new_passage_end_length = new_passage_start_length + retain_passage_lengths
    new_passage_index = torch.arange(max_final_lengths, device=hidden_states.device).unsqueeze(0)
    new_mask_passage_index_start = new_passage_index >= new_passage_start_length[:, None]
    new_mask_passage_index_end = new_passage_index < new_passage_end_length[:, None]
    new_mask_passage_index = new_mask_passage_index_start & new_mask_passage_index_end
    # print(query_lengths, prompt_lengths, retain_passage_lengths, final_useful_lengths)
    # add passage into new hidden states
    if compress_method == 'last':
        # get the index of the passage in hidden states
        raw_passage_end_length = query_lengths + passage_lengths
        raw_passage_start_length = raw_passage_end_length - retain_passage_lengths
        raw_passage_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        raw_mask_passage_index_start = raw_passage_index >= raw_passage_start_length[:, None]
        raw_mask_passage_index_end = raw_passage_index < raw_passage_end_length[:, None]
        raw_mask_passage_index = raw_mask_passage_index_start & raw_mask_passage_index_end
        new_hidden_states[new_mask_passage_index] = hidden_states[raw_mask_passage_index]
    else:
        # get mask hidden states
        psg_start_length = query_lengths
        psg_end_length = query_lengths + passage_lengths
        psg_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        mask_psg_index_start = psg_index >= psg_start_length[:, None]
        mask_psg_index_end = psg_index < psg_end_length[:, None]
        mask_psg_index = mask_psg_index_start & mask_psg_index_end
        if compress_method == 'sample':
            weights = torch.rand((weights.shape[0], max_passage_length), device=weights.device)
            _, selected_indices = torch.topk(weights, torch.max(retain_passage_lengths))
            replacement_values = torch.full_like(selected_indices, max_passage_length, device=hidden_states.device)
            replace_indices = torch.arange(selected_indices.size(1), device=hidden_states.device).unsqueeze(
                0).expand_as(
                selected_indices)
            selected_indices = torch.where(replace_indices >= retain_passage_lengths.unsqueeze(1), replacement_values,
                                           selected_indices)
            selected_indices, _ = torch.sort(selected_indices)
            selected_indices = selected_indices + query_lengths[:, None]
            hidden_states = torch.gather(hidden_states, 1,
                                         selected_indices.unsqueeze(2).expand(-1, -1, hidden_states.size(2)))
            # get the index of the passage in hidden states
            raw_passage_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            raw_mask_passage_index = raw_passage_index < retain_passage_lengths[:, None]
            new_hidden_states[new_mask_passage_index] = hidden_states[raw_mask_passage_index]
        elif compress_method == 'weighted_drop':
            # get the large hidden states
            weights = torch.sum(weights, dim=1)
            weights = weights * mask_psg_index
            weights = torch.where(weights == 0.0, torch.finfo(weights.dtype).min, weights)
            # weights[weights.eq(0.0)] = torch.finfo(weights.dtype).min
            _, selected_indices = torch.topk(weights, torch.max(retain_passage_lengths), sorted=True)
            replacement_values = torch.full_like(selected_indices, max_passage_length, device=hidden_states.device)
            replace_indices = torch.arange(selected_indices.size(1), device=hidden_states.device).unsqueeze(0).expand_as(selected_indices)
            selected_indices = torch.where(replace_indices >= retain_passage_lengths.unsqueeze(1), replacement_values,
                                            selected_indices)
            selected_indices, _ = torch.sort(selected_indices)
            hidden_states = torch.gather(hidden_states, 1, selected_indices.unsqueeze(2).expand(-1, -1, hidden_states.size(2)))
            # get the index of the passage in hidden states
            raw_passage_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            raw_mask_passage_index = raw_passage_index < retain_passage_lengths[:, None]
            new_hidden_states[new_mask_passage_index] = hidden_states[raw_mask_passage_index]
        else:
            hidden_states = hidden_states * mask_psg_index.unsqueeze(-1)
            passage_hidden_states = torch.zeros((hidden_states.shape[0],
                                                 (max_passage_length + average_num - 1) // average_num * average_num,
                                                 hidden_states.shape[-1]), dtype=hidden_states.dtype).to(hidden_states.device)
            passage_end_length = passage_lengths
            passage_index = torch.arange(passage_hidden_states.shape[1], device=hidden_states.device).unsqueeze(0) # maybe exceed the max passage length
            mask_passage_index = passage_index < passage_end_length[:, None]

            raw_passage_end_length = query_lengths + passage_lengths
            raw_passage_start_length = query_lengths
            raw_passage_index = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            raw_mask_passage_index_start = raw_passage_index >= raw_passage_start_length[:, None]
            raw_mask_passage_index_end = raw_passage_index < raw_passage_end_length[:, None]
            raw_mask_passage_index = raw_mask_passage_index_start & raw_mask_passage_index_end
            passage_hidden_states[mask_passage_index] = hidden_states[raw_mask_passage_index]

            if compress_method == 'mean':
                passage_weights = torch.zeros((weights.shape[0],
                                               (max_passage_length + average_num - 1) // average_num * average_num)
                                              , dtype=weights.dtype).to(hidden_states.device)
                passage_weights[torch.logical_and(passage_weights == 0, mask_passage_index)] = 1
                # passage_weights[mask_passage_index] = 1
                passage_weights = passage_weights.view(passage_weights.shape[0], -1, average_num)
                passage_weights = passage_weights / torch.sum(passage_weights, dim=-1
                                                              ).view(passage_weights.shape[0], -1, 1)
                passage_weights = passage_weights.view(passage_weights.shape[0], -1)
                # passage_weights = torch.where(passage_weights == torch.nan, 0, passage_weights)
                passage_hidden_states = passage_hidden_states * passage_weights.unsqueeze(-1)
                passage_hidden_states = passage_hidden_states.view(passage_hidden_states.shape[0], -1, average_num,
                                                                   passage_hidden_states.shape[-1])
                passage_hidden_states = torch.sum(passage_hidden_states, dim=2)
                passage_end_length = retain_passage_lengths
                passage_index = torch.arange(passage_hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                mask_passage_index = passage_index < passage_end_length[:, None]
                new_hidden_states[new_mask_passage_index] = passage_hidden_states[mask_passage_index]
            elif compress_method == 'weighted_mean':
                # passage_weights = torch.zeros((weights.shape[0],
                #                                (max_passage_length + average_num - 1) // average_num * average_num,
                #                               weights.shape[1])).to(hidden_states.device)
                # passage_weights[mask_passage_index] = weights.transpose(1, 2)[raw_mask_passage_index]
                # passage_weights = passage_weights.transpose(1, 2)
                # print(passage_weights[0])
                # passage_weights = passage_weights.view(passage_weights.shape[0], passage_weights.shape[1], -1, average_num)
                # print(passage_weights[0], passage_weights.shape)
                # passage_weights = passage_weights / torch.sum(passage_weights, dim=-1, keepdim=True)
                # print(passage_weights[0], passage_weights.shape)
                # passage_weights = passage_weights.view()
                # print(passage_weights[0], passage_weights.shape)
                pass
            elif compress_method == 'weighted_mean_direct':
                passage_weights = torch.zeros((weights.shape[0],
                                               (max_passage_length + average_num - 1) // average_num * average_num)
                                              , dtype=weights.dtype).to(hidden_states.device)
                weights = torch.sum(weights, dim=1)
                passage_weights[mask_passage_index] = weights[raw_mask_passage_index]
                passage_weights = passage_weights.view(passage_weights.shape[0], -1, average_num)
                passage_weights = passage_weights / torch.sum(passage_weights, dim=-1
                                                              ).view(passage_weights.shape[0], -1, 1)
                passage_weights = passage_weights.view(passage_weights.shape[0], -1)
                # passage_weights = torch.where(passage_weights == torch.nan, 0, passage_weights)
                passage_hidden_states = passage_hidden_states * passage_weights.unsqueeze(-1)
                passage_hidden_states = passage_hidden_states.view(passage_hidden_states.shape[0], -1, average_num,
                                                                   passage_hidden_states.shape[-1])
                passage_hidden_states = torch.sum(passage_hidden_states, dim=2)
                passage_end_length = retain_passage_lengths
                passage_index = torch.arange(passage_hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                mask_passage_index = passage_index < passage_end_length[:, None]
                new_hidden_states[new_mask_passage_index] = passage_hidden_states[mask_passage_index]
            elif compress_method == 'weighted_mean_double':
                passage_weights = torch.zeros((weights.shape[0],
                                               (max_passage_length + average_num - 1) // average_num * average_num,
                                               weights.shape[1]), dtype=weights.dtype).to(hidden_states.device)
                passage_weights[mask_passage_index] = weights.transpose(1, 2)[raw_mask_passage_index]
                passage_weights = passage_weights.transpose(1, 2)
                passage_weights = passage_weights.view(passage_weights.shape[0], passage_weights.shape[1], -1,
                                                       average_num)
                passage_weights = passage_weights / torch.sum(passage_weights, dim=-1, keepdim=True)
                passage_weights = passage_weights.view(passage_weights.shape[0], passage_weights.shape[1], -1)
                passage_weights = torch.mean(passage_weights, dim=1)
                passage_hidden_states = passage_hidden_states * passage_weights.unsqueeze(-1)
                passage_hidden_states = passage_hidden_states.view(passage_hidden_states.shape[0], -1, average_num,
                                                                   passage_hidden_states.shape[-1])
                passage_hidden_states = torch.sum(passage_hidden_states, dim=2)
                passage_end_length = retain_passage_lengths
                passage_index = torch.arange(passage_hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                mask_passage_index = passage_index < passage_end_length[:, None]
                new_hidden_states[new_mask_passage_index] = passage_hidden_states[mask_passage_index]

    return new_hidden_states, new_attention_mask


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->GEMMA,Llama->Gemma
class GemmaModel(GemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_position_embeddings`.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_sep: Optional[int] = None,
        average_num: Optional[int] = None,
        query_lengths: Optional[int] = None,
        prompt_lengths: Optional[int] = None,
        compress_method: str = 'mean'
    ) -> Union[Tuple, CustomModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # if compress_method is not None and compress_method in ['weighted_mean', 'weighted_mean_double',
        #                                                        'weighted_mean_direct', 'weighted_drop']:
        output_attentions = True
        if self.config._attn_implementation == 'flash_attention_2':
            raise ValueError(
                "You cannot use flash attention 2 for weighted mean"
            )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if layer_sep is not None and average_num is not None:
            logger.warning_once(
                "`use_cache=True` is incompatible with reranker. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, past_seen_tokens)

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]) and (
                torch.sum(attention_mask) != attention_mask.shape[0] * attention_mask.shape[1])
        query_lengths = [0] * hidden_states.shape[0] if query_lengths is None else query_lengths
        prompt_lengths = [0] * hidden_states.shape[0] if prompt_lengths is None else prompt_lengths
        if not isinstance(query_lengths, torch.Tensor):
            query_lengths = torch.tensor(query_lengths, device=hidden_states.device)
        if not isinstance(prompt_lengths, torch.Tensor):
            prompt_lengths = torch.tensor(prompt_lengths, device=hidden_states.device)

        """
            query + passage + prompt
        """
        for idx, decoder_layer in enumerate(self.layers):
            if layer_sep is not None and average_num is not None and idx % layer_sep == 0 and idx != 0:
                if all_self_attns is not None:
                    weights = all_self_attns[-1][:, :, -1, :]
                else:
                    weights = None

                if left_padding:
                    raise ValueError('You must use right padding...')
                hidden_states, attention_mask = token_compress(average_num, hidden_states, attention_mask,
                                                               query_lengths, prompt_lengths, compress_method,
                                                               weights)

                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
                )
                position_ids = cache_position.unsqueeze(0)
                causal_mask = self._update_causal_mask(attention_mask, hidden_states, past_seen_tokens)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in
                         [hidden_states, next_cache, all_hidden_states, all_self_attns, attention_mask, attention_mask]
                         if v is not None)
        return CustomModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            attention_masks=attention_mask
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, past_seen_tokens):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min

        causal_mask = self.causal_mask[None, None, :, :].to(dtype=dtype, device=device) * min_dtype
        causal_mask = causal_mask.expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < past_seen_tokens + input_tensor.shape[1]:
                    offset = past_seen_tokens
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->GEMMA,Llama->Gemma,llama->gemma
class GemmaForCausalLM(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_sep: Optional[int] = None,
        average_num: Optional[int] = None,
        query_lengths: Optional[int] = None,
        prompt_lengths: Optional[int] = None,
        compress_method: str = 'mean'
    ) -> Union[Tuple, CustomCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if average_num is not None and average_num == 1:
            average_num = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            layer_sep=layer_sep,
            average_num=average_num,
            query_lengths=query_lengths,
            prompt_lengths=prompt_lengths,
            compress_method=compress_method
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_masks=outputs[-1]
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(self.model.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past