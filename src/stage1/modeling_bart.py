# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model. """
from base64 import encode
import copy
from distutils.command.config import config
import math
import random
from sys import is_finalizing
import warnings
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqLMOutput, 
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.bart.configuration_bart import BartConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def sample_from_logits(y):
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

    def superforward(self, positions):
        return super().forward(positions.to(self.weight.device))


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartSentenceAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: BartConfig,
        mask_token_id: int,
        embed_dim: int,
        num_heads: int,
        max_sen_num: int, 
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sen_num = max_sen_num
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5

        self.sen_dim = 128
        self.q_proj = nn.Linear(self.sen_dim, self.sen_dim, bias=bias)
        self.k_proj = nn.Linear(self.sen_dim, self.sen_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.sen_dim, bias=bias)

        self.k_proj_tfmr = nn.Linear(self.sen_dim, self.sen_dim, bias=bias)
        self.v_proj_tfmr = nn.Linear(self.sen_dim, self.sen_dim, bias=bias)
        self.q_proj_tfmr = nn.Linear(embed_dim, self.sen_dim, bias=bias)
        self.out_proj_tfmr = nn.Linear(self.sen_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_sent(self, tensor: torch.Tensor, seq_len: int, sen_num:int, bsz: int):
        return tensor.view(bsz, self.num_heads, seq_len, sen_num, self.head_dim).transpose(1, 2).transpose(2, 3).contiguous().view(bsz, seq_len, sen_num, self.num_heads*self.head_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        past_hidden_id: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # print("input_ids", input_ids.size(), input_ids.int().cpu().numpy().tolist())
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        final_hidden_states = hidden_states
        if past_hidden_id is not None:
            # reuse k, v, self_attention
            hidden_states = torch.cat([past_hidden_id[0], hidden_states], dim=1)
            # input_ids = torch.cat([past_hidden_id[1], input_ids], dim=1)
        past_hidden_id = (hidden_states, )

        # print(final_hidden_states.size(), past_hidden_id[0].size(), input_ids.size())

        bsz, tgt_len, _ = final_hidden_states.size()

        # [batch_size, src_len]
        mask_pos = torch.eq(input_ids, self.mask_token_id).type_as(hidden_states)
        # [batch_size, src_len]
        mask_pos_sum0 = torch.cumsum(mask_pos, 1)
        # [1, sen_num]
        sen_range = torch.arange(0, self.max_sen_num)[None, :].to(hidden_states.device)

        # get query proj
        # [batch_size, sen_num, src_len]
        sen_pos_mask = torch.eq(sen_range[:, :, None] + 1, (mask_pos_sum0 * mask_pos)[:, None, :]).type_as(hidden_states)
        # [batch_size, sen_num]
        sen_num_mask = torch.sum(sen_pos_mask, 2)
        # hidden_state: [batch_size, tgt_len, hidden_size]
        # static_pos_states: [batch_size, sen_num, sen_dim]
        static_sen_states = self.v_proj(torch.bmm(sen_pos_mask, hidden_states))
        all_sen_states = static_sen_states

        # [bsz, tgt_len, hidden_size]
        q_token = self.q_proj_tfmr(final_hidden_states) * (self.sen_dim ** -0.5)
        # [bsz, 2*sen_num, hidden_size]
        k_sen = self.k_proj_tfmr(all_sen_states)
        v_sen = self.v_proj_tfmr(all_sen_states)

        attn_mask = torch.full((bsz, tgt_len, self.max_sen_num), torch.finfo(hidden_states.dtype).min).to(hidden_states.device)
        tmp_attn_mask = torch.le(sen_range[:, None, :] + 1, mask_pos_sum0[:, -tgt_len:, None])
        # tmp_attn_mask = torch.cat([tmp_attn_mask, tmp_attn_mask], 2)
        attn_mask.masked_fill_(tmp_attn_mask, 0)
        attn_mask = attn_mask.to(hidden_states.dtype).view(bsz, tgt_len, self.max_sen_num)

        # [bsz, tgt_len, 2*sen_num]
        attn_weights = F.softmax(torch.bmm(q_token, k_sen.transpose(1,2)) + attn_mask, dim=-1) * (tmp_attn_mask.type_as(hidden_states))
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        sen_attn_output = self.out_proj_tfmr(torch.bmm(attn_probs, v_sen))

        return sen_attn_output, past_hidden_id, (None, None, static_sen_states, None)



class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, use_sen_att: Optional[bool] = True):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.use_sen_att = use_sen_att
        if self.use_sen_att:
            self.sentence_attn = BartSentenceAttention(
                config,
                config.mask_token_id,
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                max_sen_num=config.max_sen_num,
            )
            self.sen_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.use_sen_att:
            ############################sentence-level attention############################
            residual = hidden_states
            past_hidden_id = past_key_value[4:] if past_key_value is not None else None
            hidden_states, present_hidden_id, sen_hidden_states = self.sentence_attn(
                input_ids = input_ids,
                hidden_states = hidden_states,
                past_hidden_id = past_hidden_id,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.sen_attn_layer_norm(hidden_states)
            ############################sentence-level attention############################

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        if self.use_sen_att:
            present_key_value = present_key_value + present_hidden_id

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        if self.use_sen_att:
            outputs += (sen_hidden_states, )

        return outputs


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        add_pos=True,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if add_pos:
            embed_pos = self.embed_positions(input_shape)
            hidden_states = inputs_embeds + embed_pos
        else:
            hidden_states = inputs_embeds

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            if len(attention_mask.size()) == 2:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.config = config

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)


        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.sen_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        use_sen_att_list = [True for _ in range(config.decoder_layers)]
        self.layers = nn.ModuleList([BartDecoderLayer(config, use_sen_att=use_sen_att_list[kkk]) for kkk in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.sen_dim = 128
        self.code_num = 512 # 512 # 8192 # 32768 # 5120
        self.char_num = 100
        self.codebook = nn.Embedding(self.code_num, self.sen_dim)

        self.char_plan = nn.Linear(config.d_model, self.char_num, bias=True)

        self.k_pred = nn.Linear(config.d_model, self.sen_dim, bias=True)
        self.v_pred = nn.Linear(config.d_model, self.sen_dim, bias=True)
        self.q_pred = nn.Linear(config.d_model, self.sen_dim, bias=True)
        self.out_pred = nn.Linear(self.sen_dim, self.sen_dim, bias=True)

        self.sen_head_enc = nn.Linear(config.d_model, self.sen_dim, bias=True)
        self.sen_head_dec = nn.Linear(self.sen_dim, config.d_model, bias=True)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        # =========================================================================
        if past_key_values is not None:
            input_ids = torch.cat([past_key_values[-1][0], input_ids], 1)
            last_k_pred_state, last_v_pred_state, last_input_hidden_states = past_key_values[-1][1], past_key_values[-1][2], past_key_values[-1][3]
        else:
            input_ids = input_ids
            last_k_pred_state, last_v_pred_state, last_input_hidden_states = None, None, None
        # [bsz, tgt_len]
        mask_pos = torch.eq(input_ids, self.config.mask_token_id).type_as(inputs_embeds)
        mask_pos_sum0 = torch.cumsum(mask_pos, 1)
        # [1, sen_num]
        sen_range = torch.arange(0, self.config.max_sen_num)[None, :].to(inputs_embeds.device)
        # [batch_size, sen_num, src_len]
        sen_pos_mask = torch.eq(sen_range[:, :, None] + 1, (mask_pos_sum0 * mask_pos)[:, None, :]).type_as(inputs_embeds)
        mask_pos_sum1 = mask_pos_sum0 - mask_pos + 1
        mask_pos_sum = mask_pos_sum1 * (1 - mask_pos)
        # [bsz, sen_num, len]
        mask_pos_matrix = torch.eq(sen_range[:, :, None] + 1, mask_pos_sum[:, None, :]).type_as(inputs_embeds)        

        all_sentence_representation = ()
        if not use_cache:
            encoder_input_ids = input_ids
            encoder_sen_attn_mask = torch.eq(mask_pos_sum1[:, :, None], mask_pos_sum1[:, None, :])[:, None, :, :]
            inverted_encoder_sen_attn_mask = 1.0 - encoder_sen_attn_mask.type_as(inputs_embeds)
            inverted_encoder_sen_attn_mask = inverted_encoder_sen_attn_mask.masked_fill(inverted_encoder_sen_attn_mask.bool(), torch.finfo(inputs_embeds.dtype).min)
            # [bsz, len, hidden_dim]
            encoder_inputs_embeds = encoder.embed_tokens(encoder_input_ids) * encoder.embed_scale

            # [bsz, len]
            sen_token_id = torch.sum(torch.cumsum(mask_pos_matrix, -1) * mask_pos_matrix, 1).type_as(input_ids)
            # [bsz, len, embed_dim]
            sen_token_positions = encoder.embed_positions.superforward(sen_token_id)
            # print(sen_token_positions.size())

            target_sen_states = encoder(
                inputs_embeds = encoder_inputs_embeds + sen_token_positions,
                attention_mask = inverted_encoder_sen_attn_mask,
                add_pos = False,
            )["last_hidden_state"]

            # [bsz, src_len, hidden_dim]
            person_target_sen_states = self.sen_head_enc(target_sen_states)
            # [bsz, src_len]
            person_pos = torch.gt(input_ids, self.config.mask_token_id).type_as(inputs_embeds)
            # [bsz, sen_num, src_len]
            sen_person_pos = mask_pos_matrix * person_pos[:, None, :]

            # sen_person_pos_random = torch.eq(torch.cumsum(torch.cumsum(sen_person_pos, -1), -1), 1).type_as(inputs_embeds)
            random_sample = (torch.rand(sen_person_pos.size()).to(inputs_embeds.device) + 1.) * sen_person_pos
            # [bsz, sen_num, src_len]
            sen_person_pos_random = sample_from_logits(random_sample) * sen_person_pos
            # [bsz, sen_num, hidden_dim]
            char_embeds = torch.bmm(sen_person_pos_random, inputs_embeds)
            bsz_zero = torch.zeros(input_shape[0], 1, self.sen_dim).to(inputs_embeds.device)
            # [bsz, sen_num]
            sen_person_id = torch.sum(sen_person_pos_random * input_ids[:, None, :], -1)

            random_sample_negative1 = torch.rand(sen_person_pos.size()).to(inputs_embeds.device) + 1.
            random_sample_negative2 = (torch.rand(sen_person_pos.size()).to(inputs_embeds.device) + 1.) * (sen_person_pos - sen_person_pos_random)
            random_sample_negative = (random_sample_negative1 + random_sample_negative2) * mask_pos_matrix
            sen_person_pos_random_negative = sample_from_logits(random_sample_negative) * sen_person_pos
            sen_person_id_negative = torch.sum(sen_person_pos_random_negative * input_ids[:, None, :], -1)
            diff_char = 1 - torch.eq(sen_person_id, sen_person_id_negative).type_as(inputs_embeds)
            sen_person_pos_random_negative = sen_person_pos_random_negative * diff_char[:, :, None]
            sen_person_id_negative = sen_person_id_negative * diff_char

            sen_person_pos_random = torch.cat([sen_person_pos_random, sen_person_pos_random_negative], 1)
            # [bsz, sen_num, sen_dim]
            plan_states_all = torch.bmm(sen_person_pos_random, person_target_sen_states)
            plan_states_all_norm = plan_states_all / (torch.norm(plan_states_all, 2, dim=-1, keepdim=True) + 1e-20)
            # [N, sen_dim]
            codebook_embed = self.codebook(torch.arange(0, self.code_num, dtype=torch.long).to(inputs_embeds.device))
            codebook_embed = codebook_embed / (torch.norm(codebook_embed, 2, dim=-1, keepdim=True) + 1e-20)
            logits_ = torch.einsum('bij,kj->bik', plan_states_all_norm, codebook_embed)

            tmp_shape = logits_.size()
            _, ind = logits_.max(dim=-1)
            logits_hard = torch.zeros_like(logits_).view(-1, tmp_shape[-1])
            logits_hard.scatter_(1, ind.view(-1, 1), 1)
            logits_hard = logits_hard.view(*tmp_shape)
            # Set gradients w.r.t. y_hard gradients w.r.t. y
            plan_states_prob = (logits_hard - logits_).detach() + logits_

            plan_states_all_cluster = torch.einsum('bik,kj->bij', plan_states_prob, codebook_embed)
            tmp_is_normal = torch.gt(torch.sum(sen_person_pos_random, -1), 0).type_as(inputs_embeds)[:, :, None]
            plan_states_all_norm = plan_states_all_norm * tmp_is_normal
            plan_states_all_cluster = plan_states_all_cluster * tmp_is_normal

            plan_states_all_norm, plan_states_all_norm_negative = plan_states_all_norm[:, :self.config.max_sen_num, :], \
                    plan_states_all_norm[:, self.config.max_sen_num:, :]
            plan_states_all_cluster, plan_states_all_cluster_negative = plan_states_all_cluster[:, :self.config.max_sen_num, :], \
                    plan_states_all_cluster[:, self.config.max_sen_num:, :]
            plan_states_soft = torch.cat([plan_states_all_norm[:,1:,:], bsz_zero], 1)
            plan_states_hard = torch.cat([plan_states_all_cluster[:,1:,:], bsz_zero], 1)
            plan_states_soft_negative = torch.cat([plan_states_all_norm_negative[:,1:,:], bsz_zero], 1)
            plan_states_hard_negative = torch.cat([plan_states_all_cluster_negative[:,1:,:], bsz_zero], 1)

            sen_person_id = torch.cat([bsz_zero[:,:,0], sen_person_id[:, 2:], bsz_zero[:,:,0]], 1)
            sen_person_id_negative = torch.cat([bsz_zero[:,:,0], sen_person_id_negative[:, 2:], bsz_zero[:,:,0]], 1)

            all_sentence_representation += (plan_states_soft, plan_states_hard, sen_person_id, logits_, plan_states_soft_negative, plan_states_hard_negative, sen_person_id_negative, )

            plan_states_all_cluster = (plan_states_all_cluster - plan_states_all_norm).detach() + plan_states_all_norm
            plan_states_perturb = self.sen_head_dec(plan_states_all_cluster) + char_embeds

            bsz_zero2 = torch.zeros(input_shape[0], 1, self.config.d_model).to(inputs_embeds.device)
            plan_states_perturb = torch.cat([bsz_zero2, plan_states_perturb[:,2:,:], bsz_zero2], 1)
            plan_states_expand_perturb = torch.bmm(sen_pos_mask.transpose(1,2), plan_states_perturb)
        else:
            bsz_zero = torch.zeros(input_shape[0], 1, self.sen_dim).to(inputs_embeds.device)
            if past_key_values is not None:
                # [bsz, 1, hidden_size]
                pred_attn_output = past_key_values[-1][4][:, -input_shape[1]:, :]
                is_normal_char = past_key_values[-1][5][:, -input_shape[1]:]
                char_embeds = past_key_values[-1][6][:, -input_shape[1]:, :]

                pred_attn_output = pred_attn_output / (torch.norm(pred_attn_output, 2, dim=-1, keepdim=True) + 1e-20)

                codebook_embed = self.codebook(torch.arange(0, self.code_num, dtype=torch.long).to(inputs_embeds.device))
                codebook_embed = codebook_embed / (torch.norm(codebook_embed, 2, dim=-1, keepdim=True) + 1e-20)
                # [bsz, 1, N]
                logits_ = torch.einsum('bij,kj->bik', pred_attn_output, codebook_embed)

                # [bsz, 1, N]
                tmp_shape = logits_.size()
                # [bsz, 1]
                _, ind = logits_.max(dim=-1)
                logits_hard = torch.zeros_like(logits_).view(-1, tmp_shape[-1])
                logits_hard.scatter_(1, ind.view(-1, 1), 1)
                logits_hard = logits_hard.view(*tmp_shape)
                # [bsz, 1, N]
                plan_states_prob = logits_hard
                # [bsz, 1, sen_dim]
                plan_states_all_cluster = torch.einsum('bik,kj->bij', plan_states_prob, codebook_embed)
                plan_states_all_cluster *= is_normal_char[:, :, None]
                # [bsz, 1, hidden_size]
                pred_attn_output = self.sen_head_dec(plan_states_all_cluster) + char_embeds

                all_sentence_representation += (None, None, None, logits_)

                first_mask_pos = torch.zeros([input_shape[0], 512]).to(mask_pos.device)
                first_mask_pos.scatter_(1, torch.ones([input_shape[0], 1]).to(mask_pos.device).type_as(ind)*2, 1)
                tmp_mask_pos = (mask_pos - first_mask_pos[:, :mask_pos.size()[1]])[:, -input_shape[1]:]
                plan_states_expand_perturb = (tmp_mask_pos[:, :, None] * pred_attn_output)

                is_normal_char *= tmp_mask_pos
                maskind = tmp_mask_pos * is_normal_char * ind
                maskind += (is_normal_char + tmp_mask_pos - 2)
            else:
                plan_states_expand_perturb = torch.zeros_like(inputs_embeds)
        hidden_states = inputs_embeds + positions + plan_states_expand_perturb # sen_embed_pos_expand +
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    input_ids,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    input_ids,
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
            if decoder_layer.use_sen_att:
                all_sentence_representation += (layer_outputs[-1], )
            else:
                all_sentence_representation += (None, )


        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        input_hidden_states = hidden_states.detach()
        k_pred_state = self.k_pred(input_hidden_states)
        v_pred_state = self.v_pred(input_hidden_states)
        if last_k_pred_state is not None:
            k_pred_state = torch.cat([last_k_pred_state, k_pred_state], dim=1)
            v_pred_state = torch.cat([last_v_pred_state, v_pred_state], dim=1)
            input_hidden_states = torch.cat([last_input_hidden_states, input_hidden_states], dim=1)

        tmplen = torch.arange(0, input_ids.size()[1]).type_as(sen_pos_mask).to(sen_pos_mask.device)
        tmpmask_all = torch.bmm(mask_pos_matrix.transpose(1,2), mask_pos_matrix) 
        tmpmask = tmpmask_all * torch.ge(tmplen[:, None], tmplen[None, :])[None, :, :]
        sen_pos_mask_before = torch.cat([sen_pos_mask[:, :, 1:], torch.zeros([input_shape[0], self.config.max_sen_num, 1]).to(sen_pos_mask.device)], 2)

        # [bsz, len, hidden_dim]
        last_sentence_meanpool = torch.bmm(tmpmask, input_hidden_states) / (torch.sum(tmpmask, 2, keepdim=True) + 1e-20)
        last_sentence_meanpool = last_sentence_meanpool[:, -inputs_embeds.size()[1]:, :]

        # [bsz, len, char_num]
        char_logits = self.char_plan(last_sentence_meanpool)

        if not use_cache:
            # [bsz, sen_num]
            is_normal_char = torch.gt(sen_person_id, 0).type_as(inputs_embeds)
            # [bsz, sen_num, hidden_dim]
            char_embed_sen = self.embed_tokens(sen_person_id.type_as(input_ids)) * is_normal_char[:, :, None]
            # [bsz, len, hidden_dim]
            char_embed = torch.bmm(sen_pos_mask_before.transpose(1,2), char_embed_sen)
        else:
            all_char_embed_id = torch.arange(self.config.mask_token_id + 1, 
                    self.config.mask_token_id + self.char_num)[None, :].to(inputs_embeds.device)

            # [1, char_num, hidden_dim]
            all_char_embed = torch.cat([torch.zeros([1, 1, self.config.d_model]).to(inputs_embeds.device), 
                self.embed_tokens(all_char_embed_id)], 1)[0]

            shape = char_logits.size()
            ind = torch.multinomial(torch.softmax(char_logits/1.2, -1).view(-1, shape[-1]), 1).view(shape[0], shape[1])
            char_logits_hard = torch.zeros_like(char_logits).view(-1, shape[-1])
            char_logits_hard.scatter_(1, ind.view(-1, 1), 1)
            char_logits_hard = char_logits_hard.view(*shape)

            # [bsz, len]
            is_normal_char = torch.gt(ind, 0).type_as(inputs_embeds)
            # [bsz, len, hidden_dim]
            char_embed = torch.einsum("blc,ch->blh", char_logits_hard, all_char_embed)
        q_pred_state = self.q_pred(last_sentence_meanpool + char_embed)

        pred_attn_weights = torch.bmm(q_pred_state, k_pred_state.transpose(1, 2))
        if attention_mask is not None:
            pred_attn_weights += attention_mask[:, 0, :, :]
        pred_attn_weights = F.softmax(pred_attn_weights, dim=-1)
        pred_attn_probs = F.dropout(pred_attn_weights, p=self.dropout, training=self.training)
        # [bsz, len, hidden_states]
        pred_attn_output = self.out_pred(torch.bmm(pred_attn_probs, v_pred_state))

        if use_cache:
            memory_state = (input_ids, k_pred_state, v_pred_state, input_hidden_states, pred_attn_output, is_normal_char, char_embed, )
            next_cache = next_decoder_cache
            next_cache += (memory_state, )
        else:
            # [bsz, sen_num, hidden_states]
            pred_attn_output = torch.bmm(sen_pos_mask_before, pred_attn_output)
            # [bsz, sen_num, char_num]
            char_logits = torch.bmm(sen_pos_mask_before, char_logits)
            next_cache = None
            all_sentence_representation += (pred_attn_output, char_logits, is_normal_char, )
        # =========================================================================        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            sentence_representation=all_sentence_representation,
        )


class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)

        self.decoder_encoder = BartEncoder(config)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        self.decoder_encoder.embed_tokens.weight.data = self.shared.weight.data.clone()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder=self.decoder_encoder,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions, 
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            sentence_representation=decoder_outputs.sentence_representation,
        )




class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            sentence_representation=outputs.sentence_representation,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            # "decoder_golden_repre": kwargs["decoder_golden_repre"] if "decoder_golden_repre" in kwargs else None,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

