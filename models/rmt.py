"""Recurrent Memory Transformer (Bulatov et al., NeurIPS 2022).

Adapted from https://github.com/booydar/LM-RMT, pytorch/mem_transformer.py
(Apache-2.0; see licenses/LM-RMT-LICENSE). Changes: continuous feature inputs,
PyTorch SDPA attention, no Transformer-XL cache or language-model head, and the
repository's sequence/state interface. Relative positions, memory masking and
read/write recurrence follow the reference's attn_type=0, mem_at_end=True path.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _RelativeAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, attention_dropout, pre_lnorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pre_lnorm = pre_lnorm
        self.attention_dropout = attention_dropout
        self.qkv_net = nn.Linear(d_model, 3 * d_model, bias=False)
        self.r_net = nn.Linear(d_model, d_model, bias=False)
        self.o_net = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _relative_shift(scores):
        # Match Transformer-XL's relative shift, including the unmasked memory
        # blocks. The leading dimensions are query and relative-position index.
        length = scores.size(0)
        padded = torch.cat((torch.zeros_like(scores[:, :1]), scores), dim=1)
        return padded.reshape(length + 1, length, *scores.shape[2:])[1:].reshape_as(scores)

    def forward(self, x, positions, r_w_bias, r_r_bias, mask):
        length, batch_size, d_model = x.shape
        normed = self.layer_norm(x) if self.pre_lnorm else x
        q, k, v = self.qkv_net(normed).chunk(3, dim=-1)
        q, k, v = [t.reshape(length, batch_size, self.num_heads, self.head_dim) for t in (q, k, v)]
        relative_keys = self.r_net(positions).reshape(length, self.num_heads, self.head_dim)
        relative_scores = torch.einsum("ibhd,jhd->ijbh", q + r_r_bias, relative_keys)
        relative_scores = self._relative_shift(relative_scores).permute(2, 3, 0, 1)
        # SDPA scales the content score; its additive bias must already be scaled.
        attention_bias = (relative_scores / math.sqrt(self.head_dim)).masked_fill(mask, float("-inf"))
        attended = F.scaled_dot_product_attention(
            (q + r_w_bias).permute(1, 2, 0, 3),
            k.permute(1, 2, 0, 3),
            v.permute(1, 2, 0, 3),
            attn_mask=attention_bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.permute(2, 0, 1, 3).reshape(length, batch_size, d_model)
        output = x + self.drop(self.o_net(attended))
        return output if self.pre_lnorm else self.layer_norm(output)


class _FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm):
        super().__init__()
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.layer_norm(x) if self.pre_lnorm else x
        output = x + self.CoreNet(normed)
        return output if self.pre_lnorm else self.layer_norm(output)


class _RMTLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_inner, dropout, attention_dropout, pre_lnorm):
        super().__init__()
        self.dec_attn = _RelativeAttention(d_model, num_heads, dropout, attention_dropout, pre_lnorm)
        self.pos_ff = _FeedForward(d_model, d_inner, dropout, pre_lnorm)

    def forward(self, x, positions, r_w_bias, r_r_bias, mask):
        return self.pos_ff(self.dec_attn(x, positions, r_w_bias, r_r_bias, mask))


class RecurrentMemoryTransformer(nn.Module):
    """Causal segment-level RMT for time-major continuous inputs (T, B, D).

    Each segment processes [previous memory, inputs, previous memory]. Only
    input-token outputs are returned; final write-token outputs become the next
    memory. State is a one-element tuple containing (B, num_mem_tokens, d_model).
    Gradients flow through all segments, including state passed between calls.

    A forward call ends its final segment even if shorter than segment_length.
    To reproduce a single long call when streaming, split at segment boundaries.
    Positions restart per segment, so total sequence length is unbounded.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 4,
        segment_length: int = 10,
        num_mem_tokens: int = 32,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pre_lnorm: bool = False,
    ):
        super().__init__()
        for name, value in (
            ("d_input", d_input), ("d_model", d_model), ("num_heads", num_heads),
            ("num_layers", num_layers), ("segment_length", segment_length),
            ("num_mem_tokens", num_mem_tokens), ("ff_multiplier", ff_multiplier),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads or d_model % 2:
            raise ValueError("d_model must be even and divisible by num_heads")
        if not 0 <= dropout < 1 or not 0 <= attention_dropout < 1:
            raise ValueError("dropout and attention_dropout must be in [0, 1)")

        self.d_input = d_input
        self.d_model = d_model
        self.segment_length = segment_length
        self.num_mem_tokens = num_mem_tokens
        # Keep a projection even at equal widths so parameter-budget search is
        # monotonic in d_model and all continuous inputs use the same embedding.
        self.input_proj = nn.Linear(d_input, d_model)
        self.embedding_scale = math.sqrt(d_model)
        # The reference initializes all slots from one random vector, but they
        # remain separate learnable parameters.
        self.mem_tokens = nn.Parameter(torch.randn(1, 1, d_model).repeat(1, num_mem_tokens, 1))
        self.r_w_bias = nn.Parameter(torch.empty(num_heads, d_model // num_heads))
        self.r_r_bias = nn.Parameter(torch.empty(num_heads, d_model // num_heads))
        self.register_buffer(
            "inv_freq", 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            _RMTLayer(d_model, num_heads, ff_multiplier * d_model, dropout, attention_dropout, pre_lnorm)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.apply(self._init_weights)
        nn.init.normal_(self.r_w_bias, std=0.02)
        nn.init.normal_(self.r_r_bias, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

    def get_initial_state(self, batch_size: int = 1):
        return (self.mem_tokens.expand(batch_size, -1, -1),)

    def _segment(self, inputs, memory):
        num_mem = self.num_mem_tokens
        memory = memory.transpose(0, 1)
        x = torch.cat((memory, inputs, memory), dim=0)
        length = x.size(0)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        mask[:num_mem, :num_mem] = False
        mask[-num_mem:, -num_mem:] = False
        position = torch.arange(length - 1, -1, -1, device=x.device, dtype=self.inv_freq.dtype)
        angles = position[:, None] * self.inv_freq[None, :]
        x = self.drop(x)
        positions = self.drop(torch.cat((angles.sin(), angles.cos()), dim=-1).to(x.dtype))
        for layer in self.layers:
            x = layer(x, positions, self.r_w_bias, self.r_r_bias, mask)
        x = self.drop(x)
        return x[num_mem:-num_mem], x[-num_mem:].transpose(0, 1)

    def forward(self, input: torch.Tensor, state=None):
        if input.ndim != 3 or input.size(-1) != self.d_input:
            raise ValueError(f"Expected input with shape (T, B, {self.d_input})")
        if state is None:
            state = self.get_initial_state(input.size(1))
        if len(state) != 1 or state[0].shape != (input.size(1), self.num_mem_tokens, self.d_model):
            raise ValueError("RMT state must contain one (B, num_mem_tokens, d_model) tensor")
        memory = state[0]
        if input.size(0) == 0:
            return input.new_empty(0, input.size(1), self.d_model), (memory,)

        embedded = self.input_proj(input) * self.embedding_scale
        outputs = []
        for segment in embedded.split(self.segment_length, dim=0):
            hidden, memory = self._segment(segment, memory)
            outputs.append(hidden)
        return torch.cat(outputs, dim=0), (memory,)


class RMTCell(nn.Module):
    """One RMT segment per adaptive visual glimpse.

    CrossAttentionRecurrentModel retrieves patches using the previous output,
    so each tick must finish before the next input is available. With one input
    token per segment, the sequence RMT already implements exactly this update.
    State contains recurrent memory followed by the next cross-attention query.
    """

    def __init__(self, model: RecurrentMemoryTransformer):
        super().__init__()
        if model.segment_length != 1:
            raise ValueError(
                "Patch-mode RMT requires model.segment_length=1: each recurrent "
                "step retrieves patches using the previous step's output."
            )
        self.model = model
        self.d_input = model.d_input
        self.d_model = model.d_model
        self.initial_query = nn.Parameter(model.mem_tokens.new_zeros(model.d_model))

    def initialize_state(self):
        return (*self.model.get_initial_state(), self.initial_query.unsqueeze(0))

    def forward(self, input: torch.Tensor, state):
        if input.ndim != 2 or len(state) != 2:
            raise ValueError("RMTCell expects (B, d_input) inputs and (memory, query) state")
        hidden, (memory,) = self.model(input.unsqueeze(0), (state[0],))
        output = hidden.squeeze(0)
        return output, (memory, output)
