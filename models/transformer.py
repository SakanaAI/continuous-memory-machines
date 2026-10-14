import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cells.base import RecurrentCell


def _sinusoidal_position_embedding(max_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional embeddings matching LRA reference."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class TransformerSequenceModel(nn.Module):
    """Transformer encoder with optional causal masking.

    The default remains bidirectional for the LRA experiments. Setting
    ``causal=True`` provides a drop-in sequence baseline for recurrent tasks.
    """

    def __init__(self, d_input: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, max_seq_len: int,
                 learn_pos_emb: bool = False, causal: bool = False):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.causal = causal

        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()

        if learn_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        else:
            self.register_buffer("pos_emb", _sinusoidal_position_embedding(max_seq_len, d_model).unsqueeze(0))  # (1, max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self._disable_attention_bias()
        self._init_weights()

    def _disable_attention_bias(self):
        for layer in self.encoder.layers:
            attn = layer.self_attn
            attn.in_proj_bias = None
            if attn.out_proj.bias is not None:
                attn.out_proj.bias = None

    def _init_weights(self):
        for layer in self.encoder.layers:
            attn = layer.self_attn
            nn.init.xavier_uniform_(attn.in_proj_weight)
            nn.init.xavier_uniform_(attn.out_proj.weight)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def get_initial_state(self, batch_size: int = 1):
        return ()

    def forward(self, input: torch.Tensor, state=None):
        # input: (T, B, d_input)
        T, B, _ = input.shape
        x = self.input_proj(input)  # (T, B, d_model)
        x = x + self.pos_emb[0, :T, :].unsqueeze(1)  # (T, 1, d_model) broadcast
        x = self.drop(x)
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=input.device), diagonal=1
            )
        x = self.encoder(
            x, mask=causal_mask, is_causal=self.causal
        )  # (T, B, d_model)
        x = self.final_norm(x)  # (T, B, d_model)
        return x, ()


class IncrementalTransformerBlock(nn.Module):
    """Pre-normalization Transformer block with an incremental KV cache."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)

    def forward_step(self, x, key_cache, value_cache):
        batch_size = x.size(0)
        normed = self.attn_norm(x)
        q = self.q_proj(normed).view(batch_size, self.nhead, 1, self.head_dim)
        k = self.k_proj(normed).view(batch_size, self.nhead, 1, self.head_dim)
        v = self.v_proj(normed).view(batch_size, self.nhead, 1, self.head_dim)
        key_cache = torch.cat((key_cache, k), dim=2)
        value_cache = torch.cat((value_cache, v), dim=2)

        # The cache contains only the current and preceding ticks, so no
        # additional causal mask is required for this single-token query.
        attended = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = attended.reshape(batch_size, -1)
        x = x + self.attn_dropout(self.out_proj(attended))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x, key_cache, value_cache


class CausalTransformerCell(RecurrentCell):
    """Incremental causal Transformer for iterative cross-attention tasks.

    Each input is one visual glimpse retrieved at the current internal tick.
    Per-layer KV caches hold the preceding tick representations, while the
    final state is used by ``CrossAttentionRecurrentModel`` to query the image
    at the following tick.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        learn_pos_emb: bool = False,
    ):
        super().__init__(d_input=d_input, d_model=d_model)
        assert num_layers > 0, "num_layers must be positive"
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        if learn_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        else:
            self.register_buffer(
                "pos_emb",
                _sinusoidal_position_embedding(max_seq_len, d_model).unsqueeze(0),
            )
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                IncrementalTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.initial_query = nn.Parameter(torch.zeros(d_model))

        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def initialize_state(self):
        empty_cache = self.initial_query.new_empty(1, self.nhead, 0, self.head_dim)
        caches = tuple(
            cache
            for _ in self.blocks
            for cache in (empty_cache.clone(), empty_cache.clone())
        )
        return (*caches, self.initial_query.unsqueeze(0))

    def forward(self, input: torch.Tensor, state):
        assert input.ndim == 2, "Input should be of size B x D"
        position = state[0].size(2)
        if position >= self.max_seq_len:
            raise ValueError(
                f"Transformer KV cache length {position + 1} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.input_proj(input)
        x = self.input_dropout(x + self.pos_emb[:, position, :])
        new_caches = []
        for layer_index, block in enumerate(self.blocks):
            key_cache = state[2 * layer_index]
            value_cache = state[2 * layer_index + 1]
            x, key_cache, value_cache = block.forward_step(x, key_cache, value_cache)
            new_caches.extend((key_cache, value_cache))

        output = self.final_norm(x)
        return output, (*new_caches, output)
