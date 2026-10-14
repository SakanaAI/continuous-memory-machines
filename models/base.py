import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_embed: int, d_out: int, num_heads: int = 4, use_rope: bool = False, rope_theta: float = 10000.0):
        super().__init__()
        assert d_embed % num_heads == 0, f"d_embed ({d_embed}) must be divisible by num_heads ({num_heads})"
        self.num_heads, self.head_dim, self.d_embed, self.d_out = num_heads, d_embed // num_heads, d_embed, d_out

        self.q_proj, self.k_proj, self.v_proj = nn.LazyLinear(d_embed, bias=False), nn.LazyLinear(d_embed, bias=False), nn.LazyLinear(d_embed, bias=False)
        self.out_proj = nn.Linear(d_embed, d_out, bias=False)

        self.use_rope = use_rope
        if use_rope:
            assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even for RoPE"
            self.register_buffer("rope_inv_freq", 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False) # (Dh/2,)

        self.return_weights = False

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float().unsqueeze(-1) * self.rope_inv_freq # (N,) x (Dh/2,) -> (N, Dh/2)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype) # (N, Dh/2) each
        x1, x2 = x[..., 0::2], x[..., 1::2] # (B, H, N, Dh) -> (B, H, N, Dh/2) each
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2) # -> (B, H, N, Dh)

    def forward(self, query: torch.Tensor, kv: torch.Tensor, v_in: torch.Tensor | None = None, q_pos: torch.Tensor | None = None, kv_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = kv.shape
        H, Dh = self.num_heads, self.head_dim
        v_src = kv if v_in is None else v_in

        q = self.q_proj(query).view(B, 1, H, Dh).transpose(1, 2) # (B, d_query) -> (B, H, 1, Dh)
        k = self.k_proj(kv).view(B, N, H, Dh).transpose(1, 2) # (B, N, d_kv) -> (B, H, N, Dh)
        v = self.v_proj(v_src).view(B, N, H, Dh).transpose(1, 2) # (B, N, d_v) -> (B, H, N, Dh)

        if self.use_rope:
            assert q_pos is not None and kv_pos is not None, "use_rope=True requires q_pos and kv_pos"
            q, k = self._apply_rope(q, q_pos), self._apply_rope(k, kv_pos) # rotate Q, K by their positions

        weights = None
        if self.return_weights:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
            attn = attn.softmax(dim=-1)
            weights = attn.squeeze(2).detach()
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, self.d_embed) # (B, H, 1, Dh) -> (B, d_embed)
        return self.out_proj(out), weights # -> (B, d_out)


class RecurrentModel(nn.Module):
    def __init__(self, cell: nn.Module):
        super().__init__()
        self.recurrent_module = cell

    def get_initial_state(self, batch_size: int = 1):
        state = self.recurrent_module.initialize_state()
        return tuple(s.expand(batch_size, *(-1,) * (s.ndim - 1)) for s in state)

    def forward(self, input, state):
        outputs = []
        for element in input:
            hidden_state, state = self.recurrent_module(element, state)
            outputs.append(hidden_state)
        return torch.stack(outputs, dim=0), state


class CrossAttentionRecurrentModel(RecurrentModel):
    def __init__(self, cell: nn.Module, num_heads: int = 4):
        super().__init__(cell)
        self.cross_attn = CrossAttention(d_embed=cell.d_input, d_out=cell.d_input, num_heads=num_heads)

    def forward(self, input, state):
        outputs = []
        for tokens in input: # (B, N, d_token) per tick
            cell_input, _ = self.cross_attn(state[-1], tokens) # State for query is assumed to be the final element of state tuple: CODE SMELL
            prediction_vector, state = self.recurrent_module(cell_input, state)
            outputs.append(prediction_vector)
        return torch.stack(outputs, dim=0), state
