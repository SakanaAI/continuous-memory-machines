import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.cells.base import RecurrentCell


class LSAMCell(RecurrentCell):
    def __init__(self, d_input: int, d_model: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__(d_input=d_input, d_model=d_model)
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # --- QKV and read/write probability projections ---
        self.Wqkv = nn.Linear(d_input + d_model, 3 * d_model)
        self.Wrw = nn.Linear(d_input + d_model, 2 * num_heads)

        # --- Learned initial hidden state ---
        self.h_0 = nn.Parameter(torch.zeros(d_model))

        # --- Memory (constant near-zero init) ---
        self.register_buffer("mem_bias", torch.full((num_heads, self.head_dim, self.head_dim), 1e-6))

        self.dropout_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wqkv.weight, gain=1.4)
        nn.init.normal_(self.Wqkv.bias, std=0.01)
        nn.init.xavier_uniform_(self.Wrw.weight, gain=1.0)
        nn.init.normal_(self.Wrw.bias, std=0.01)

    def initialize_state(self):
        h = self.h_0.unsqueeze(0)  # (1, d_model)
        M = self.mem_bias.unsqueeze(0)  # (1, H, D, D)
        return (h, M)

    def forward(self, input: Tensor, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.size(0)
        H, D = self.num_heads, self.head_dim

        # --- Unpack State ---
        h_prev, memory = state  # (B, d_model), (B, H, D, D)

        # --- Combined input ---
        combined = torch.cat([input, h_prev], dim=1)  # (B, d_input + d_model)

        # --- Project to q, k, v ---
        qkv = self.Wqkv(combined)  # (B, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, d_model)
        q = F.normalize(q.view(B, H, D), dim=-1)  # (B, H, D)
        k = F.normalize(k.view(B, H, D), dim=-1)  # (B, H, D)
        v = v.view(B, H, D)  # (B, H, D)

        # --- Read/Write probabilities ---
        rw = torch.sigmoid(self.Wrw(combined))  # (B, 2*H)
        pr = rw[:, :H].unsqueeze(-1)  # (B, H, 1)
        pw = rw[:, H:].unsqueeze(-1)  # (B, H, 1)

        # --- NAM Write: M' = M + pw * (v - M@k) @ k^T ---
        Mk = torch.matmul(memory, k.unsqueeze(-1)).squeeze(-1)  # (B, H, D)
        delta = v - Mk  # (B, H, D)
        update = torch.matmul(delta.unsqueeze(-1), k.unsqueeze(-2))  # (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
        memory = memory + pw.unsqueeze(-1) * update  # (B, H, D, D)

        # --- NAM Read: h = pr * M' @ q ---
        read = torch.matmul(memory, q.unsqueeze(-1)).squeeze(-1)  # (B, H, D)
        h_new = (pr * read).reshape(B, -1)  # (B, d_model)
        h_new = self.dropout_layer(h_new)

        return h_new, (h_new, memory)


class CrossAttentionLSAMCell(LSAMCell):
    """LSAM with a query state for CrossAttentionRecurrentModel compatibility."""

    def initialize_state(self):
        h, M = super().initialize_state()
        query = torch.zeros(1, self.d_model, device=h.device)  # (1, d_model)
        return (h, M, query)

    def forward(self, input: Tensor, state):  # noqa: A002
        h_new, (h, memory) = super().forward(input, (state[0], state[1]))
        return h_new, (h, memory, h_new)
