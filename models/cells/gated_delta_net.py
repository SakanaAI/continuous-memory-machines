import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.cells.base import RecurrentCell


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight).to(x.dtype)


class GatedDeltaNetCell(RecurrentCell):
    """Gated Delta Net, from "Gated Delta Networks: Improving Mamba2 with Delta Rule"."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        conv_size: int = 4,
        qk_norm: str = "l2",
        dropout: float = 0.0,
    ):
        super().__init__(d_input=d_input, d_model=d_model)
        assert qk_norm in ("l2", "softmax"), f"qk_norm must be 'l2' or 'softmax', got '{qk_norm}'"
        self.num_heads = num_heads
        self.key_dim = round(d_model * expand_k / num_heads) * num_heads
        self.value_dim = round(d_model * expand_v / num_heads) * num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.conv_size = conv_size
        self.qk_norm = qk_norm

        # --- Projections ---
        self.q_proj = nn.Linear(d_input, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_input, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_input, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_input, self.value_dim, bias=False)
        self.gk_proj = nn.Linear(d_input, num_heads, bias=False)
        self.b_proj = nn.Linear(d_input, num_heads, bias=True)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        # --- Causal conv1d weights (depthwise, one kernel per channel) ---
        self.q_conv_weight = nn.Parameter(torch.empty(self.key_dim, conv_size))
        self.k_conv_weight = nn.Parameter(torch.empty(self.key_dim, conv_size))
        self.v_conv_weight = nn.Parameter(torch.empty(self.value_dim, conv_size))

        # --- Output gate norm ---
        self.g_norm = RMSNorm(self.head_v_dim)

        # --- Mamba-style decay gating ---
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # --- Initial states (always zeros) ---
        self.register_buffer("S_0", torch.zeros(num_heads, self.head_qk_dim, self.head_v_dim))
        self.register_buffer("conv_q_0", torch.zeros(1, self.key_dim, conv_size))
        self.register_buffer("conv_k_0", torch.zeros(1, self.key_dim, conv_size))
        self.register_buffer("conv_v_0", torch.zeros(1, self.value_dim, conv_size))

        self.dropout_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.gk_proj.weight)
        nn.init.xavier_uniform_(self.b_proj.weight)
        nn.init.zeros_(self.b_proj.bias)
        for w in (self.q_conv_weight, self.k_conv_weight, self.v_conv_weight):
            nn.init.kaiming_normal_(w.unsqueeze(1))  # treat as (out, 1, k) for init, then squeeze back
            w.data.squeeze_(1) if w.ndim == 3 else None  # no-op guard

    def _conv_step(self, x: Tensor, conv_state: Tensor, conv_weight: Tensor) -> tuple[Tensor, Tensor]:
        """Single step of causal depthwise conv1d + SiLU."""
        # Stacking individual taps avoids nested slice/cat indexing, which
        # fails in PyTorch 2.6 CUDA Inductor when compiling inference sequences.
        new_state = torch.stack([*conv_state.unbind(-1)[1:], x], dim=-1)  # (B, D, K)
        out = (new_state * conv_weight).sum(-1)  # (B, D)
        return F.silu(out), new_state

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.S_0.unsqueeze(0), self.conv_q_0, self.conv_k_0, self.conv_v_0

    def _compute_beta(self, input: Tensor) -> Tensor:
        return self.b_proj(input).float().sigmoid()

    def _update_memory(self, S: Tensor, k: Tensor, v: Tensor, beta: Tensor, decay: Tensor) -> Tensor:
        """Keep the original GDN recurrence for existing experiments."""
        retrieved = torch.einsum("bhkv,bhk->bhv", S, k)
        delta = v - retrieved
        write = torch.einsum("bhk,bhv->bhkv", k, beta.unsqueeze(-1) * delta)
        return decay * S + write

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
        assert input.ndim == 2, "Input should be of size B x D"
        S, conv_q, conv_k, conv_v = state if state is not None else self.initialize_state()
        B = input.size(0)
        H, Dk, Dv = self.num_heads, self.head_qk_dim, self.head_v_dim

        # Project
        q = self.q_proj(input)  # (B, key_dim)
        k = self.k_proj(input)  # (B, key_dim)
        v = self.v_proj(input)  # (B, value_dim)

        # Causal conv1d step + SiLU
        q, conv_q = self._conv_step(q, conv_q, self.q_conv_weight)
        k, conv_k = self._conv_step(k, conv_k, self.k_conv_weight)
        v, conv_v = self._conv_step(v, conv_v, self.v_conv_weight)

        # Decay gate (Mamba-style): gk in log-space, always negative
        gk = self.gk_proj(input).float()  # (B, H)
        gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)  # (B, H)

        # Beta gate
        beta = self._compute_beta(input)  # (B, H)

        # Reshape to multi-head
        q = q.view(B, H, Dk)  # (B, H, Dk)
        k = k.view(B, H, Dk)  # (B, H, Dk)
        v = v.view(B, H, Dv)  # (B, H, Dv)

        # QK normalization
        if self.qk_norm == "l2":
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        else:
            q, k = q.softmax(dim=-1), k.softmax(dim=-1)

        # --- Gated delta rule ---
        decay = gk.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S_new = self._update_memory(S, k, v, beta, decay)  # (B, H, Dk, Dv)

        # Read
        o = torch.einsum("bhkv,bhk->bhv", S_new, q)  # (B, H, Dv)

        # Output gating: RMSNorm + SiLU gate
        g = self.g_proj(input).view(B, H, Dv)  # (B, H, Dv)
        o = self.g_norm(o) * F.silu(g)  # (B, H, Dv)

        o = self.o_proj(o.reshape(B, self.value_dim))  # (B, d_model)
        o = self.dropout_layer(o)
        return o, (S_new, conv_q, conv_k, conv_v)


class CrossAttentionGatedDeltaNetCell(GatedDeltaNetCell):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        conv_size: int = 4,
        qk_norm: str = "l2",
        dropout: float = 0.0,
    ):
        super().__init__(
            d_input=d_input,
            d_model=d_model,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            conv_size=conv_size,
            qk_norm=qk_norm,
            dropout=dropout,
        )

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        S, conv_q, conv_k, conv_v = super().initialize_state()
        query = torch.zeros(1, self.d_model, device=S.device, dtype=S.dtype)  # (1, d_model)
        return S, conv_q, conv_k, conv_v, query

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
        if state is None:
            state = self.initialize_state()
        S, conv_q, conv_k, conv_v, _ = state
        output, (S_new, conv_q, conv_k, conv_v) = super().forward(input, (S, conv_q, conv_k, conv_v))
        return output, (S_new, conv_q, conv_k, conv_v, output)


class StateTrackingGatedDeltaNetCell(GatedDeltaNetCell):
    """GDN with the extended beta range from Grazzi et al. (ICLR 2025).

    See arXiv:2411.12537, Section 4.2 and Appendix E.4.1: beta = 2 * sigmoid.
    Adapted to gated memory using the decay-before-retrieval update from
    "Gated Delta Networks" (arXiv:2412.06464):

        S_new = alpha * (I - beta * k k^T) @ S + beta * k v^T.

    For unit L2-normalized keys, the transition eigenvalues are alpha and
    alpha * (1 - beta), allowing negative values while keeping their absolute
    values <= 1. Softmax keys retain the bound but need not have unit norm,
    so use qk_norm="l2" to access the full extended range. The decay gate
    remains learned; this adapts the paper's DeltaNet change to GDN.
    """

    def _compute_beta(self, input: Tensor) -> Tensor:
        return 2.0 * super()._compute_beta(input)

    def _update_memory(self, S: Tensor, k: Tensor, v: Tensor, beta: Tensor, decay: Tensor) -> Tensor:
        # Decay before retrieval: using the old S here would instead give
        # alpha * I - beta * k k^T, whose eigenvalues can fall below -1.
        decayed_state = decay * S
        retrieved = torch.einsum("bhkv,bhk->bhv", decayed_state, k)
        write = torch.einsum("bhk,bhv->bhkv", k, beta.unsqueeze(-1) * (v - retrieved))
        return decayed_state + write


class CrossAttentionStateTrackingGatedDeltaNetCell(
    CrossAttentionGatedDeltaNetCell, StateTrackingGatedDeltaNetCell
):
    """Reuse the cross-attention state/query wrapper with the extended update."""
