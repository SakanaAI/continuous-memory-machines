import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell


class RelationalMemoryCore(RecurrentCell):
    """Relational Memory Core, from "Relational Recurrent Neural Networks"."""

    def __init__(self, d_input: int, d_model: int, mem_slots: int = 8, num_heads: int = 4, head_size: int | None = None,
                 num_blocks: int = 1, gate_style: str | None = "unit", forget_bias: float = 1.0, input_bias: float = 0.0,
                 attention_mlp_layers: int = 2, key_size: int | None = None, dropout: float = 0.0):
        super().__init__(d_input=d_input, d_model=d_model)
        assert gate_style in ("unit", "memory", None), f"gate_style must be 'unit', 'memory', or None, got {gate_style!r}"
        assert num_blocks >= 1, f"num_blocks must be >= 1, got {num_blocks}"
        assert attention_mlp_layers >= 1, f"attention_mlp_layers must be >= 1, got {attention_mlp_layers}"
        assert 0 <= dropout < 1, "dropout must be in [0, 1)"

        self.mem_slots, self.num_heads = mem_slots, num_heads
        self.head_size = head_size if head_size is not None else d_model // num_heads
        self.key_size = key_size if key_size is not None else self.head_size
        self.mem_size = self.head_size * num_heads
        self.num_blocks, self.gate_style = num_blocks, gate_style
        self.forget_bias, self.input_bias = forget_bias, input_bias
        self.attention_mlp_layers = attention_mlp_layers

        qkv_per_head = 2 * self.key_size + self.head_size
        total_qkv = qkv_per_head * num_heads

        self.input_proj = nn.Linear(d_input, self.mem_size)  # x: (B, d_input) -> (B, mem_size)

        self.blocks = nn.ModuleList([nn.ModuleDict({
            "qkv_proj": nn.Linear(self.mem_size, total_qkv),
            "qkv_ln": nn.LayerNorm(total_qkv),  # Reference applies LayerNorm on QKV projection output before splitting
            "ln_attn": nn.LayerNorm(self.mem_size),
            "mlp": nn.Sequential(*[layer for _ in range(attention_mlp_layers)
                                   for layer in (nn.Linear(self.mem_size, self.mem_size), nn.ReLU())][:-1]),  # drop trailing ReLU
            "ln_mlp": nn.LayerNorm(self.mem_size),
        }) for _ in range(num_blocks)])

        if gate_style is not None:
            gate_size = self.mem_size if gate_style == "unit" else 1
            num_gates = 2 * gate_size
            self.gate_in_proj = nn.Linear(d_input, num_gates)
            self.gate_mem_proj = nn.Linear(self.mem_size, num_gates)
            self.gate_size = gate_size

        self.output_proj = nn.Linear(mem_slots * self.mem_size, d_model)
        self.dropout_layer = nn.Dropout(p=dropout)

        # Initial memory: identity matrix (mem_slots x mem_slots) padded/truncated to mem_size cols, faithful to reference
        M_0 = torch.eye(mem_slots)  # (N, N)
        if self.mem_size > mem_slots:
            M_0 = torch.cat([M_0, torch.zeros(mem_slots, self.mem_size - mem_slots)], dim=-1)
        elif self.mem_size < mem_slots:
            M_0 = M_0[:, :self.mem_size]
        self.register_buffer("M_0", M_0)  # (N, F)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _multihead_attention(self, memory: Tensor, block: nn.ModuleDict) -> Tensor:
        B, N, _ = memory.shape  # (B, N, F)
        H, Dk, Dv = self.num_heads, self.key_size, self.head_size

        qkv = block["qkv_ln"](block["qkv_proj"](memory))  # (B, N, H*(2Dk+Dv))
        qkv = qkv.view(B, N, H, 2 * Dk + Dv).transpose(1, 2)  # (B, H, N, 2Dk+Dv)
        q, k, v = qkv.split([Dk, Dk, Dv], dim=-1)  # (B, H, N, Dk), (B, H, N, Dk), (B, H, N, Dv)

        q = q * (Dk ** -0.5)
        weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # (B, H, N, N)
        out = weights @ v  # (B, H, N, Dv)
        return out.transpose(1, 2).reshape(B, N, H * Dv)  # (B, N, F)

    def _attend_over_memory(self, memory: Tensor) -> Tensor:
        for block in self.blocks:
            attended = self._multihead_attention(memory, block)
            memory = block["ln_attn"](memory + attended)
            memory = block["ln_mlp"](memory + block["mlp"](memory))
        return memory

    def _create_gates(self, x: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        gate_in = self.gate_in_proj(x).unsqueeze(1)  # (B, 1, num_gates)
        gate_mem = self.gate_mem_proj(torch.tanh(memory))  # (B, N, num_gates)
        gates = gate_in + gate_mem  # broadcasts over slot dim if gate_in is per-input
        input_gate, forget_gate = gates.chunk(2, dim=-1)  # each (B, N, gate_size)
        return torch.sigmoid(input_gate + self.input_bias), torch.sigmoid(forget_gate + self.forget_bias)

    def initialize_state(self) -> tuple[Tensor]:
        return (self.M_0.unsqueeze(0),)  # (1, N, F)

    def forward(self, input: Tensor, state: tuple[Tensor] | None) -> tuple[Tensor, tuple[Tensor]]:  # noqa: A002
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.size(0)
        M = (state[0] if state is not None else self.initialize_state()[0])  # (B or 1, N, F)
        if M.size(0) != B:
            M = M.expand(B, -1, -1)

        x_proj = self.input_proj(input).unsqueeze(1)  # (B, 1, F)
        M_plus = torch.cat([M, x_proj], dim=1)  # (B, N+1, F)
        attended = self._attend_over_memory(M_plus)  # (B, N+1, F)
        M_tilde = attended[:, :-1, :]  # (B, N, F) — strip the input row

        if self.gate_style is not None:
            input_gate, forget_gate = self._create_gates(input, M)  # each (B, N, gate_size or 1)
            M_new = input_gate * torch.tanh(M_tilde) + forget_gate * M
        else:
            M_new = M_tilde

        out = self.output_proj(M_new.reshape(B, self.mem_slots * self.mem_size))  # (B, d_model)
        out = self.dropout_layer(out)
        return out, (M_new,)


class CrossAttentionRelationalMemoryCore(RelationalMemoryCore):
    """RMC with a separate memory-to-query projection for patch attention."""

    # Explicit arguments let construct_rnn discover and forward model overrides.
    def __init__(self, d_input: int, d_model: int, mem_slots: int = 8, num_heads: int = 4, head_size: int | None = None,
                 num_blocks: int = 1, gate_style: str | None = "unit", forget_bias: float = 1.0, input_bias: float = 0.0,
                 attention_mlp_layers: int = 2, key_size: int | None = None, dropout: float = 0.0):
        super().__init__(
            d_input=d_input, d_model=d_model, mem_slots=mem_slots, num_heads=num_heads, head_size=head_size,
            num_blocks=num_blocks, gate_style=gate_style, forget_bias=forget_bias, input_bias=input_bias,
            attention_mlp_layers=attention_mlp_layers, key_size=key_size, dropout=dropout,
        )
        self.query_proj = nn.Linear(self.mem_slots * self.mem_size, self.d_input)

    def initialize_state(self) -> tuple[Tensor, Tensor]:
        M_0 = super().initialize_state()[0]  # (1, N, F)
        query = torch.zeros(1, self.d_input, device=M_0.device, dtype=M_0.dtype)  # (1, d_input)
        return (M_0, query)

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
        if state is not None:
            state = (state[0],)  # strip query, pass only M to parent
        out, (M_new,) = super().forward(input, state)
        query = self.query_proj(M_new.flatten(1))  # (B, N*F) -> (B, d_input)
        return out, (M_new, query)
