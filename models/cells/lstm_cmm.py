import torch
from torch import Tensor, nn

from models.cells.lstm import LSTMCell
from models.cells.cmm import MemoryUpdater


class LSTMCMMCell(LSTMCell):
    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, T_max: int | None = None,
                 num_long_term_memory_tokens: int = 16, memory_transformer_layers: int = 1, num_heads: int = 4, memory_update_gate: bool = False, memory_update_gate_init: float = 0.0):
        super().__init__(d_input=d_input, d_model=d_model, dropout=dropout, layer_norm=layer_norm, T_max=T_max)
        self.num_long_term_memory_tokens = num_long_term_memory_tokens
        self.memory_updater = MemoryUpdater(d_model, num_ltm_tokens=num_long_term_memory_tokens, num_stm_tokens=1, num_layers=memory_transformer_layers, num_heads=num_heads, dropout=dropout, memory_update_gate=memory_update_gate, memory_update_gate_init=memory_update_gate_init)
        self.start_ltm = nn.Parameter(torch.randn(num_long_term_memory_tokens, d_model) * 0.01) # (L, d_model)

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor]:
        h_0, c_0 = super().initialize_state()
        ltm = self.start_ltm.unsqueeze(0) # (1, L, d_model)
        return h_0, c_0, ltm

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:  # noqa: A002
        assert input.ndim == 2, "Input should be of size B x D"
        h_0, c_0, ltm = state if state is not None else self.initialize_state()

        h_1, (_, c_1) = super().forward(input, (h_0, c_0))

        h_1_mixed, ltm = self.memory_updater(h_1.unsqueeze(1), ltm) # (B, 1, d_model), (B, L, d_model)
        h_1_mixed = h_1_mixed.squeeze(1) # (B, 1, d_model) -> (B, d_model)

        return h_1_mixed, (h_1, c_1, ltm)


class CrossAttentionLSTMCMMCell(LSTMCMMCell):
    # Explicit arguments let construct_rnn discover and forward model overrides.
    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, T_max: int | None = None,
                 num_long_term_memory_tokens: int = 16, memory_transformer_layers: int = 1, num_heads: int = 4,
                 memory_update_gate: bool = False, memory_update_gate_init: float = 0.0):
        super().__init__(
            d_input=d_input, d_model=d_model, dropout=dropout, layer_norm=layer_norm, T_max=T_max,
            num_long_term_memory_tokens=num_long_term_memory_tokens, memory_transformer_layers=memory_transformer_layers,
            num_heads=num_heads, memory_update_gate=memory_update_gate, memory_update_gate_init=memory_update_gate_init,
        )
        # Keep the attention query in the input/token space.  Besides avoiding
        # an unnecessarily wide lazy q_proj when d_model >> d_input, this is
        # the layout used by existing LSTM-CMM checkpoints.
        self.query_proj = nn.Linear(self.d_model, self.d_input)

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_0, c_0, ltm = super().initialize_state()
        query = torch.zeros(1, self.d_input, device=h_0.device, dtype=h_0.dtype) # (1, d_input)
        return h_0, c_0, ltm, query

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
        recurrent_state = None if state is None else (state[0], state[1], state[2])
        output, (h_1, c_1, ltm) = super().forward(input, recurrent_state)
        query = self.query_proj(output) # (B, d_model) -> (B, d_input)
        return output, (h_1, c_1, ltm, query)
