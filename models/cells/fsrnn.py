from torch import Tensor, nn

from models.cells.base import RecurrentCell
from models.cells.lstm import LSTMCell


# Fast-Slow topology from "Fast-Slow Recurrent Neural Networks", Section 3 (https://arxiv.org/abs/1705.08639)
# Reuse the local LSTM's layer norm, recurrent dropout, and initialization for comparisons with the LSTM baseline.
class FSRNNCell(RecurrentCell):
    def __init__(self, d_input: int, d_model: int, fast_layers: int = 2, d_slow: int | None = None,
                 dropout: float = 0.0, layer_norm: bool = True, T_max: int | None = None):
        super().__init__(d_input=d_input, d_model=d_model)
        assert fast_layers >= 2, "At least two fast layers are needed"
        assert d_input > 0 and d_model > 0, "Input and fast cell sizes should be positive"
        assert d_slow is None or d_slow > 0, "Slow cell size should be positive"
        self.fast_layers, self.d_slow = fast_layers, d_model if d_slow is None else d_slow
        self.layer_norm, self.T_max = layer_norm, T_max

        input_sizes = [d_input, self.d_slow] + [0] * (fast_layers - 2)
        self.fast_cells = nn.ModuleList([
            LSTMCell(size, d_model, dropout=dropout, layer_norm=layer_norm, T_max=T_max, learn_initial_state=index == 0)
            for index, size in enumerate(input_sizes)
        ])  # Only F1 owns the initial fast state; later transitions receive the preceding fast (h, c).
        self.slow_cell = LSTMCell(d_model, self.d_slow, dropout=dropout, layer_norm=layer_norm, T_max=T_max)

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (*self.fast_cells[0].initialize_state(), *self.slow_cell.initialize_state())

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        assert input.ndim == 2, "Input should be of size B x D"
        h_fast, c_fast, h_slow, c_slow = state if state is not None else self.initialize_state()

        fast_output, fast_state = self.fast_cells[0](input, (h_fast, c_fast))  # F1: previous final fast state + current input
        slow_output, slow_state = self.slow_cell(fast_output, (h_slow, c_slow))  # S: one update per input, driven by F1
        fast_output, fast_state = self.fast_cells[1](slow_output, fast_state)  # F2: continue F1's (h, c), with the new slow hidden state as input
        empty_input = input.new_empty(input.size(0), 0)
        for cell in self.fast_cells[2:]:
            fast_output, fast_state = cell(empty_input, fast_state)  # F3...Fk: distinct transitions of the same fast state, with no external input

        return fast_output, (*fast_state, *slow_state)


class CrossAttentionFSRNNCell(FSRNNCell):
    def __init__(self, d_input: int, d_model: int, fast_layers: int = 2, d_slow: int | None = None,
                 dropout: float = 0.0, layer_norm: bool = True, T_max: int | None = None):
        super().__init__(d_input=d_input, d_model=d_model, fast_layers=fast_layers, d_slow=d_slow,
                         dropout=dropout, layer_norm=layer_norm, T_max=T_max)
        self.query_proj = nn.Linear(d_model, d_input)

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state = super().initialize_state()
        query = state[0].new_zeros(1, self.d_input)
        return (*state, query)

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        state = self.initialize_state() if state is None else state
        output, recurrent_state = super().forward(input, state[:4])
        query = self.query_proj(recurrent_state[0])  # Final fast hidden state supplies the next patch-attention query
        return output, (*recurrent_state, query)
