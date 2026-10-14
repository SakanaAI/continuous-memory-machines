import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell


# Long short-term memory, from "Long Short-Term Memory", with forget gate, from "Learning to Forget: Continual Prediction with LSTM"
class LSTMCell(RecurrentCell):
    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, T_max: int | None = None, learn_initial_state: bool = True):
        super().__init__(d_input=d_input, d_model=d_model)
        assert 0 <= dropout < 1, "Dropout should be >= 0 and < 1"
        assert T_max is None or T_max >= 2, "T_max should be >= 2 if set"
        self.layer_norm, self.T_max = layer_norm, T_max
        if learn_initial_state:
            self.h_0, self.c_0 = nn.Parameter(torch.zeros(d_model)), nn.Parameter(torch.zeros(d_model))  # Learnable initial hidden states
        else:
            self.register_buffer("h_0", torch.zeros(d_model))
            self.register_buffer("c_0", torch.zeros(d_model))  # Internal transitions receive an existing state, so need no learned initial state
        self.weight_h = nn.Parameter(torch.empty(d_model, 4 * d_model))
        self.weight_x = nn.Parameter(torch.empty(d_input, 4 * d_model)) if d_input else None
        self.bias = nn.Parameter(torch.empty(4 * d_model))
        self.dropout = nn.Dropout(p=dropout)  # Recurrent dropout for hidden state updates, from "Recurrent Dropout without Memory Loss"
        if layer_norm:
            self.ln_1, self.ln_2, self.ln_3 = nn.LayerNorm(4 * d_model), nn.LayerNorm(4 * d_model) if d_input else None, nn.LayerNorm(d_model)  # LayerNorm configuration from "Layer Normalization"
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_x is not None:
            nn.init.orthogonal_(self.weight_x)
        nn.init.orthogonal_(self.weight_h)
        nn.init.zeros_(self.bias)
        if self.T_max:
            bias_f = torch.log(torch.zeros(self.d_model).uniform_(1, self.T_max - 1))  # If T_max is given, use chrono initialisation from "Can recurrent neural networks warp time?"
            self.bias.data[: self.d_model], self.bias.data[self.d_model : 2 * self.d_model] = bias_f, -bias_f
        else:
            nn.init.ones_(self.bias[: self.d_model])  # Initialise high forget gate bias, from "An Empirical Exploration of Recurrent Network Architectures"

    def initialize_state(self) -> tuple[Tensor, Tensor]:
        return self.h_0.unsqueeze(0), self.c_0.unsqueeze(0)

    def forward(self, input: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
        assert input.ndim == 2, "Input should be of size B x D"
        h_0, c_0 = state if state is not None else self.initialize_state()

        if self.d_input == 0:  # State-only transition, used by FSRNN's additional fast cells
            gates = self.ln_1(h_0 @ self.weight_h) if self.layer_norm else h_0 @ self.weight_h
            gates = gates.expand(input.size(0), -1) + self.bias.unsqueeze(0)
        elif self.layer_norm:  # noqa: SIM108
            gates = self.ln_1(h_0 @ self.weight_h) + self.ln_2(input @ self.weight_x) + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = LN(W_h⋅h_t−1; α_1, β_1) + LN(W_x⋅x_t; α_2, β_2) + b
        else:
            gates = h_0 @ self.weight_h + input @ self.weight_x + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = W_h⋅h_t−1 + W_x⋅x_t + b
        f, i, o, g = [fn(x) for fn, x in zip((torch.sigmoid, torch.sigmoid, torch.sigmoid, torch.tanh), gates.chunk(4, dim=1), strict=True)]  # f_t, i_t, o_t, g_t = σ(f_t), σ(i_t), σ(o_t), tanh(g_t)
        c_1 = f * c_0 + i * self.dropout(g)  # c_t = f_t ⊙ c_t−1 + i_t ⊙ d(g_t)
        h_1 = o * torch.tanh(self.ln_3(c_1) if self.layer_norm else c_1)  # h_t = o_t ⊙ tanh(LN(c_t; α_3, β_3)) [or tanh(c_t) without LN]
        return h_1, (h_1, c_1)
