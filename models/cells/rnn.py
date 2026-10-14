import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell

class RNNCell(RecurrentCell):
  def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True):
    super().__init__(d_input=d_input, d_model=d_model)
    self.h_0 = nn.Parameter(torch.zeros(d_model))  # Learnable initial hidden state
    self.weight_x = nn.Parameter(torch.empty(d_input, d_model))
    self.weight_h = nn.Parameter(torch.empty(d_model, d_model))
    self.bias = nn.Parameter(torch.empty(d_model))
    self.layer_norm = layer_norm
    self.dropout = nn.Dropout(p=dropout)
    if layer_norm:
      self.ln = nn.LayerNorm(d_model)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight_x)
    nn.init.orthogonal_(self.weight_h)
    nn.init.zeros_(self.bias)

  def initialize_state(self) -> tuple[Tensor]:
    return (self.h_0.unsqueeze(0),)

  def forward(self, input: Tensor, state: tuple[Tensor] | None) -> tuple[Tensor, tuple[Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0 = self.dropout(state[0]) if state is not None else self.h_0.unsqueeze(0)

    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.tanh(self.ln(input @ self.weight_x + self.dropout(h_0) @ self.weight_h + self.bias.unsqueeze(0)))  # h_t = tanh(LN(W_x⋅x_t + W_h⋅d(h_t−1) + b))
    else:
      h_1 = torch.tanh(input @ self.weight_x + self.dropout(h_0) @ self.weight_h + self.bias.unsqueeze(0))  # h_t = tanh(W_x⋅x_t + W_h⋅d(h_t−1) + b)
    return h_1, (h_1,)