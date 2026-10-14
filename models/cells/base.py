from abc import abstractmethod
import torch.nn as nn
from torch import Tensor

class RecurrentCell(nn.Module):
    """Base class for recurrent neural network cells."""

    def __init__(self, d_input: int, d_model: int):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model

    @abstractmethod
    def initialize_state(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: Tensor, state):
        raise NotImplementedError
