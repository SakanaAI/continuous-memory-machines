import torch
from torch import Tensor, nn


class Bounded(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: Tensor) -> Tensor:
        return (self.high - self.low) * torch.sigmoid(x) + self.low

    def right_inverse(self, y: Tensor) -> Tensor:
        y_normalised = torch.clamp((y - self.low) / (self.high - self.low), 1e-6, 1 - 1e-6)
        return torch.logit(y_normalised)
