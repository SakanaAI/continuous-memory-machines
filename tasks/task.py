from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn


class Task(ABC):
	def maybe_compile(self, cfg: DictConfig):
		if cfg.training.compile:
			print("Compiling model with torch.compile(mode='reduce-overhead')...")
			self.model = torch.compile(self.model, mode="reduce-overhead")
			print("Compiled! (first forward pass will trigger actual compilation)")
	@abstractmethod
	def build_model(self, cfg: DictConfig) -> nn.Module:
		pass

	@abstractmethod
	def train(self, epoch: int) -> dict[str, float]:
		pass

	@abstractmethod
	def validate(self, epoch: int) -> dict[str, float]:
		pass

	def eval(self, epoch: int) -> dict[str, float]:
		"""Legacy method name for backward compatibility. Use validate() instead."""
		return self.validate(epoch)

	@abstractmethod
	def calculate_performance(self, metrics: dict[str, float]) -> float:
		pass

	def test(self, epoch: int = None) -> dict[str, float]:
		return None

	def get_parameter_count(self) -> int:
		return sum(param.numel() for param in self.model.parameters())

	def get_backbone_parameter_count(self) -> int:
		return sum(param.numel() for param in self.model.backbone.parameters())

	def perform_diagnostics(self):
		pass
