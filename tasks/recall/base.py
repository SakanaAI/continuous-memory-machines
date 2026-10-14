from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from models import construct_rnn, construct_backbone
from tasks.task import Task
from tasks.utils import get_device


class RecallModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, d_output: int):
        super().__init__()
        self.backbone = construct_backbone(model_cfg.backbone, model_cfg.d_input)
        if model_cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.rnn = construct_rnn(model_cfg)
        self.output_proj = nn.Linear(model_cfg.d_model, d_output)

    def forward(self, input, states):
        assert input.ndim == 3, "Input should be of size B x T x D_raw"

        features = self.backbone(input)  # (B, T, D_raw) -> (B, T, d_input)
        features = features.transpose(0, 1)  # (B, T, d_input) -> (T, B, d_input)

        hidden, final_state = self.rnn(features, state=states)  # (T, B, d_input) -> (T, B, d_model)
        logits = self.output_proj(hidden)  # (T, B, d_model) -> (T, B, d_output)
        logits = logits.transpose(0, 1)  # (T, B, d_output) -> (B, T, d_output)

        return logits, final_state

    def get_initial_states(self, batch_size):
        return self.rnn.get_initial_state(batch_size)


class BaseRecallTask(Task):
    _performance_metric_key: str
    _performance_higher_is_better: bool

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = get_device(cfg)

        self.setup_dataset(cfg)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model = self.build_model(cfg).to(self.device)
        self.maybe_compile(cfg)
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        if cfg.training.use_scheduler:
            warmup_epochs = cfg.training.warmup_epochs
            warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-2, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs - warmup_epochs)
            self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        else:
            self.scheduler = None
        self.max_grad_norm = cfg.training.max_grad_norm

    @abstractmethod
    def setup_dataset(self, cfg: DictConfig):
        """Subclasses must set self.train_dataset, self.val_dataset, self.test_dataset."""

    @abstractmethod
    def d_output(self, cfg: DictConfig) -> int:
        """Output projection size for the recall model."""

    @abstractmethod
    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Loss to backprop, given (B, T, d_output) logits and task-specific targets."""

    @abstractmethod
    def _compute_metrics(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        """Per-batch scalar metrics (must include 'loss')."""

    @abstractmethod
    def _pseudo_input_shape(self, cfg: DictConfig) -> tuple[int, ...]:
        """Return (T, D_raw) shape used to materialize lazy modules without touching the dataset."""

    def build_model(self, cfg: DictConfig) -> nn.Module:
        model = RecallModel(cfg.model, d_output=self.d_output(cfg))
        # Materialize lazy modules with a pseudo input on the current default device
        # (CPU at __init__ time, meta during the max_parameters binary search)
        pseudo_inputs = torch.empty((1, *self._pseudo_input_shape(cfg)))
        initial_states = model.get_initial_states(batch_size=1)
        with torch.no_grad():
            model(pseudo_inputs, states=initial_states)
        return model

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accumulator = defaultdict(list)
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
            logits, _ = self.model(inputs, states=initial_states)  # (B, T, d_output)
            loss = self._compute_loss(logits, targets)

            loss.backward()
            grad_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None)
            )
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            batch_metrics = self._compute_metrics(logits, targets, loss)
            batch_metrics["grad_norm"] = grad_norm.item()
            for k, v in batch_metrics.items():
                accumulator[k].append(v)
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in batch_metrics.items() if k != "grad_norm"})

        if self.scheduler is not None:
            self.scheduler.step()
        metrics = {k: sum(v) / len(v) for k, v in accumulator.items()}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        self.model.eval()
        accumulator = defaultdict(list)
        for inputs, targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
                logits, _ = self.model(inputs, states=initial_states)
                loss = self._compute_loss(logits, targets)
                batch_metrics = self._compute_metrics(logits, targets, loss)
                for k, v in batch_metrics.items():
                    accumulator[k].append(v)
        return {k: sum(v) / len(v) for k, v in accumulator.items()}

    def validate(self, epoch: int) -> dict[str, float]:
        return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
        return self._evaluate(self.test_dataloader, desc)

    def calculate_performance(self, metrics: dict[str, list], window_size: int = 5) -> float:
        if self._performance_metric_key not in metrics:
            return 0.0
        values = [pair[1] for pair in metrics[self._performance_metric_key]]
        recent = values[-window_size:]
        if not recent:
            return 0.0
        mean = sum(recent) / len(recent)
        return mean if self._performance_higher_is_better else -mean
