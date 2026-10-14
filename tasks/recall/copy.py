import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from tasks.recall.base import BaseRecallTask


class CopyDataset(Dataset):
    def __init__(self, dim: int, min_seq_length: int, max_seq_length: int, delay: int, num_samples: int, binary: bool = False):
        self.dim = dim
        self.min_seq_length, self.max_seq_length = min_seq_length, max_seq_length
        self.delay, self.num_samples, self.binary = delay, num_samples, binary

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        T = torch.randint(self.min_seq_length, self.max_seq_length + 1, ()).item()
        max_T = self.max_seq_length
        total = 2 * max_T + self.delay

        seq = torch.randint(0, 2, (T, self.dim)).float() if self.binary else torch.empty((T, self.dim)).uniform_(-1, 1)  # (T, D)

        input = torch.zeros((total, self.dim + 2))  # (2*max_T+delay, D+2)
        input[:T, :self.dim] = seq  # sample phase: data
        input[:T, self.dim] = 1  # sample_flag=1
        input[T + self.delay:T + self.delay + T, -1] = 1  # query_flag=1

        target = torch.zeros((total, self.dim + 1))  # (2*max_T+delay, D+1); last channel = loss mask
        target[T + self.delay:T + self.delay + T, :self.dim] = seq  # query-phase targets
        target[T + self.delay:T + self.delay + T, -1] = 1  # mask=1 for query phase

        return input, target


class CopyTask(BaseRecallTask):
    _performance_metric_key = "eval_loss"
    _performance_higher_is_better = False

    def setup_dataset(self, cfg: DictConfig):
        self.copy_dim = cfg.task.copy_dim
        self.copy_max_sequence_length = cfg.task.copy_max_sequence_length
        self.copy_delay = cfg.task.copy_delay
        self.copy_binary = cfg.task.copy_binary

        kwargs = dict(
            dim=cfg.task.copy_dim,
            min_seq_length=cfg.task.copy_min_sequence_length,
            max_seq_length=cfg.task.copy_max_sequence_length,
            delay=cfg.task.copy_delay,
            num_samples=cfg.task.num_samples,
            binary=cfg.task.copy_binary,
        )
        self.train_dataset = CopyDataset(**kwargs)
        self.val_dataset = CopyDataset(**kwargs)
        self.test_dataset = CopyDataset(**kwargs)
        self.fixed_length_val_dataloaders = {
            seq_length: DataLoader(
                CopyDataset(
                    dim=cfg.task.copy_dim,
                    min_seq_length=seq_length,
                    max_seq_length=seq_length,
                    delay=cfg.task.copy_delay,
                    num_samples=cfg.task.num_samples,
                    binary=cfg.task.copy_binary,
                ),
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=0,
            )
            for seq_length in cfg.task.copy_eval_sequence_lengths
        }

    def validate(self, epoch: int) -> dict[str, float]:
        metrics = super().validate(epoch)
        for seq_length, dataloader in self.fixed_length_val_dataloaders.items():
            fixed_metrics = self._evaluate(dataloader, f"Val Length {seq_length} Epoch {epoch}")
            metrics.update({f"{name}_len_{seq_length}": value for name, value in fixed_metrics.items()})
        return metrics

    def d_output(self, cfg: DictConfig) -> int:
        return cfg.task.copy_dim

    def _pseudo_input_shape(self, cfg: DictConfig) -> tuple[int, int]:
        return (2 * cfg.task.copy_max_sequence_length + cfg.task.copy_delay, cfg.task.copy_dim + 2)

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        mask = targets[:, :, -1]  # (B, T)
        targets = targets[:, :, :-1]  # (B, T, D)
        if self.copy_binary:
            per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, T, D)
        else:
            per_elem = (logits - targets) ** 2  # (B, T, D)
        masked = per_elem * mask.unsqueeze(-1)  # (B, T, D)
        return masked.sum() / (mask.sum() * logits.size(-1))

    def _compute_metrics(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        metrics = {"loss": loss.item()}
        if self.copy_binary:
            mask = targets[:, :, -1]  # (B, T)
            targets = targets[:, :, :-1]  # (B, T, D)
            preds = (logits > 0).float()  # (B, T, D)
            correct = ((preds == targets) * mask.unsqueeze(-1)).sum()
            total = mask.sum() * logits.size(-1)
            metrics["bit_accuracy"] = (correct / total).item()
        return metrics
