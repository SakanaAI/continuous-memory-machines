import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from omegaconf import DictConfig

from tasks.recall.base import BaseRecallTask


class PrioritySortDataset(Dataset):
    def __init__(self, value_width: int, num_input_items: int, num_output_items: int, num_samples: int):
        self.value_width, self.num_input_items = value_width, num_input_items
        self.num_output_items, self.num_samples = num_output_items, num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # A length-aware batch sampler can pass ``(sample_index, N)`` so every
        # item in a batch has the same shape while N varies between batches.
        N = int(index[1]) if isinstance(index, tuple) else self.num_input_items
        W, K = self.value_width, self.num_output_items
        if K > N:
            raise ValueError(f"num_output_items ({K}) must not exceed num_input_items ({N})")
        total_steps = N + K

        priorities = torch.empty(N).uniform_(-1, 1)  # (N,)
        values = torch.bernoulli(torch.full((N, W), 0.5))  # (N, W)

        sorted_indices = priorities.argsort()
        sorted_values = values[sorted_indices[:K]]  # (K, W) lowest-priority first

        # Input: [priority | value(W) | input_flag | output_flag]
        input = torch.zeros(total_steps, W + 3)
        input[:N, 0] = priorities  # priority channel
        input[:N, 1:W+1] = values  # value channels
        input[:N, W+1] = 1.0  # input_flag
        input[N:, W+2] = 1.0  # output_flag

        target = torch.zeros(total_steps, W)
        target[N:] = sorted_values

        return input, target


class VariableLengthBatchSampler(Sampler):
    """Yield batches whose examples share one balanced, variable input length."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        min_num_input_items: int,
        max_num_input_items: int,
        seed: int,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if min_num_input_items <= 0:
            raise ValueError("min_num_input_items must be positive")
        if max_num_input_items < min_num_input_items:
            raise ValueError("max_num_input_items must be >= min_num_input_items")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.min_num_input_items = min_num_input_items
        self.max_num_input_items = max_num_input_items
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sample_indices = torch.randperm(self.dataset_size, generator=generator).tolist()

        possible_lengths = torch.arange(self.min_num_input_items, self.max_num_input_items + 1)
        complete_cycles, remainder = divmod(len(self), len(possible_lengths))
        batch_lengths = possible_lengths.repeat(complete_cycles)
        if remainder:
            remainder_order = torch.randperm(len(possible_lengths), generator=generator)[:remainder]
            batch_lengths = torch.cat([batch_lengths, possible_lengths[remainder_order]])
        permutation = torch.randperm(batch_lengths.numel(), generator=generator)
        batch_lengths = batch_lengths[permutation].tolist()

        for batch_index, num_input_items in enumerate(batch_lengths):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            yield [(sample_indices[i], num_input_items) for i in range(start, stop)]


class PrioritySortTask(BaseRecallTask):
    _performance_metric_key = "eval_loss"
    _performance_higher_is_better = False

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # ``get`` keeps checkpoints created before these options were added
        # loadable for evaluation and resumption.
        train_min = cfg.task.get("train_min_num_input_items")
        train_max = cfg.task.get("train_max_num_input_items")
        train_min = self.num_input_items if train_min is None else int(train_min)
        train_max = self.num_input_items if train_max is None else int(train_max)
        if self.num_output_items > train_min:
            raise ValueError(
                f"num_output_items ({self.num_output_items}) must not exceed "
                f"train_min_num_input_items ({train_min})"
            )

        self.variable_length_batch_sampler = None
        if train_min != self.num_input_items or train_max != self.num_input_items:
            self.variable_length_batch_sampler = VariableLengthBatchSampler(
                dataset_size=len(self.train_dataset),
                batch_size=cfg.training.batch_size,
                min_num_input_items=train_min,
                max_num_input_items=train_max,
                seed=cfg.seed,
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=self.variable_length_batch_sampler,
                num_workers=0,
            )

    def train(self, epoch: int) -> dict[str, float]:
        if self.variable_length_batch_sampler is not None:
            self.variable_length_batch_sampler.set_epoch(epoch)
        return super().train(epoch)

    def setup_dataset(self, cfg: DictConfig):
        self.value_width = cfg.task.value_width
        self.num_input_items = cfg.task.num_input_items
        self.num_output_items = cfg.task.num_output_items

        kwargs = dict(
            value_width=cfg.task.value_width,
            num_input_items=cfg.task.num_input_items,
            num_output_items=cfg.task.num_output_items,
            num_samples=cfg.task.num_samples,
        )
        self.train_dataset = PrioritySortDataset(**kwargs)
        self.val_dataset = PrioritySortDataset(**kwargs)
        self.test_dataset = PrioritySortDataset(**kwargs)

    def d_output(self, cfg: DictConfig) -> int:
        return cfg.task.value_width

    def _pseudo_input_shape(self, cfg: DictConfig) -> tuple[int, int]:
        return (cfg.task.num_input_items + cfg.task.num_output_items, cfg.task.value_width + 3)

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            logits[:, -self.num_output_items:, :],
            targets[:, -self.num_output_items:, :],
        )

    def _compute_metrics(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        preds = (torch.sigmoid(logits[:, -self.num_output_items:, :]) > 0.5).float()
        tgts = targets[:, -self.num_output_items:, :]
        bit_accuracy = (preds == tgts).float().mean().item()
        vector_accuracy = (preds == tgts).all(dim=-1).float().mean().item()
        return {"loss": loss.item(), "bit_accuracy": bit_accuracy, "vector_accuracy": vector_accuracy}
