import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from tasks.recall.base import BaseRecallTask
from tasks.recall.priority_sort import VariableLengthBatchSampler


class ParityDataset(Dataset):
    """Binary parity followed by a query token.

    Inputs have two channels. The first contains the input bit and the second
    marks the final query step. The target is one when the number of set bits
    is odd and zero when it is even.
    """

    def __init__(self, seq_length: int, num_samples: int):
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seq_length = int(index[1]) if isinstance(index, tuple) else self.seq_length
        bits = torch.randint(0, 2, (seq_length,), dtype=torch.float32)

        inputs = torch.zeros(seq_length + 1, 2)
        inputs[:seq_length, 0] = bits
        inputs[-1, 1] = 1.0
        target = bits.sum().remainder(2)
        return inputs, target


class ParityTask(BaseRecallTask):
    _performance_metric_key = "eval_accuracy"
    _performance_higher_is_better = True

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.variable_length_batch_sampler = VariableLengthBatchSampler(
            dataset_size=len(self.train_dataset),
            batch_size=cfg.training.batch_size,
            min_num_input_items=cfg.task.train_min_seq_length,
            max_num_input_items=cfg.task.train_max_seq_length,
            seed=cfg.seed,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.variable_length_batch_sampler,
            num_workers=0,
        )

    def setup_dataset(self, cfg: DictConfig):
        self.train_min_seq_length = int(cfg.task.train_min_seq_length)
        self.train_max_seq_length = int(cfg.task.train_max_seq_length)
        if self.train_min_seq_length <= 0:
            raise ValueError("train_min_seq_length must be positive")
        if self.train_max_seq_length < self.train_min_seq_length:
            raise ValueError("train_max_seq_length must be >= train_min_seq_length")

        train_samples = int(cfg.task.num_samples)
        eval_samples = int(cfg.task.get("eval_num_samples", train_samples))
        self.train_dataset = ParityDataset(self.train_max_seq_length, train_samples)
        self.val_dataset = ParityDataset(self.train_max_seq_length, eval_samples)
        self.test_dataset = ParityDataset(self.train_max_seq_length, eval_samples)
        self.test_dataloaders = {
            int(length): DataLoader(
                ParityDataset(int(length), eval_samples),
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=0,
            )
            for length in cfg.task.test_seq_lengths
        }

    def train(self, epoch: int) -> dict[str, float]:
        self.variable_length_batch_sampler.set_epoch(epoch)
        return super().train(epoch)

    def test(self, epoch: int = None) -> dict[str, float]:
        results = {}
        losses, accuracies = [], []
        for length, dataloader in self.test_dataloaders.items():
            metrics = self._evaluate(dataloader, f"Test Length {length}")
            losses.append(metrics["loss"])
            accuracies.append(metrics["accuracy"])
            results[f"loss_len_{length}"] = metrics["loss"]
            results[f"accuracy_len_{length}"] = metrics["accuracy"]
        results["loss"] = sum(losses) / len(losses)
        results["accuracy"] = sum(accuracies) / len(accuracies)
        return results

    def d_output(self, cfg: DictConfig) -> int:
        return 1

    def _pseudo_input_shape(self, cfg: DictConfig) -> tuple[int, int]:
        return (cfg.task.train_max_seq_length + 1, 2)

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits[:, -1, 0], targets.float())

    def _compute_metrics(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        predictions = (logits[:, -1, 0] > 0).float()
        accuracy = (predictions == targets).float().mean().item()
        return {"loss": loss.item(), "accuracy": accuracy}
