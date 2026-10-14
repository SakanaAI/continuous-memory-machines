import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import DictConfig

from tasks.recall.base import BaseRecallTask


class AssociativeRecallDataset(Dataset):
    def __init__(self, bits_per_vector: int, num_vectors_per_item: int, max_items: int, min_items: int, num_samples: int):
        self.bits_per_vector, self.num_vectors_per_item = bits_per_vector, num_vectors_per_item
        self.max_items, self.min_items, self.num_samples = max_items, min_items, num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        W, V, K_max = self.bits_per_vector, self.num_vectors_per_item, self.max_items
        K = torch.randint(self.min_items, K_max + 1, ()).item()  # sample num items U[min, max]

        items = torch.bernoulli(torch.full((K, V, W), 0.5))  # (K, V, W)

        query_idx = torch.randint(0, K - 1, ()).item()  # must have a successor
        query_item = items[query_idx]  # (V, W)
        target_item = items[query_idx + 1]  # (V, W)

        # Build input: items | EOS | query | EOS | output_slots
        # Pad to max_items length so all sequences in a batch have the same shape
        item_rows = items.reshape(K * V, W)  # (K*V, W)
        pad_rows = K_max * V - K * V
        item_input = torch.cat([item_rows, torch.zeros(pad_rows, W)], dim=0)  # (K_max*V, W)
        item_input = torch.cat([item_input, torch.zeros(K_max * V, 1)], dim=1)  # (K_max*V, W+1)

        eos = torch.ones(1, W + 1)  # (1, W+1)
        query_input = torch.cat([query_item, torch.zeros(V, 1)], dim=1)  # (V, W+1)
        output_slots = torch.zeros(V, W + 1)  # (V, W+1)

        input = torch.cat([item_input, eos, query_input, eos, output_slots], dim=0)  # (K_max*V + 2 + 2*V, W+1)

        total_steps = input.shape[0]
        target = torch.zeros(total_steps, W)
        target[-V:] = target_item  # last V rows are the answer

        return input, target


class AssociativeRecallTask(BaseRecallTask):
    _performance_metric_key = "eval_loss"
    _performance_higher_is_better = False

    def setup_dataset(self, cfg: DictConfig):
        self.bits_per_vector = cfg.task.bits_per_vector
        self.num_vectors_per_item = cfg.task.num_vectors_per_item
        self.max_items = cfg.task.max_items

        kwargs = dict(
            bits_per_vector=cfg.task.bits_per_vector,
            num_vectors_per_item=cfg.task.num_vectors_per_item,
            max_items=cfg.task.max_items,
            min_items=cfg.task.min_items,
            num_samples=cfg.task.num_samples,
        )
        self.train_dataset = AssociativeRecallDataset(**kwargs)
        self.val_dataset = AssociativeRecallDataset(**kwargs)
        self.test_dataset = AssociativeRecallDataset(**kwargs)

    def d_output(self, cfg: DictConfig) -> int:
        return cfg.task.bits_per_vector

    def _pseudo_input_shape(self, cfg: DictConfig) -> tuple[int, int]:
        K, V, W = cfg.task.max_items, cfg.task.num_vectors_per_item, cfg.task.bits_per_vector
        total_steps = K * V + 2 + 2 * V  # items + 2 EOS + query + output
        return (total_steps, W + 1)

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        V = self.num_vectors_per_item
        return F.binary_cross_entropy_with_logits(logits[:, -V:, :], targets[:, -V:, :])

    def _compute_metrics(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        V = self.num_vectors_per_item
        preds = (torch.sigmoid(logits[:, -V:, :]) > 0.5).float()
        tgts = targets[:, -V:, :]
        bit_accuracy = (preds == tgts).float().mean().item()
        vector_accuracy = (preds == tgts).all(dim=-1).float().mean().item()
        return {"loss": loss.item(), "bit_accuracy": bit_accuracy, "vector_accuracy": vector_accuracy}
