import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import construct_rnn, construct_backbone
from tasks.task import Task
from tasks.utils import get_device


class DualLoss(nn.Module):
    """Averages min-CE-tick loss and most-certain-tick loss across internal ticks."""

    def forward(self, logits, targets):
        B, C, T = logits.shape # (B, num_classes, T)
        targets_expanded = targets.unsqueeze(-1).expand(B, T) # (B, T)
        losses = F.cross_entropy(logits, targets_expanded, reduction="none") # (B, T)

        batch_idx = torch.arange(B, device=logits.device)

        loss_min_ce = losses[batch_idx, losses.argmin(dim=1)].mean() # lowest CE tick

        probs = logits.softmax(dim=1) # (B, C, T)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1) # (B, T)
        loss_most_certain = losses[batch_idx, entropy.argmin(dim=1)].mean() # most certain tick

        return (loss_min_ce + loss_most_certain) / 2


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item[self.image_key]
        label = item[self.label_key]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


class ImageClassificationModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig):
        super().__init__()

        self.iterations = task_cfg.iterations
        self.backbone = construct_backbone(model_cfg.backbone, model_cfg.d_input)
        if model_cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.rnn = construct_rnn(model_cfg)
        self.output_proj = nn.Linear(model_cfg.d_model, task_cfg.out_dims)

    def forward(self, input, states):
        assert input.ndim == 4, "Input should be of size B x C x H x W"
        features = self.backbone(input) # (B, C, H, W) -> (B, d_input) | (B, N, d_token)
        features = features.unsqueeze(0).expand(self.iterations, *features.shape) # (T, B, ...)
        hidden, final_state = self.rnn(features, state=states) # (T, B, d_model)
        logits = self.output_proj(hidden) # (T, B, num_classes)
        logits = logits.permute(1, 2, 0) # (B, num_classes, T)
        return logits, final_state

    def get_initial_states(self, batch_size):
        return self.rnn.get_initial_state(batch_size)


class BaseImageClassificationTask(Task):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = get_device(cfg)
        self.iterations = cfg.task.iterations

        self.setup_dataset(cfg)

        nw = cfg.training.num_workers if "num_workers" in cfg.training else 7
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

        self.model = self.build_model(cfg).to(self.device)
        self.maybe_compile(cfg)
        print(f"Model Architecture:\n{self.model}")
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

        self.use_dual_loss = cfg.training.loss == "dual"
        self.loss_fn = DualLoss() if self.use_dual_loss else nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs)
        self.max_grad_norm = cfg.training.max_grad_norm

    def setup_dataset(self, cfg: DictConfig):
        raise NotImplementedError("Subclasses must implement setup_dataset to define self.train_dataset, self.val_dataset, and self.test_dataset")

    def _build_datasets(self, hf_dataset, train_transform, eval_transform, splits=("train", "test", "test"), image_key="image",):
        train_split, val_split, test_split = splits
        self.train_dataset = ImageClassificationDataset(hf_dataset[train_split], train_transform, image_key=image_key)
        self.val_dataset = ImageClassificationDataset(hf_dataset[val_split], eval_transform, image_key=image_key)
        self.test_dataset = ImageClassificationDataset(hf_dataset[test_split], eval_transform, image_key=image_key)

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        objectives = []
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device) # (B, C, H, W)
            targets = targets.to(self.device) # (B)

            self.optimizer.zero_grad()
            initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
            logits, _ = self.model(inputs, states=initial_states) # (B, num_classes, iterations)

            loss = self.loss_fn(logits, targets) if self.use_dual_loss else self.loss_fn(logits[:, :, -1], targets)

            predictions = logits[:, :, -1].argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()

            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            objectives.append((loss.item(), grad_norm.item(), accuracy))
            pbar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{accuracy:.4f}")

        self.scheduler.step()
        objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]
        metrics = {"loss": objectives[0], "grad_norm": objectives[1], "accuracy": objectives[2], "lr": self.optimizer.param_groups[0]["lr"]}
        return metrics

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_accuracy = 0

        for inputs, targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
                logits, _ = self.model(inputs, states=initial_states)

                loss = self.loss_fn(logits, targets) if self.use_dual_loss else self.loss_fn(logits[:, :, -1], targets)

                total_loss += loss.item()

                predictions = logits[:, :, -1].argmax(dim=1)
                accuracy = (predictions == targets).float().mean().item()
                total_accuracy += accuracy

        num_batches = len(dataloader)
        return {"loss": total_loss / num_batches, "accuracy": total_accuracy / num_batches,}

    def validate(self, epoch: int) -> dict[str, float]:
        return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
        return self._evaluate(self.test_dataloader, desc)

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
        accuracies = [pair[1] for pair in metrics["eval_accuracy"]]
        recent_accuracies = accuracies[-window_size:]
        if not recent_accuracies:
            return 0.0
        return sum(recent_accuracies) / len(recent_accuracies)

    def build_model(self, cfg: DictConfig) -> nn.Module:
        model = ImageClassificationModel(cfg.model, cfg.task)
        # Materialize lazy modules with a pseudo input on the current default device
        # (CPU at __init__ time, meta during the max_parameters binary search)
        pseudo_inputs = torch.zeros((1, *self.input_shape))
        initial_states = model.get_initial_states(batch_size=1)
        with torch.no_grad():
            model(pseudo_inputs, states=initial_states)
        return model
