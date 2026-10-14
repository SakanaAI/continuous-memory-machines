import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from omegaconf import DictConfig
from datasets import load_dataset
from tqdm import tqdm

from models import construct_rnn, construct_backbone
from tasks.task import Task
from tasks.utils import get_device
from utils.ema import ModelEMA


# ---------- Dataset ----------

class HuggingFaceMazeDataset(Dataset):
    """Loads maze images from HuggingFace SakanaAI/mazes-{small,medium,large}.

    Produces (image, path) pairs where:
      image: (3, H, W) float tensor in [-1, 1]
             walls=black, empty=white, start=red, goal=green
      path:  (route_length,) long tensor of actions
             0=Up, 1=Down, 2=Left, 3=Right, 4=Wait (padding)
    """


    _HF_TO_CTM = [0, 1, 2, 3, 4]

    _dataset_cache = {}

    def __init__(self, dataset_name: str, split: str, route_length: int, augment: bool = False):
        cache_key = f"{dataset_name}:{split}"
        if cache_key not in HuggingFaceMazeDataset._dataset_cache:
            print(f"Loading HuggingFace maze dataset ({dataset_name}, split={split})...")
            HuggingFaceMazeDataset._dataset_cache[cache_key] = load_dataset(dataset_name, split=split, trust_remote_code=False)
            print(f"Loaded {len(HuggingFaceMazeDataset._dataset_cache[cache_key])} mazes")
        self.hf_data = HuggingFaceMazeDataset._dataset_cache[cache_key]
        self.route_length = route_length
        self.augment = augment

        if "large" in dataset_name:
            self.grid_size = 39
        elif "medium" in dataset_name:
            self.grid_size = 19
        else:
            self.grid_size = 15

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        rgb = np.array(item["image"], dtype=np.float32) / 255.0 # (H, W, 3) in [0, 1]

        path = np.full(self.route_length, 4, dtype=np.int64) # 4 = Wait
        for i, action in enumerate(item["solution_path"]):
            if i >= self.route_length or action == 4:
                break
            path[i] = self._HF_TO_CTM[action]

        if self.augment:
            rgb, path = self._apply_augmentation(rgb, path)

        sample = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)  # (3, H, W)
        sample = sample * 2 - 1  # scale to [-1, 1]
        return sample, torch.from_numpy(path).long()

    @staticmethod
    def _apply_augmentation(image, path):
        # Random rotation ±90°  (CTM encoding: 0=Up, 1=Down, 2=Left, 3=Right)
        if random.random() < 0.5:
            k = random.choice([-1, 1])
            image = np.rot90(image, k=k, axes=(0, 1))
            new_path = np.copy(path)
            if k == -1:  # 90° CW:  Up→Right, Down→Left, Left→Up, Right→Down
                remap = {0: 3, 1: 2, 2: 0, 3: 1}
            else:  # 90° CCW: Up→Left, Down→Right, Left→Down, Right→Up
                remap = {0: 2, 1: 3, 2: 1, 3: 0}
            for i in range(len(path)):
                if path[i] < 4:
                    new_path[i] = remap[path[i]]
            path = new_path

        # Random horizontal flip:  Left↔Right
        if random.random() < 0.5:
            image = np.fliplr(image)
            new_path = np.copy(path)
            for i in range(len(path)):
                if path[i] == 2:
                    new_path[i] = 3
                elif path[i] == 3:
                    new_path[i] = 2
            path = new_path

        # Random vertical flip:  Up↔Down
        if random.random() < 0.5:
            image = np.flipud(image)
            new_path = np.copy(path)
            for i in range(len(path)):
                if path[i] == 0:
                    new_path[i] = 1
                elif path[i] == 1:
                    new_path[i] = 0
            path = new_path

        return np.ascontiguousarray(image), path


# ---------- Loss ----------

def maze_loss(predictions, targets, curriculum_lookahead=5):
    """Auto-curriculum maze loss (from CTM paper).

    Args:
        predictions: (B, route_length, 5, T) logits per tick
        targets:     (B, route_length) ground-truth actions in [0..4]
        curriculum_lookahead: steps beyond last correct to include in loss

    Returns:
        loss:           scalar
        most_certain:   (B,) tick index with lowest output entropy
        upto_where:     np.array (B,) curriculum boundary per sample
    """
    B, R, C, T = predictions.shape

    # Cross-entropy at every (route_step, tick)  →  (B, R, T)
    targets_expanded = targets.unsqueeze(-1).expand(B, R, T).flatten(0, 1).long()  # (B*R, T)
    losses = F.cross_entropy(predictions.flatten(0, 1), targets_expanded, reduction="none")  # (B*R, T)
    losses = losses.reshape(B, R, T)

    # ---- auto-curriculum mask ----
    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)  # (B, R, T)
    correct_mask = (iscorrects == torch.arange(1, R + 1, device=predictions.device).reshape(1, -1, 1))
    correct_mask[:, 0, :] = 1
    prefix_argmax = correct_mask.cumsum(1).argmax(1)  # (B, T)
    upto_where = prefix_argmax.max(dim=1)[0] + curriculum_lookahead  # (B,)
    upto_where = upto_where.clamp(max=R)
    r_idx = torch.arange(R, device=losses.device)
    loss_mask = (r_idx[None, :] < upto_where[:, None]).float().unsqueeze(-1).expand_as(losses)  # (B, R, T)

    # Masked mean over route dimension → (B, T)
    losses = (losses * loss_mask).sum(1) / loss_mask.sum(1).clamp(min=1)

    # ---- tick selection (dual-loss) ----
    loss_index_min = losses.argmin(dim=1)  # tick with minimum CE

    # Most-certain tick: lowest mean entropy over route predictions
    probs = F.softmax(predictions, dim=2)  # (B, R, 5, T)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=2)  # (B, R, T)
    loss_index_certain = entropy.mean(dim=1).argmin(dim=1)  # (B,)

    batch_idx = torch.arange(B, device=predictions.device)
    loss = ((losses[batch_idx, loss_index_min] + losses[batch_idx, loss_index_certain]) / 2).mean()
    return loss, loss_index_certain, upto_where.detach().cpu().numpy()


# ---------- Model ----------

class MazeModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig):
        super().__init__()
        self.iterations = task_cfg.iterations
        self.backbone = construct_backbone(model_cfg.backbone, model_cfg.d_input)
        if model_cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.rnn = construct_rnn(model_cfg)
        self.output_proj = nn.Linear(model_cfg.d_model, task_cfg.out_dims)

    def forward(self, input, states):  # noqa: A002
        features = self.backbone(input)  # (B, C, H, W) -> (B, d_input) | (B, N, d_token)
        features = features.unsqueeze(0).expand(self.iterations, *features.shape)  # (T, B, ...)
        hidden, final_state = self.rnn(features, state=states)  # (T, B, d_model)
        logits = self.output_proj(hidden)  # (T, B, out_dims)
        logits = logits.permute(1, 2, 0)  # (B, out_dims, T)
        return logits, final_state

    def get_initial_states(self, batch_size):
        return self.rnn.get_initial_state(batch_size)


# ---------- Task ----------

class MazeTask(Task):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.accumulation_steps = cfg.training.get("accumulation_steps", 1)
        if (
            isinstance(self.accumulation_steps, bool)
            or not isinstance(self.accumulation_steps, int)
            or self.accumulation_steps < 1
        ):
            raise ValueError("training.accumulation_steps must be a positive integer")
        self.device = get_device(cfg)
        self.route_length = cfg.task.route_length
        self.curriculum_lookahead = cfg.task.curriculum_lookahead

        dataset_name = cfg.task.dataset
        if "large" in dataset_name:
            self.input_shape = (3, 39, 39)
        elif "medium" in dataset_name:
            self.input_shape = (3, 19, 19)
        else:
            self.input_shape = (3, 15, 15)

        # Datasets
        self.train_dataset = HuggingFaceMazeDataset(dataset_name, split="train", route_length=self.route_length, augment=True)
        self.test_dataset = HuggingFaceMazeDataset(dataset_name, split="test", route_length=self.route_length, augment=False)

        nw = cfg.training.num_workers if "num_workers" in cfg.training else 7
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
        self.val_dataloader = DataLoader(self.test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
        self.test_dataloader = self.val_dataloader  # same split

        # Model
        self.model = self.build_model(cfg).to(self.device)
        self.ema = ModelEMA(
            self.model,
            beta=cfg.training.ema.beta,
            update_every=cfg.training.ema.update_every,
            update_after_step=cfg.training.ema.update_after_step,
        ) if cfg.training.ema.enabled else None
        self.maybe_compile(cfg)
        print(f"Model Architecture:\n{self.model}")
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

        self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        if cfg.training.use_scheduler:
            steps_per_epoch = (len(self.train_dataloader) + self.accumulation_steps - 1) // self.accumulation_steps
            total_steps = cfg.training.epochs * steps_per_epoch
            warmup_steps = min(10000, total_steps // 10)
            warmup = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])
        else:
            self.scheduler = None
        self.max_grad_norm = cfg.training.max_grad_norm

    def build_model(self, cfg: DictConfig) -> nn.Module:
        model = MazeModel(cfg.model, cfg.task)
        pseudo = torch.zeros((1, *self.input_shape))
        states = model.get_initial_states(batch_size=1)
        with torch.no_grad():
            model(pseudo, states=states)
        return model

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        objectives = []
        num_batches = len(self.train_dataloader)
        updates_per_epoch = (num_batches + self.accumulation_steps - 1) // self.accumulation_steps
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        cell = model.rnn.recurrent_module if hasattr(model.rnn, "recurrent_module") else None
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()
                # drop_last=True makes microbatches equal-sized. Normalize a
                # shorter final accumulation group by its actual batch count.
                window_size = min(self.accumulation_steps, num_batches - batch_idx)
                window_metrics = {}

            attention_trace = []
            attention_handles = self._register_memory_attention_hooks(model, attention_trace)
            states = self.model.get_initial_states(batch_size=images.size(0))
            try:
                logits, _ = self.model(images, states=states)  # (B, route_length*5, T)
            finally:
                for attention_handle in attention_handles:
                    attention_handle.remove()

            predictions = logits.reshape(logits.size(0), self.route_length, 5, logits.size(-1))  # (B, R, 5, T)
            loss, most_certain, upto_where = maze_loss(predictions, targets, self.curriculum_lookahead)

            # Per-step accuracy at the most-certain tick
            batch_indices = torch.arange(predictions.size(0), device=predictions.device)
            pred_at_certain = predictions.argmax(2)[batch_indices, :, most_certain]  # (B, R)
            step_accuracy = (pred_at_certain == targets).float().mean().item()
            full_maze_accuracy = (pred_at_certain == targets).all(dim=1).float().mean().item()

            (loss / window_size).backward()
            batch_metrics = {
                "loss": loss.item(),
                "accuracy": step_accuracy,
                "full_maze_accuracy": full_maze_accuracy,
                "curriculum_boundary": float(upto_where.mean()),
                **self._memory_attention_budget(attention_trace, cell),
            }
            for key, value in batch_metrics.items():
                window_metrics[key] = window_metrics.get(key, 0.0) + value / window_size

            if (batch_idx + 1) % self.accumulation_steps != 0 and batch_idx + 1 != num_batches:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{step_accuracy:.3f}", maze=f"{full_maze_accuracy:.3f}", curriculum=f"{upto_where.mean():.1f}")
                continue

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
            grad_norm_backbone = self._module_grad_norm(model.backbone)
            grad_norm_cross_attn = self._module_grad_norm(model.rnn.cross_attn if hasattr(model.rnn, "cross_attn") else None)
            grad_norm_memory_updater = self._module_grad_norm(cell.memory_updater if hasattr(cell, "memory_updater") else None)
            grad_norm_synapses = self._module_grad_norm(cell.synapses if hasattr(cell, "synapses") else None)
            grad_norm_synchronizer = self._module_grad_norm(cell.synchronizer if hasattr(cell, "synchronizer") else None)
            grad_norm_memory_layers = self._memory_layer_grad_norms(cell)
            memory_update_gates = self._memory_layer_gates(cell)
            if self.max_grad_norm > 0:
                grad_norm_clip_return = clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            else:
                grad_norm_clip_return = grad_norm
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()
            if self.scheduler is not None:
                self.scheduler.step()

            update_step = (epoch - 1) * updates_per_epoch + batch_idx // self.accumulation_steps + 1
            update_metrics = {
                **window_metrics,
                "grad_norm": grad_norm.item(),
                "grad_norm_clip_return": grad_norm_clip_return.item(),
                "grad_norm_backbone": grad_norm_backbone,
                "grad_norm_cross_attn": grad_norm_cross_attn,
                "grad_norm_memory_updater": grad_norm_memory_updater,
                "grad_norm_synapses": grad_norm_synapses,
                "grad_norm_synchronizer": grad_norm_synchronizer,
                "lr": self.optimizer.param_groups[0]["lr"],
                **grad_norm_memory_layers,
                **memory_update_gates,
            }
            self._log_update_metrics(update_step, update_metrics)

            objectives.append((window_size, update_metrics))
            pbar.set_postfix(loss=f"{window_metrics['loss']:.4f}", grad=f"{grad_norm.item():.2f}", acc=f"{window_metrics['accuracy']:.3f}", maze=f"{window_metrics['full_maze_accuracy']:.3f}", curriculum=f"{window_metrics['curriculum_boundary']:.1f}")

        metrics = {
            key: sum(size * values[key] for size, values in objectives) / num_batches
            for key in objectives[0][1]
        }
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    @staticmethod
    def _module_grad_norm(module):
        if module is None:
            return 0.0
        grad_norms = [param.grad.norm() for param in module.parameters() if param.grad is not None]
        if not grad_norms:
            return 0.0
        return torch.sqrt(sum(norm ** 2 for norm in grad_norms)).item()

    def _log_update_metrics(self, update_step, metrics):
        if not hasattr(self, "logger"):
            return
        self.logger.log_update_metrics(metrics, update_step)

    @staticmethod
    def _memory_layer_grad_norms(cell):
        if cell is None or not hasattr(cell, "memory_updater"):
            return {}
        return {
            f"grad_norm_memory_layer{layer_idx}": MazeTask._module_grad_norm(block)
            for layer_idx, block in enumerate(cell.memory_updater.blocks)
        }

    @staticmethod
    def _memory_layer_gates(cell):
        if cell is None or not hasattr(cell, "memory_updater"):
            return {}
        return {
            f"memory_update_gate_layer{layer_idx}": block.memory_update_gate.item()
            for layer_idx, block in enumerate(cell.memory_updater.blocks)
            if block.memory_update_gate is not None
        }

    @staticmethod
    def _register_memory_attention_hooks(model, attention_trace):
        if not hasattr(model.rnn, "recurrent_module"):
            return []
        cell = model.rnn.recurrent_module
        if not hasattr(cell, "memory_updater") or len(cell.memory_updater.blocks) == 0:
            return []

        attention_trace.extend([] for _ in cell.memory_updater.blocks)
        handles = []
        for layer_idx, block in enumerate(cell.memory_updater.blocks):
            def _attention_hook(module, _inputs, _output, layer_idx=layer_idx):
                attention_trace[layer_idx].append(module._last_attn.detach())  # (B, H, S, S)

            handles.append(block.register_forward_hook(_attention_hook))
        return handles

    @staticmethod
    def _memory_attention_budget(attention_trace, cell):
        names = (
            "ltm_to_sink", "ltm_to_ltm", "ltm_to_stm",
            "stm_to_sink", "stm_to_ltm", "stm_to_stm",
            "sink_to_sink", "sink_to_ltm", "sink_to_stm",
        )
        budget = {f"attention_{name}": 0.0 for name in names}
        if hasattr(cell, "memory_updater"):
            for layer_idx in range(len(cell.memory_updater.blocks)):
                for name in names:
                    budget[f"attention_layer{layer_idx}_{name}"] = 0.0
        if not attention_trace or not hasattr(cell, "memory_updater"):
            return budget

        for layer_idx, layer_trace in enumerate(attention_trace):
            if not layer_trace:
                continue

            attention = torch.stack(layer_trace, dim=0)  # list[(B, H, S, S)] -> (T, B, H, S, S)
            attention = attention.mean(dim=(0, 1, 2))  # (T, B, H, S, S) -> (S, S)
            L = cell.memory_updater.num_ltm_tokens
            S = attention.shape[0]
            sink_sl = slice(0, 1)
            ltm_sl = slice(1, 1 + L)
            stm_sl = slice(1 + L, S)

            sink_q = attention[sink_sl, :].mean(dim=0)  # (1, S) -> (S,)
            ltm_q = attention[ltm_sl, :].mean(dim=0)  # (L, S) -> (S,)
            stm_q = attention[stm_sl, :].mean(dim=0)  # (M, S) -> (S,)
            layer_budget = {
                "ltm_to_sink": ltm_q[sink_sl].sum().item(),
                "ltm_to_ltm": ltm_q[ltm_sl].sum().item(),
                "ltm_to_stm": ltm_q[stm_sl].sum().item(),
                "stm_to_sink": stm_q[sink_sl].sum().item(),
                "stm_to_ltm": stm_q[ltm_sl].sum().item(),
                "stm_to_stm": stm_q[stm_sl].sum().item(),
                "sink_to_sink": sink_q[sink_sl].sum().item(),
                "sink_to_ltm": sink_q[ltm_sl].sum().item(),
                "sink_to_stm": sink_q[stm_sl].sum().item(),
            }
            for name, value in layer_budget.items():
                budget[f"attention_layer{layer_idx}_{name}"] = value
                if layer_idx == 0:
                    budget[f"attention_{name}"] = value
        return budget

    def _evaluate(self, dataloader, desc: str, model=None) -> dict[str, float]:
        model = self.model if model is None else model
        model.eval()
        total_loss, total_step_acc, total_maze_acc, total_curriculum = 0.0, 0.0, 0.0, 0.0

        for images, targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                images, targets = images.to(self.device), targets.to(self.device)
                states = model.get_initial_states(batch_size=images.size(0))
                logits, _ = model(images, states=states)

                predictions = logits.reshape(logits.size(0), self.route_length, 5, logits.size(-1))
                loss, most_certain, upto_where = maze_loss(predictions, targets, self.curriculum_lookahead)

                batch_idx = torch.arange(predictions.size(0), device=predictions.device)
                pred_at_certain = predictions.argmax(2)[batch_idx, :, most_certain]
                total_loss += loss.item()
                total_step_acc += (pred_at_certain == targets).float().mean().item()
                total_maze_acc += (pred_at_certain == targets).all(dim=1).float().mean().item()
                total_curriculum += upto_where.mean()

        n = len(dataloader)
        return {
            "loss": total_loss / n,
            "accuracy": total_step_acc / n,
            "full_maze_accuracy": total_maze_acc / n,
            "curriculum_boundary": total_curriculum / n,
        }

    @staticmethod
    def _format_eval_stats(metrics: dict[str, float]) -> str:
        keys = ("loss", "accuracy", "full_maze_accuracy", "curriculum_boundary")
        return ", ".join(f"{key}={metrics[key]:.4f}" for key in keys)

    def _print_eval_stats(self, desc: str, current_metrics: dict[str, float], ema_metrics: dict[str, float] | None):
        if ema_metrics is None:
            print(f"{desc} current: {self._format_eval_stats(current_metrics)}")
            return
        print(f"{desc} current: {self._format_eval_stats(current_metrics)}")
        print(f"{desc} EMA:     {self._format_eval_stats(ema_metrics)}")

    def validate(self, epoch: int) -> dict[str, float]:
        metrics = self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")
        ema_metrics = None
        if self.ema is not None:
            ema_metrics = self._evaluate(self.val_dataloader, f"EMA Val Epoch {epoch}", model=self.ema.ema_model)
            metrics.update({f"ema_{key}": value for key, value in ema_metrics.items()})
        self._print_eval_stats(f"Val Epoch {epoch}", metrics, ema_metrics)
        return metrics

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch else "Test"
        metrics = self._evaluate(self.test_dataloader, desc)
        ema_metrics = None
        if self.ema is not None:
            ema_metrics = self._evaluate(self.test_dataloader, f"EMA {desc}", model=self.ema.ema_model)
            metrics.update({f"ema_{key}": value for key, value in ema_metrics.items()})
        self._print_eval_stats(desc, metrics, ema_metrics)
        return metrics

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
        accuracies = [pair[1] for pair in metrics["eval_full_maze_accuracy"]]
        recent = accuracies[-window_size:]
        return sum(recent) / len(recent) if recent else 0.0
