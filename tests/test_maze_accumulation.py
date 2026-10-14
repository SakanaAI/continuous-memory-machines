import copy
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import construct_rnn
from tasks.maze.maze import MazeTask, maze_loss
from utils.ema import ModelEMA


class SmallMazeModel(nn.Module):
    """Use the real recurrent cells with a small deterministic patch encoder."""

    def __init__(self, model_name):
        super().__init__()
        self.backbone = nn.Linear(3, 16)
        self.rnn = construct_rnn(OmegaConf.create({
            "model": model_name,
            "d_input": 8,
            "d_model": 8,
            "num_heads": 2,
            "num_tapes": 2,
            "tape_length": 3,
            "head_dim": 2,
            "dropout": 0.0,
            "backbone": {"mode": "patches", "num_heads": 2},
        }))
        self.output_proj = nn.Linear(8, 10)
        with torch.no_grad():
            self(torch.zeros(1, 3), self.get_initial_states(1))

    def get_initial_states(self, batch_size):
        return self.rnn.get_initial_state(batch_size)

    def forward(self, images, states):
        features = self.backbone(images).reshape(images.size(0), 2, 8)
        features = features.unsqueeze(0).expand(2, -1, -1, -1)
        hidden, states = self.rnn(features, states)
        return self.output_proj(hidden).permute(1, 2, 0), states


class RecordingLogger:
    def __init__(self):
        self.updates = []

    def log_update_metrics(self, metrics, step):
        self.updates.append((step, dict(metrics)))


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_name", ["LSAM", "NAMTM"])
@pytest.mark.parametrize("accumulation_steps", [1, 4, 16])
@pytest.mark.parametrize("optimizer_name,max_grad_norm", [("SGD", 0.0), ("SGD", 0.05), ("AdamW", 0.05)])
def test_accumulation_matches_large_batches(model_name, accumulation_steps, optimizer_name, max_grad_norm):
    torch.manual_seed(12)
    dataset = TensorDataset(torch.randn(14, 3), torch.randint(0, 5, (14, 2)))
    task = object.__new__(MazeTask)
    task.device = "cpu"
    task.route_length = 2
    task.curriculum_lookahead = 1
    task.accumulation_steps = accumulation_steps
    task.max_grad_norm = max_grad_norm
    task.train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
    task.model = SmallMazeModel(model_name)
    reference = copy.deepcopy(task.model)
    optimizer_cls = getattr(torch.optim, optimizer_name)
    task.optimizer = optimizer_cls(task.model.parameters(), lr=0.01)
    reference_optimizer = optimizer_cls(reference.parameters(), lr=0.01)
    task.scheduler = torch.optim.lr_scheduler.StepLR(task.optimizer, step_size=1, gamma=0.9)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.9)
    task.ema = ModelEMA(task.model, beta=0.8, update_every=1, update_after_step=1)
    reference_ema = ModelEMA(reference, beta=0.8, update_every=1, update_after_step=1)
    task.logger = RecordingLogger()
    optimizer_steps = []
    task.optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(1))

    reference_loader = DataLoader(dataset, batch_size=2 * accumulation_steps, shuffle=False)
    expected_updates = []
    for epoch in (1, 2):
        epoch_metrics = task.train(epoch)
        weighted_loss = 0.0
        for images, targets in reference_loader:
            reference_optimizer.zero_grad()
            logits, _ = reference(images, reference.get_initial_states(images.size(0)))
            loss, _, _ = maze_loss(logits.reshape(images.size(0), 2, 5, 2), targets, 1)
            loss.backward()
            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in reference.parameters() if p.grad is not None))
            expected_updates.append((loss.item(), grad_norm.item()))
            weighted_loss += loss.item() * images.size(0)
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(reference.parameters(), max_grad_norm, error_if_nonfinite=True)
            reference_optimizer.step()
            reference_ema.update()
            reference_scheduler.step()
        assert epoch_metrics["loss"] == pytest.approx(weighted_loss / len(dataset), rel=2e-5, abs=2e-6)
        torch.testing.assert_close(task.model.state_dict(), reference.state_dict(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(task.ema.ema_model.state_dict(), reference_ema.ema_model.state_dict(), rtol=2e-5, atol=2e-6)

    num_updates = 2 * math.ceil(len(task.train_dataloader) / accumulation_steps)
    assert len(optimizer_steps) == num_updates
    assert task.ema.step == reference_ema.step == num_updates
    assert task.scheduler.last_epoch == reference_scheduler.last_epoch == num_updates
    assert [step for step, _ in task.logger.updates] == list(range(1, num_updates + 1))
    assert task.optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]
    for (_, metrics), (loss, grad_norm) in zip(task.logger.updates, expected_updates, strict=True):
        assert metrics["loss"] == pytest.approx(loss, rel=2e-5, abs=2e-6)
        assert metrics["grad_norm"] == pytest.approx(grad_norm, rel=2e-5, abs=2e-6)
        assert metrics["grad_norm_clip_return"] == pytest.approx(grad_norm, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "four"])
def test_invalid_accumulation_rejected_before_loading_data(invalid):
    cfg = OmegaConf.create({"training": {"accumulation_steps": invalid}})
    with pytest.raises(ValueError, match="accumulation_steps must be a positive integer"):
        MazeTask(cfg)


@pytest.mark.parametrize("accumulation_steps", [None, 1, 4])
def test_scheduler_uses_optimizer_update_count(monkeypatch, accumulation_steps):
    dataset = TensorDataset(torch.zeros(14, 3), torch.zeros(14, 2, dtype=torch.long))
    monkeypatch.setattr("tasks.maze.maze.HuggingFaceMazeDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(MazeTask, "build_model", lambda self, cfg: SmallMazeModel("LSAM"))
    cfg = OmegaConf.create({
        "device": -1,
        "task": {"dataset": "test-large", "route_length": 2, "curriculum_lookahead": 1},
        "training": {
            "batch_size": 2, "epochs": 20, "learning_rate": 1e-3,
            "weight_decay": 0.0, "max_grad_norm": 1, "use_scheduler": True,
            "compile": False, "num_workers": 0, "ema": {"enabled": False},
        },
    })
    if accumulation_steps is not None:
        cfg.training.accumulation_steps = accumulation_steps
    task = MazeTask(cfg)
    assert task.accumulation_steps == (accumulation_steps or 1)
    total_updates = 20 * math.ceil(7 / task.accumulation_steps)
    assert task.scheduler._schedulers[0].total_iters == total_updates // 10
    assert task.scheduler._schedulers[1].T_max == total_updates - total_updates // 10
