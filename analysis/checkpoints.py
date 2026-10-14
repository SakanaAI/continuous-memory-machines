"""Strict loading of the CMM checkpoints used for paper attention analyses."""

import random

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmm_task(path, device, num_workers):
    from tasks import construct_task

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    if checkpoint["model_type"] != "CMM":
        raise ValueError(f"Not a CMM checkpoint: {path}")
    with open_dict(cfg):
        cfg.device = device
        cfg.logging.wandb.enabled = False
        cfg.training.compile = False
        cfg.training.num_workers = num_workers
        cfg.training.setdefault("use_scheduler", False)
        cfg.model.backbone.pretrained = False  # all backbone weights are in checkpoint
        if "ema" in cfg.training:
            cfg.training.ema.enabled = False
        if "copy_dim" in cfg.task:
            cfg.task.setdefault("copy_eval_sequence_lengths", [])
            cfg.task.setdefault("copy_binary", False)
            cfg.task.setdefault(
                "copy_min_sequence_length", cfg.task.copy_max_sequence_length
            )
    seed_all(0)
    task = construct_task(cfg, task_type=checkpoint["task_type"])
    state = {
        k.removeprefix("_orig_mod.").replace("start_trace", "start_stm"): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    # April checkpoints predate LTM LayerNorm. Restore their actual trained
    # architecture, as in the original trained architecture; do not insert an
    # untrained normalization or silently ignore missing weights.
    norm_prefix = "rnn.recurrent_module.memory_updater.ltm_norm."
    if not any(k.startswith(norm_prefix) for k in state):
        task.model.rnn.recurrent_module.memory_updater.ltm_norm = torch.nn.Identity()
    task.model.load_state_dict(state, strict=True)
    task.model.eval()
    return task, cfg, checkpoint["task_type"], int(checkpoint["epoch"])
