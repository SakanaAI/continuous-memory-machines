import logging
import os
import torch

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def save_checkpoint(epoch: int, model_state: dict, optimizer_state: dict, scheduler_state: dict, metrics: dict, rng_state: dict, cfg: DictConfig, task_type: str, model_type: str, checkpoint_dir: str = ".", ema_state: dict | None = None):
    checkpoint = {"epoch": epoch, "model_state_dict": model_state, "optimizer_state_dict": optimizer_state, "scheduler_state_dict": scheduler_state, "metrics": metrics, "rng_state": rng_state, "cfg": OmegaConf.to_container(cfg, resolve=True), "task_type": task_type, "model_type": model_type}
    if ema_state is not None:
        checkpoint["ema_state_dict"] = ema_state
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)


def get_rng_state() -> dict:
    return {
        "python": torch.get_rng_state(),
        "numpy": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(rng_state: dict):
    torch.set_rng_state(rng_state["python"])
    torch.random.set_rng_state(rng_state["numpy"])
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])

def load_cfg_from_checkpoint(checkpoint_path: str) -> tuple[DictConfig, str, str]:
    """Recover the (cfg, task_type, model_type) triple a checkpoint was trained with."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.training.setdefault("use_scheduler", False)
    if "copy_dim" in cfg.task:
        cfg.task.setdefault("copy_eval_sequence_lengths", [])
    OmegaConf.set_struct(cfg, True)
    return cfg, checkpoint["task_type"], checkpoint["model_type"]


_DRIFT_IGNORE = {"checkpoint.path", "checkpoint.load_checkpoint"}


def _flatten_cfg(d: dict, prefix: str = "") -> dict:
    """Flatten a nested cfg dict into {dot.path: leaf}; non-dict values are leaves."""
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): out.update(_flatten_cfg(v, path))
        else: out[path] = v
    return out


def _check_cfg_drift(live_cfg: DictConfig, saved_cfg: DictConfig) -> None:
    """Raise on architecture drift (model.*/task.*); warn on any other drift."""
    live = _flatten_cfg(OmegaConf.to_container(live_cfg, resolve=True))
    saved = _flatten_cfg(OmegaConf.to_container(saved_cfg, resolve=True))
    arch_drift, other_drift = [], []
    for k in sorted((set(live) | set(saved)) - _DRIFT_IGNORE):
        lv = live[k] if k in live else "<missing>"
        sv = saved[k] if k in saved else "<missing>"
        if lv == sv:
            continue
        bucket = arch_drift if k.startswith(("model.", "task.")) else other_drift
        bucket.append((k, sv, lv))

    if other_drift:
        msg = "\n".join(f"  {k}: saved={s!r} -> live={l!r}" for k, s, l in other_drift)
        log.warning(f"Cfg drift from checkpoint (non-architecture, continuing):\n{msg}")
    if arch_drift:
        msg = "\n".join(f"  {k}: saved={s!r} -> live={l!r}" for k, s, l in arch_drift)
        raise ValueError(
            f"Architecture cfg drift from checkpoint (cannot safely resume):\n{msg}\n"
            f"Re-run with the original model/task overrides, or train from scratch."
        )


def load_checkpoint(cfg: DictConfig, task, original_cwd: str) -> tuple[int, dict]:

    if cfg.checkpoint.path is None:
        raise FileNotFoundError("cfg.checkpoint.path is None — set it to a checkpoint .pt file to load from.")

    path = cfg.checkpoint.path
    if not os.path.isabs(path):
        path = os.path.join(original_cwd, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    cfg.checkpoint.path = path

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _check_cfg_drift(cfg, OmegaConf.create(checkpoint["cfg"]))

    os.chdir(os.path.dirname(os.path.abspath(path)))
    log.info(f"Loaded checkpoint dir: {os.getcwd()}")

    state_dict = {k.replace("start_trace", "start_stm"): v for k, v in checkpoint["model_state_dict"].items()}
    task.model.load_state_dict(state_dict)
    if hasattr(task, "ema") and task.ema is not None and "ema_state_dict" in checkpoint:
        task.ema.load_state_dict(checkpoint["ema_state_dict"])
    task.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if task.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        task.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    set_rng_state(checkpoint["rng_state"])
    log.info(f"Loaded checkpoint at epoch {checkpoint['epoch']}")
    return checkpoint["epoch"] + 1, checkpoint["metrics"]
