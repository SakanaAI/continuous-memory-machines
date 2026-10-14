import json
import os
from omegaconf import OmegaConf
import wandb


class LocalLogger:
    def __init__(self, cfg, task_name: str, model_name: str):
        self.metrics = {}  # Stores metrics as {'name': [(step_a, value_a), (step_b, value_b), ...]}
        self.wandb_step = 0

        wandb_cfg = cfg.logging.wandb
        self.use_wandb = wandb_cfg.get("enabled", False) and wandb is not None

        if self.use_wandb:
            timestamp = os.path.basename(os.getcwd())
            run_name = f"{task_name}_{model_name}_{timestamp}"

            wandb.init(
                project=wandb_cfg.get("project", "smctm"),
                entity=wandb_cfg.get("entity", None),
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=wandb_cfg.get("tags", []),
                name=run_name,
            )
    def log(self, name: str, value: float, step: int):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))

        if self.use_wandb:
            wandb.log({"epoch": step, name: value}, step=self.wandb_step)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = ""):
        wandb_payload = {"epoch": step}
        for name, value in metrics.items():
            key = f"{prefix}_{name}"
            if isinstance(value, (int, float)):
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append((step, value))
                wandb_payload[key] = value
            else:
                wandb_payload[key] = value
        if self.use_wandb and wandb_payload:
            wandb.log(wandb_payload, step=self.wandb_step)

    def log_update_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train_update"):
        self.wandb_step = step
        wandb_payload = {f"{prefix}_step": step}
        for name, value in metrics.items():
            key = f"{prefix}_{name}"
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
            wandb_payload[f"{prefix}/{name}"] = value
            wandb_payload[key] = value
        if self.use_wandb:
            wandb.log(wandb_payload, step=step)

    def get_metrics(self) -> dict:
        return self.metrics

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.metrics, f)

    def finish(self):
        if self.use_wandb:
            wandb.finish()
