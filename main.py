
import json
import logging
import timeit

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

torch.set_float32_matmul_precision("high")

from tasks import construct_task
from tasks.utils import optimise_max_d_model
from utils import LocalLogger, plot_metrics, set_random_seeds, save_checkpoint, get_rng_state, load_checkpoint

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> float:
    set_random_seeds(cfg.seed)

    task_type, model_type = HydraConfig.get().runtime.choices.task, HydraConfig.get().runtime.choices.model
    if cfg.max_parameters is not None:
        compile_was = cfg.training.compile
        ema_was = cfg.training.ema.enabled if "ema" in cfg.training else None
        cfg.training.compile = False  # don't compile the throwaway model
        if ema_was is not None:
            cfg.training.ema.enabled = False  # don't build EMA for the throwaway model
        task = construct_task(cfg, task_type=task_type)
        cfg = optimise_max_d_model(task, cfg)
        cfg.training.compile = compile_was
        if ema_was is not None:
            cfg.training.ema.enabled = ema_was
        task = construct_task(cfg, task_type=task_type)
    else:
        task = construct_task(cfg, task_type=task_type)

    start_epoch, resumed_metrics = 1, {}
    if cfg.checkpoint.load_checkpoint:
        start_epoch, resumed_metrics = load_checkpoint(cfg, task, HydraConfig.get().runtime.cwd)

    log.info(OmegaConf.to_yaml(cfg))
    logger = LocalLogger(cfg=cfg, task_name=task_type, model_name=model_type)
    logger.metrics = resumed_metrics
    task.logger = logger

    timer = timeit.default_timer()
    pbar = tqdm(range(start_epoch, cfg.training.epochs + 1), desc="Training", initial=start_epoch - 1, total=cfg.training.epochs)
    for epoch in pbar:
        train_metrics = task.train(epoch)
        logger.log_metrics(train_metrics, epoch, prefix="train")

        if epoch % cfg.evaluation.evaluation_interval == 0 or epoch == 1:
            eval_metrics = task.eval(epoch)
            logger.log_metrics(eval_metrics, epoch, prefix="eval")

            test_metrics = task.test(epoch)
            if test_metrics is not None:
                logger.log_metrics(test_metrics, epoch, prefix="test")

            if epoch % cfg.logging.plot_interval == 0:
                plot_metrics(logger.get_metrics())

            pbar_desc = f"Val: {eval_metrics}"
            if test_metrics is not None:
                pbar_desc += f" | Test: {test_metrics}"
            pbar.set_description(pbar_desc)

        if cfg.checkpoint.enabled and epoch % cfg.checkpoint.checkpoint_every == 0:
            ema_state = task.ema.state_dict() if hasattr(task, "ema") and task.ema is not None else None
            save_checkpoint(epoch=epoch, model_state=task.model.state_dict(), optimizer_state=task.optimizer.state_dict(), scheduler_state=task.scheduler.state_dict() if task.scheduler is not None else None, metrics=logger.get_metrics(), rng_state=get_rng_state(), cfg=cfg, task_type=task_type, model_type=model_type, ema_state=ema_state)

    logger.save("metrics.json")

    summary = {"parameter_count": task.get_parameter_count(), "runtime": timeit.default_timer() - timer, "performance": task.calculate_performance(logger.get_metrics())}

    metrics = logger.get_metrics()
    if any(name.startswith("test_") for name in metrics):
        summary["test_metrics"] = {name.replace("test_", ""): values[-1][1] for name, values in metrics.items() if name.startswith("test_")}
        log.info(f"Final test metrics: {summary['test_metrics']}")

    with open("summary.json", "w") as f:
        json.dump(summary, f)

    task.perform_diagnostics()
    log.info(f"Performance: {summary['performance']}")
    logger.finish()

    return summary["performance"]


if __name__ == "__main__":
    main()
