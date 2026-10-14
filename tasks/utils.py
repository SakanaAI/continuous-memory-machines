import logging
import math

import torch
from omegaconf import DictConfig
from torch import nn

log = logging.getLogger(__name__)


def get_device(cfg):
    device_id = cfg.device
    if device_id != -1:
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'
    return device


def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimise_max_d_model(task, cfg: DictConfig) -> DictConfig:
    """Binary search for the largest model.d_model whose total parameter count
    (including frozen backbone params) fits within cfg.max_parameters.

    Builds candidate models on the meta device via task.build_model(cfg) so no real memory
    is allocated. Pretrained backbone loading is temporarily disabled because (a) checkpoint
    files cannot be loaded onto the meta device and (b) parameter counts are independent of
    whether weights were loaded from disk.
    """
    width_multiple = getattr(cfg.model, "num_heads", getattr(cfg.model, "nhead", 1))
    if cfg.model.model == "RMT":
        # Its reference sinusoidal encoding also requires an even model width.
        width_multiple = math.lcm(width_multiple, 2)
    min_idx = 1 if cfg.model.model == "DNC" else max(1, 8 // width_multiple)
    low_idx = min_idx
    d_model_search_cap = 4096
    max_d_model = max(min_idx * width_multiple, (d_model_search_cap // width_multiple) * width_multiple)
    high_idx = max(low_idx, max_d_model // width_multiple)
    best_size, best_count = None, None
    has_pretrained = hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "pretrained")
    if has_pretrained:
        pretrained_was = cfg.model.backbone.pretrained
        cfg.model.backbone.pretrained = False
    print(f"Searching for d_model <= {max_d_model} to fit {cfg.max_parameters} params...")
    while low_idx <= high_idx:
        mid_idx = (low_idx + high_idx) // 2
        mid_size = mid_idx * width_multiple
        cfg.model.d_model = mid_size
        with torch.device("meta"):
            model = task.build_model(cfg)
            parameter_count = get_parameter_count(model)
        log.debug(f"  d_model={mid_size}: {parameter_count} params")
        if parameter_count <= cfg.max_parameters:
            best_size, best_count = mid_size, parameter_count
            low_idx = mid_idx + 1
        else:
            high_idx = mid_idx - 1

    if best_size is None:
        floor_size = min_idx * width_multiple
        cfg.model.d_model = floor_size
        with torch.device("meta"):
            floor_count = get_parameter_count(task.build_model(cfg))
        banner = "!" * 80
        log.warning(f"\n{banner}\n!! max_parameters={cfg.max_parameters} CANNOT BE SATISFIED.\n!! Floor at d_model={floor_size} is {floor_count} params (backbone + cross-attention with fixed d_input contribute a fixed cost).\n!! Falling back to d_model={floor_size}. Increase max_parameters, shrink the backbone, or use mode=vector to drop cross-attention.\n{banner}")
    else:
        cfg.model.d_model = best_size
        log.info(f"max_parameters={cfg.max_parameters} resolved d_model={best_size} (total params: {best_count})")

    if has_pretrained:
        cfg.model.backbone.pretrained = pretrained_was
    return cfg
