import inspect
from omegaconf import DictConfig
from torch import nn

from models.base import RecurrentModel, CrossAttentionRecurrentModel
from models.cells.rnn import RNNCell
from models.cells.lstm import LSTMCell
from models.cells.fsrnn import FSRNNCell, CrossAttentionFSRNNCell
from models.cells.ctm import (
    ContinuousThoughtMachine,
    CrossAttentionContinuousThoughtMachine,
)
from models.cells.cmm import (
    ContinuousMemoryMachine,
    CrossAttentionContinuousMemoryMachine,
)
from models.cells.ctm_transformer import (
    CTMTransformerCell,
    CrossAttentionCTMTransformerCell,
)
from models.cells.lstm_cmm import LSTMCMMCell, CrossAttentionLSTMCMMCell
from models.cells.dnc import DifferentiableNeuralComputerCell
from models.cells.lsam import LSAMCell, CrossAttentionLSAMCell
from models.cells.nam_tm import NAMTuringMachineCell
from models.cells.gated_delta_net import (
    GatedDeltaNetCell,
    CrossAttentionGatedDeltaNetCell,
    StateTrackingGatedDeltaNetCell,
    CrossAttentionStateTrackingGatedDeltaNetCell,
)
from models.cells.rmc import RelationalMemoryCore, CrossAttentionRelationalMemoryCore
from models.transformer import CausalTransformerCell, TransformerSequenceModel
from models.rmt import RecurrentMemoryTransformer, RMTCell
from models.backbones import (
    ViT5Backbone,
    LinearBackbone,
    ResNetBackbone,
)

_CELLS = {
    "RNN": RNNCell,
    "LSTM": LSTMCell,
    "FSRNN": FSRNNCell,
    "CTM": ContinuousThoughtMachine,
    "CMM": ContinuousMemoryMachine,
    "CTMTransformer": CTMTransformerCell,
    "DNC": DifferentiableNeuralComputerCell,
    "LSAM": LSAMCell,
    "NAMTM": NAMTuringMachineCell,
    "LSTM_CMM": LSTMCMMCell,
    "GDN": GatedDeltaNetCell,
    "GDNStateTracking": StateTrackingGatedDeltaNetCell,
    "RMC": RelationalMemoryCore,
}


def construct_rnn(model_cfg: DictConfig) -> nn.Module:
    if model_cfg.model == "RMT":
        valid_params = set(inspect.signature(RecurrentMemoryTransformer.__init__).parameters) - {"self"}
        model = RecurrentMemoryTransformer(**{
            k: v for k, v in model_cfg.items() if k in valid_params
        })
        if model_cfg.get("backbone", {}).get("mode") == "patches":
            return CrossAttentionRecurrentModel(
                RMTCell(model), num_heads=model_cfg.backbone.num_heads
            )
        return model
    if model_cfg.model == "Transformer":
        patch_mode = (
            hasattr(model_cfg, "backbone")
            and model_cfg.backbone.mode == "patches"
        )
        if patch_mode:
            if not model_cfg.causal:
                raise ValueError(
                    "Patch-mode Transformer uses the internal tick history and requires model.causal=true."
                )
            cell = CausalTransformerCell(
                d_input=model_cfg.d_input,
                d_model=model_cfg.d_model,
                nhead=model_cfg.nhead,
                num_layers=model_cfg.num_layers,
                dim_feedforward=model_cfg.dim_feedforward,
                dropout=model_cfg.dropout,
                max_seq_len=model_cfg.max_seq_len,
                learn_pos_emb=model_cfg.learn_pos_emb,
            )
            return CrossAttentionRecurrentModel(
                cell, num_heads=model_cfg.backbone.num_heads
            )
        return TransformerSequenceModel(
            d_input=model_cfg.d_input,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
            max_seq_len=model_cfg.max_seq_len,
            learn_pos_emb=model_cfg.learn_pos_emb,
            causal=model_cfg.causal,
        )
    cell_cls = _CELLS[model_cfg.model]
    if model_cfg.backbone.mode == "patches":
        if model_cfg.model == "CTM":
            cell_cls = CrossAttentionContinuousThoughtMachine
        elif model_cfg.model == "CMM":
            cell_cls = CrossAttentionContinuousMemoryMachine
        elif model_cfg.model == "CTMTransformer":
            cell_cls = CrossAttentionCTMTransformerCell
        elif model_cfg.model == "RMC":
            cell_cls = CrossAttentionRelationalMemoryCore
        elif model_cfg.model == "LSAM":
            cell_cls = CrossAttentionLSAMCell
        elif model_cfg.model == "GDN":
            cell_cls = CrossAttentionGatedDeltaNetCell
        elif model_cfg.model == "GDNStateTracking":
            cell_cls = CrossAttentionStateTrackingGatedDeltaNetCell
        elif model_cfg.model == "LSTM_CMM":
            cell_cls = CrossAttentionLSTMCMMCell
        elif model_cfg.model == "FSRNN":
            cell_cls = CrossAttentionFSRNNCell
    valid_params = set(inspect.signature(cell_cls.__init__).parameters) - {"self"}
    cell_kwargs = {
        k: v
        for k, v in model_cfg.items()
        if k in valid_params and k not in ("d_model", "d_input")
    }
    cell = cell_cls(d_input=model_cfg.d_input, d_model=model_cfg.d_model, **cell_kwargs)
    if model_cfg.backbone.mode == "patches":
        return CrossAttentionRecurrentModel(
            cell, num_heads=model_cfg.backbone.num_heads
        )
    return RecurrentModel(cell)


def construct_backbone(backbone_cfg: DictConfig, d_input: int) -> nn.Module:
    backbone_type = backbone_cfg.type
    if backbone_type == "vit5":
        assert d_input == 384, "For ViT5 backbone, d_input must be 384."
        output_type = "patches" if backbone_cfg.mode == "patches" else "vector"
        return ViT5Backbone(output_type=output_type, pretrained=backbone_cfg.pretrained)
    elif backbone_type == "linear":
        return LinearBackbone(out_dim=d_input)
    elif backbone_type == "resnet":
        return ResNetBackbone(
            in_channels=3,
            num_blocks=backbone_cfg.num_blocks,
            stride=2,
            arch=backbone_cfg.arch,
            do_initial_pool=backbone_cfg.pool,
            output_type=backbone_cfg.mode,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
