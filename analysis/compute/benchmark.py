"""Shared Copy-100 full-sequence benchmark implementation."""

import time
from statistics import mean

import torch
from torch import nn


from models.cells.rnn import RNNCell
from models.cells.lstm import LSTMCell
from models.cells.fsrnn import FSRNNCell
from models.cells.ctm import ContinuousThoughtMachine
from models.cells.cmm import ContinuousMemoryMachine
from models.cells.dnc import DifferentiableNeuralComputerCell
from models.cells.lsam import LSAMCell
from models.cells.nam_tm import NAMTuringMachineCell
from models.cells.lstm_cmm import LSTMCMMCell
from models.cells.gated_delta_net import StateTrackingGatedDeltaNetCell
from models.cells.rmc import RelationalMemoryCore
from models.transformer import TransformerSequenceModel

# Copy-100 task constants
D_INPUT = 64  # d_input after LinearBackbone(12 -> 64)
D_RAW = 12  # copy_dim(10) + 2 control flags
D_OUTPUT = 10  # copy_dim
MAX_PARAMS = 500_000

# (name, cell_class, kwargs_fn, num_heads_for_divisibility)
# kwargs match the Copy-100 commands in scripts/algorithmic/copy/
MODEL_SPECS = [
    (
        "RNN",
        RNNCell,
        lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True),
        1,
    ),
    (
        "LSTM",
        LSTMCell,
        lambda d: dict(d_input=D_INPUT, d_model=d, dropout=0.0, layer_norm=True),
        1,
    ),
    (
        "FSRNN",
        FSRNNCell,
        lambda d: dict(
            d_input=D_INPUT, d_model=d, fast_layers=2, dropout=0.0, layer_norm=True
        ),
        1,
    ),
    (
        "CTM",
        ContinuousThoughtMachine,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            memory_length=10,
            dropout=0.0,
            synapse_type="mlp",
            synapse_depth=8,
        ),
        1,
    ),
    (
        "CMM",
        ContinuousMemoryMachine,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            memory_length=10,
            dropout=0.0,
            num_long_term_memory_tokens=128,
            memory_transformer_layers=1,
            num_heads=4,
            synapse_type="mlp",
            synapse_depth=8,
        ),
        4,
    ),
    (
        "DNC",
        DifferentiableNeuralComputerCell,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            controller_type="LSTM",
            memory_n=128,
            memory_m=20,
            num_read_heads=1,
            num_write_heads=1,
            dropout=0.0,
        ),
        1,
    ),
    (
        "LSAM",
        LSAMCell,
        lambda d: dict(d_input=D_INPUT, d_model=d, num_heads=4, dropout=0.0),
        4,
    ),
    (
        "NAMTM",
        NAMTuringMachineCell,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            controller_type="LSTM",
            num_tapes=4,
            tape_length=32,
            head_dim=64,
            dropout=0.0,
        ),
        1,
    ),
    (
        "GDNStateTracking",
        StateTrackingGatedDeltaNetCell,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            num_heads=4,
            expand_k=0.75,
            expand_v=1.5,
            conv_size=4,
            qk_norm="l2",
            dropout=0.0,
        ),
        4,
    ),
    (
        "RMC",
        RelationalMemoryCore,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            mem_slots=32,
            num_heads=4,
            num_blocks=1,
            gate_style="unit",
            dropout=0.0,
        ),
        4,
    ),
    (
        "LSTM_CMM",
        LSTMCMMCell,
        lambda d: dict(
            d_input=D_INPUT,
            d_model=d,
            dropout=0.0,
            layer_norm=True,
            num_long_term_memory_tokens=32,
            memory_transformer_layers=1,
            num_heads=4,
        ),
        4,
    ),
]


TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FEEDFORWARD = 384


def _model_overhead(d_model):
    """Parameters from backbone (LazyLinear 12->64) and output projection (Linear d_model->10)."""
    return D_RAW * D_INPUT + D_INPUT + d_model * D_OUTPUT + D_OUTPUT


def _expand_state(state, batch_size, device):
    return tuple(
        s.to(device).expand(batch_size, *(-1,) * (s.ndim - 1)).contiguous()
        for s in state
    )


def _count_params(cell):
    """Count parameters after lazy modules have been materialised."""
    return sum(p.numel() for p in cell.parameters())


def _materialise_lazy(cell):
    """Single dummy forward to materialise any LazyLinear weights."""
    x = torch.randn(1, D_INPUT)
    state = _expand_state(cell.initialize_state(), 1, "cpu")
    with torch.no_grad():
        cell(x, state)


def find_param_matched_d_model(name, cls, kw_fn, num_heads, max_params):
    low, high = max(1, 8 // num_heads), 1024 // num_heads
    best_d, best_count = None, None
    while low <= high:
        mid = (low + high) // 2
        d = mid * num_heads
        try:
            cell = cls(**kw_fn(d))
            _materialise_lazy(cell)
            total = _count_params(cell) + _model_overhead(d)
            if total <= max_params:
                best_d, best_count = d, total
                low = mid + 1
            else:
                high = mid - 1
            del cell
        except Exception:
            high = mid - 1
    return best_d, best_count


class FullSequenceModel(nn.Module):
    """Copy-task model used to compare complete sequence execution."""

    def __init__(self, core, d_model, recurrent):
        super().__init__()
        self.backbone = nn.Linear(D_RAW, D_INPUT)
        self.core = core
        self.output_proj = nn.Linear(d_model, D_OUTPUT)
        self.recurrent = recurrent

    def initial_state(self, batch_size, device):
        if not self.recurrent:
            return ()
        return _expand_state(self.core.initialize_state(), batch_size, device)

    def forward(self, inputs):
        features = self.backbone(inputs)
        if self.recurrent:
            state = self.initial_state(inputs.shape[1], inputs.device)
            outputs = []
            for element in features:
                hidden, state = self.core(element, state)
                outputs.append(hidden)
            hidden = torch.stack(outputs, dim=0)
        else:
            hidden, _ = self.core(features, state=())
        return self.output_proj(hidden)


def _build_transformer(d_model, max_seq_len):
    return TransformerSequenceModel(
        d_input=D_INPUT,
        d_model=d_model,
        nhead=TRANSFORMER_NUM_HEADS,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
        dropout=0.0,
        max_seq_len=max_seq_len,
        learn_pos_emb=False,
        causal=True,
    )


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_steps(fn, device, n_warmup, n_measure):
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_measure):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def profile_sequence(
    model, batch_size, sequence_length, device, n_warmup=3, n_measure=10
):
    """Profile a complete sequence, including backbone and output projection."""
    model = model.to(device)
    model.train()
    inputs = torch.randn(sequence_length, batch_size, D_RAW, device=device)

    with torch.no_grad():
        model(inputs)

    def fwd_sequence():
        with torch.no_grad():
            model(inputs)

    def fwd_bwd_sequence():
        model.zero_grad(set_to_none=True)
        output = model(inputs)
        output.sum().backward()

    fwd_times = _time_steps(fwd_sequence, device, n_warmup, n_measure)
    fwd_bwd_times = _time_steps(fwd_bwd_sequence, device, n_warmup, n_measure)

    state_mem_kb = 0.0
    if model.recurrent:
        state = model.initial_state(1, device)
        state_floats = sum(s.numel() for s in state)
        state_mem_kb = state_floats * 4 / 1024

    peak_mem_mb = 0.0
    if device.startswith("cuda"):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd_sequence()
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    device_name = (
        torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda")
        else "CPU"
    )
    return {
        "params": _count_params(model),
        "fwd_ms": mean(fwd_times) * 1000,
        "fwd_bwd_ms": mean(fwd_bwd_times) * 1000,
        "state_mem_kb": state_mem_kb,
        "peak_mem_mb": peak_mem_mb,
        "device": device_name,
    }
