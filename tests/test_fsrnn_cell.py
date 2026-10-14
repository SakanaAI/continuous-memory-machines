from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from models import construct_rnn
from models.cells.fsrnn import FSRNNCell
from models.cells.lstm import LSTMCell


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _reference_lstm(cell, inputs, state):
    """NumPy LSTM equations using the local baseline's layer normalization."""
    def normalize(value, layer):
        return (value - value.mean(axis=1, keepdims=True)) / np.sqrt(value.var(axis=1, keepdims=True) + layer.eps) * _numpy(layer.weight) + _numpy(layer.bias)

    h, c = state
    recurrent = h @ _numpy(cell.weight_h)
    gates = normalize(recurrent, cell.ln_1) if cell.layer_norm else recurrent
    if inputs is not None:
        projected = inputs @ _numpy(cell.weight_x)
        gates = gates + (normalize(projected, cell.ln_2) if cell.layer_norm else projected)
    f, i, o, g = np.split(gates + _numpy(cell.bias), 4, axis=1)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    new_c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    new_h = sigmoid(o) * np.tanh(normalize(new_c, cell.ln_3) if cell.layer_norm else new_c)
    return new_h, new_c


@pytest.mark.parametrize("fast_layers", [2, 4])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_paper_topology_with_local_lstm_equations(fast_layers, layer_norm):
    torch.manual_seed(3)
    cell = FSRNNCell(d_input=3, d_model=5, d_slow=7, fast_layers=fast_layers, layer_norm=layer_norm,
                     dropout=0.3).double().eval()
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.uniform_(-0.4, 0.4)  # Exercise learned gate scales/biases and nonzero initial states
    fast_state = tuple(_numpy(value.expand(3, -1)) for value in cell.initialize_state()[:2])
    slow_state = tuple(_numpy(value.expand(3, -1)) for value in cell.initialize_state()[2:])
    state = None

    for inputs in torch.randn(5, 3, 3, dtype=torch.float64):
        fast_state = _reference_lstm(cell.fast_cells[0], _numpy(inputs), fast_state)
        slow_state = _reference_lstm(cell.slow_cell, fast_state[0], slow_state)
        fast_state = _reference_lstm(cell.fast_cells[1], slow_state[0], fast_state)
        for transition in cell.fast_cells[2:]:
            fast_state = _reference_lstm(transition, None, fast_state)

        output, state = cell(inputs, state)
        torch.testing.assert_close(output, torch.from_numpy(fast_state[0]), rtol=1e-10, atol=1e-10)
        for actual, expected in zip(state, (*fast_state, *slow_state), strict=True):
            torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("layer_norm", [False, True])
def test_backward_reaches_all_transitions_and_initial_states(layer_norm):
    torch.manual_seed(4)
    cell = FSRNNCell(d_input=3, d_model=5, d_slow=7, fast_layers=4, layer_norm=layer_norm,
                     dropout=0.1)
    inputs = torch.randn(4, 8, 3, requires_grad=True)
    state = None
    for element in inputs:
        output, state = cell(element, state)
    assert [value.shape for value in state] == [(8, 5), (8, 5), (8, 7), (8, 7)]
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad[0].abs().sum() > 0
    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_reuses_local_lstm_options_and_only_learns_two_initial_states():
    cell = FSRNNCell(d_input=3, d_model=5, d_slow=7, fast_layers=4, dropout=0.2, layer_norm=True, T_max=20)
    for transition in [*cell.fast_cells, cell.slow_cell]:
        assert type(transition) is LSTMCell
        assert transition.layer_norm and transition.dropout.p == 0.2 and transition.T_max == 20
        width = transition.d_model
        torch.testing.assert_close(transition.bias[:width], -transition.bias[width:2 * width])
    initial_parameters = {name for name, _ in cell.named_parameters() if name.endswith(("h_0", "c_0"))}
    assert initial_parameters == {"fast_cells.0.h_0", "fast_cells.0.c_0", "slow_cell.h_0", "slow_cell.c_0"}
    for transition in cell.fast_cells[2:]:
        assert transition.weight_x is None and transition.ln_2 is None


@pytest.mark.parametrize("mode", ["vector", "patches"])
def test_hydra_factory_sequence_continuation_and_gradients(mode):
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "conf")):
        cfg = compose(config_name="base", overrides=[
            "model=FSRNN", "task=Copy", "model.d_model=10", "model.d_input=8", "model.fast_layers=4",
            "model.d_slow=6", "model.layer_norm=false", "model.dropout=0.2", "model.T_max=20", f"model.backbone.mode={mode}", "+model.backbone.num_heads=2",
        ])
    model = construct_rnn(cfg.model).double().eval()
    cell = model.recurrent_module
    assert cell.fast_layers == 4 and cell.d_slow == 6 and not cell.layer_norm
    assert cell.T_max == 20
    assert all(transition.dropout.p == 0.2 and transition.T_max == 20 for transition in [*cell.fast_cells, cell.slow_cell])
    inputs = torch.randn((4, 2, 5, 8) if mode == "patches" else (4, 2, 8), dtype=torch.float64)
    state = model.get_initial_state(batch_size=2)
    assert all(value.dtype == inputs.dtype and value.shape[0] == 2 for value in state)
    if mode == "patches":
        assert torch.equal(state[-1], torch.zeros(2, 8, dtype=inputs.dtype))
        direct_output, direct_state = cell(torch.randn(2, 8, dtype=inputs.dtype), state=None)
        torch.testing.assert_close(direct_state[-1], cell.query_proj(direct_state[0]))
        assert direct_output.shape == (2, 10)

    outputs, final_state = model(inputs, state)
    first, middle_state = model(inputs[:2], state)
    second, chunked_state = model(inputs[2:], middle_state)
    assert outputs.shape == (4, 2, 10)
    torch.testing.assert_close(outputs, torch.cat([first, second]))
    for full, chunked in zip(final_state, chunked_state, strict=True):
        torch.testing.assert_close(full, chunked)
    outputs[-1].square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if mode == "patches":
        assert model.cross_attn.q_proj.weight.shape == (8, 8)
        assert cell.query_proj.weight.grad.abs().sum() > 0


def test_default_slow_width_scales_on_meta_device_for_parameter_matching():
    counts = []
    with torch.device("meta"):
        for width in (8, 16):
            cell = FSRNNCell(d_input=3, d_model=width, fast_layers=4)
            output, state = cell(torch.empty(2, 3), state=None)
            assert cell.d_slow == width and output.shape == (2, width)
            assert all(value.shape == (2, width) for value in state)
            counts.append(sum(parameter.numel() for parameter in cell.parameters()))
    assert counts[1] > counts[0]


@pytest.mark.parametrize("overrides", [{"fast_layers": 1}, {"d_slow": 0}, {"dropout": 1}, {"T_max": 1}])
def test_invalid_configuration(overrides):
    with pytest.raises(AssertionError):
        FSRNNCell(d_input=3, d_model=5, **overrides)


def test_compiled_cell_matches_eager():
    cell = FSRNNCell(d_input=3, d_model=5, d_slow=7, fast_layers=4).eval()
    inputs = torch.randn(2, 3)
    expected_output, expected_state = cell(inputs, state=None)
    compiled = torch.compile(cell, backend="eager", fullgraph=True)
    output, state = compiled(inputs, state=None)
    torch.testing.assert_close(output, expected_output)
    for actual, expected in zip(state, expected_state, strict=True):
        torch.testing.assert_close(actual, expected)
