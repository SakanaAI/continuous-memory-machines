import torch

from models.cells.lstm import LSTMCell


def test_forward_shapes_and_statefulness():
    cell = LSTMCell(d_input=8, d_model=16)
    x = torch.randn(4, 8)

    h_1, (h_state, c_state) = cell(x, state=None)  # state=None should use learnt h_0/c_0 broadcast to batch
    assert h_1.shape == (4, 16) and h_state.shape == (4, 16) and c_state.shape == (4, 16)

    h_2, _ = cell(x, state=(h_state, c_state))  # feeding state back in should change the output
    assert not torch.allclose(h_1, h_2)


def test_forget_gate_bias_init():
    cell_default = LSTMCell(d_input=4, d_model=8)
    assert torch.equal(cell_default.bias[:8], torch.ones(8))  # forget gate bias initialised to 1

    cell_chrono = LSTMCell(d_input=4, d_model=8, T_max=100)
    assert torch.equal(cell_chrono.bias[:8], -cell_chrono.bias[8:16])  # chrono init: f bias == -i bias


def test_layer_norm_path_and_backward():
    cell = LSTMCell(d_input=8, d_model=16, layer_norm=True, dropout=0.1)
    h_1, _ = cell(torch.randn(4, 8), state=None)
    h_1.sum().backward()
    assert torch.isfinite(h_1).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
