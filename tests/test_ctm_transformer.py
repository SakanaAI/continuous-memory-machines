import torch

from models.cells.ctm_transformer import CTMTransformerCell


def test_forward_shapes_state_and_backward():
    cell = CTMTransformerCell(
        d_input=8,
        d_model=16,
        memory_length=12,
        dropout=0.0,
        memory_transformer_layers=1,
        num_heads=4,
    )
    batch_size = 3
    state = tuple(
        value.expand(batch_size, *(-1,) * (value.ndim - 1))
        for value in cell.initialize_state()
    )

    output, next_state = cell(torch.randn(batch_size, 8), state)

    assert output.shape == (batch_size, 16)
    assert next_state[1].shape == (batch_size, 16, 12)
    assert len(next_state) == 5
    assert cell.memory_updater.num_ltm_tokens == 0

    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in cell.parameters()
        if parameter.requires_grad
    )
