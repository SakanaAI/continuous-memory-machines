import pytest
import torch
from omegaconf import OmegaConf

from models import construct_rnn


@pytest.mark.parametrize("model_name", ["RMC", "LSTM_CMM"])
def test_projected_queries_preserve_factory_overrides_and_gradients(model_name):
    overrides = {
        "RMC": {
            "mem_slots": 3, "num_heads": 2, "head_size": 3, "key_size": 2,
            "num_blocks": 2, "gate_style": "memory", "forget_bias": 0.5,
            "input_bias": -0.5, "attention_mlp_layers": 3,
        },
        "LSTM_CMM": {
            "num_long_term_memory_tokens": 3, "num_heads": 1, "memory_transformer_layers": 2,
            "memory_update_gate": True, "memory_update_gate_init": 0.25,
            "layer_norm": False, "T_max": 20,
        },
    }[model_name]
    cfg = OmegaConf.create({
        "model": model_name, "d_input": 8, "d_model": 14, "dropout": 0.2,
        "backbone": {"mode": "patches", "num_heads": 2}, **overrides,
    })
    model = construct_rnn(cfg).double().eval()
    cell = model.recurrent_module
    if model_name == "RMC":
        for name, value in overrides.items():
            assert getattr(cell, name) == value
        assert cell.dropout_layer.p == 0.2
        assert cell.query_proj.in_features == 18  # 3 slots * 2 heads * head_size 3
    else:
        assert cell.start_ltm.shape == (3, 14)
        assert len(cell.memory_updater.blocks) == 2
        assert cell.layer_norm is False
        assert cell.T_max == 20
        assert cell.dropout.p == 0.2
        for block in cell.memory_updater.blocks:
            assert block.num_heads == 1
            assert block.memory_update_gate.item() == 0.25
            assert block.dropout.p == 0.2

    inputs = torch.randn(3, 2, 5, 8, dtype=torch.float64)
    states = model.get_initial_state(batch_size=2)
    assert states[-1].dtype == inputs.dtype
    assert torch.equal(states[-1], torch.zeros(2, 8, dtype=inputs.dtype))
    outputs, states = model(inputs, states)
    assert outputs.shape == (3, 2, 14)
    assert states[-1].shape == (2, 8)
    assert model.cross_attn.q_proj.weight.shape == (8, 8)

    # Later predictions must train the query projection through patch attention.
    outputs[-1].square().mean().backward()
    grad = cell.query_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_rmc_query_is_independent_of_prediction_projection():
    cfg = OmegaConf.create({
        "model": "RMC", "d_input": 8, "d_model": 12, "mem_slots": 3, "num_heads": 2,
        "backbone": {"mode": "patches", "num_heads": 2},
    })
    cell = construct_rnn(cfg).recurrent_module.eval()
    inputs = torch.randn(2, 8)
    with torch.no_grad():
        output_before, (memory_before, query_before) = cell(inputs, state=None)
        cell.output_proj.bias.add_(1.0)
        output_after, (memory_after, query_after) = cell(inputs, state=None)
    torch.testing.assert_close(output_after, output_before + 1)
    torch.testing.assert_close(memory_after, memory_before)
    torch.testing.assert_close(query_after, query_before)
