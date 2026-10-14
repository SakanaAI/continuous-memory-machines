import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from models import construct_rnn
from models.base import RecurrentModel
from models.cells.gated_delta_net import (
    GatedDeltaNetCell,
    StateTrackingGatedDeltaNetCell,
    CrossAttentionStateTrackingGatedDeltaNetCell,
)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


@pytest.mark.parametrize("cell_cls", [GatedDeltaNetCell, StateTrackingGatedDeltaNetCell])
@pytest.mark.parametrize("qk_norm", ["l2", "softmax"])
@pytest.mark.parametrize("conv_size", [1, 3, 4])
def test_memory_matches_independent_matrix_recurrence(cell_cls, qk_norm, conv_size):
    """Check the affine transition against explicit matrices over several steps."""
    torch.manual_seed(11)
    cell = cell_cls(d_input=5, d_model=8, num_heads=2, conv_size=conv_size, qk_norm=qk_norm).eval()
    extended = cell_cls is StateTrackingGatedDeltaNetCell
    state = tuple(torch.randn_like(value.expand(3, *value.shape[1:])) for value in cell.initialize_state())
    memory, *conv_states = [value.copy() for value in map(_numpy, state)]

    for inputs in torch.randn(6, 3, 5):
        projected = []
        for index, (proj, weight) in enumerate(zip(
            (cell.q_proj, cell.k_proj, cell.v_proj),
            (cell.q_conv_weight, cell.k_conv_weight, cell.v_conv_weight),
            strict=True,
        )):
            raw = _numpy(inputs) @ _numpy(proj.weight).T
            conv_states[index] = np.concatenate([conv_states[index][:, :, 1:], raw[:, :, None]], axis=-1)
            convolved = (conv_states[index] * _numpy(weight)).sum(axis=-1)
            activated = convolved / (1 + np.exp(-convolved))
            projected.append(activated.reshape(3, cell.num_heads, -1))
        q, k, v = projected
        if qk_norm == "l2":
            q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        else:
            def softmax(value):
                exp = np.exp(value - value.max(axis=-1, keepdims=True))
                return exp / exp.sum(axis=-1, keepdims=True)
            q, k = softmax(q), softmax(k)

        logits = _numpy(inputs) @ _numpy(cell.b_proj.weight).T + _numpy(cell.b_proj.bias)
        beta = (2.0 if extended else 1.0) / (1 + np.exp(-logits))
        decay_logits = _numpy(inputs) @ _numpy(cell.gk_proj.weight).T + _numpy(cell.dt_bias)
        alpha = np.exp(-np.exp(_numpy(cell.A_log)) * np.logaddexp(0, decay_logits))
        outer = k[..., :, None] * k[..., None, :]
        identity = np.eye(cell.head_qk_dim)
        if extended:
            transition = alpha[..., None, None] * (identity - beta[..., None, None] * outer)
        else:
            transition = alpha[..., None, None] * identity - beta[..., None, None] * outer
        memory = transition @ memory + beta[..., None, None] * k[..., :, None] * v[..., None, :]

        read = (memory * q[..., :, None]).sum(axis=-2)
        normalized = read / np.sqrt((read ** 2).mean(axis=-1, keepdims=True) + cell.g_norm.eps)
        gate = (_numpy(inputs) @ _numpy(cell.g_proj.weight).T).reshape(3, cell.num_heads, -1)
        gated = normalized * _numpy(cell.g_norm.weight) * gate / (1 + np.exp(-gate))
        expected_output = gated.reshape(3, cell.value_dim) @ _numpy(cell.o_proj.weight).T

        output, state = cell(inputs, state)
        np.testing.assert_allclose(_numpy(state[0]), memory, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(_numpy(output), expected_output, rtol=3e-5, atol=3e-6)
        for actual, expected in zip(state[1:], conv_states, strict=True):
            np.testing.assert_allclose(_numpy(actual), expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("alpha", [0.2, 0.9])
@pytest.mark.parametrize("beta", [0.2, 1.8])
def test_measured_transition_has_bounded_eigenvalues(alpha, beta):
    """Recover the transition through forward() by subtracting the affine write."""
    torch.manual_seed(12)
    cell = StateTrackingGatedDeltaNetCell(
        d_input=4, d_model=4, num_heads=1, expand_k=1, expand_v=1, conv_size=1,
    ).eval()
    with torch.no_grad():
        cell.b_proj.weight.zero_()
        cell.b_proj.bias.fill_(math.log(beta / (2 - beta)))
        cell.gk_proj.weight.zero_()
        cell.dt_bias.fill_(math.log(math.expm1(1.0)))  # softplus(dt_bias) = 1
        cell.A_log.fill_(math.log(-math.log(alpha)))
        inputs = torch.randn(1, 4)
        state = cell.initialize_state()
        _, zero_state = cell(inputs, state)
        _, identity_state = cell(inputs, (torch.eye(4).reshape(1, 1, 4, 4), *state[1:]))
        transition = (identity_state[0] - zero_state[0])[0, 0]
        eigenvalues = torch.linalg.eigvalsh(transition)

    expected = torch.tensor([alpha * (1 - beta), alpha, alpha, alpha])
    torch.testing.assert_close(eigenvalues, expected, rtol=1e-5, atol=1e-6)
    assert eigenvalues.abs().max() <= 1
    if beta > 1:
        assert eigenvalues.min() < 0


def test_constructed_parity_over_long_sequences():
    """The full cell can toggle a state on ones and retain it on zeros."""
    cell = StateTrackingGatedDeltaNetCell(
        d_input=2, d_model=1, num_heads=1, expand_k=1, expand_v=1, conv_size=1,
    ).eval()
    with torch.no_grad():
        for proj in (cell.q_proj, cell.k_proj, cell.v_proj):
            proj.weight.fill_(1.0)
        for weight in (cell.q_conv_weight, cell.k_conv_weight, cell.v_conv_weight):
            weight.fill_(1.0)
        cell.b_proj.weight.copy_(torch.tensor([[-100.0, 100.0]]))
        cell.b_proj.bias.zero_()
        cell.A_log.fill_(-100.0)  # alpha rounds to 1 in float32
        cell.gk_proj.weight.zero_()

        generator = torch.Generator().manual_seed(13)
        bits = torch.randint(0, 2, (1024, 3), generator=generator)
        bits[:4] = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]])
        inputs = torch.nn.functional.one_hot(bits, num_classes=2).float()
        expected_parity = bits.cumsum(dim=0).remainder(2)
        state = RecurrentModel(cell).get_initial_state(batch_size=3)
        one_state = 2 * torch.nn.functional.silu(torch.tensor(1.0))
        for tokens, parity in zip(inputs, expected_parity, strict=True):
            _, state = cell(tokens, state)
            torch.testing.assert_close(state[0].flatten(), parity.float() * one_state, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", ["vector", "patches"])
def test_hydra_factory_continuation_checkpoint_and_gradients(mode):
    torch.manual_seed(14)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "conf")):
        cfg = compose(config_name="base", overrides=[
            "model=GDNStateTracking", "task=Parity", "model.d_input=8", "model.d_model=12",
            "model.num_heads=2", "model.expand_k=1", "model.expand_v=1", "model.conv_size=3",
            "model.dropout=0.2", f"model.backbone.mode={mode}", "+model.backbone.num_heads=2",
        ])
    model = construct_rnn(cfg.model).eval()
    cell = model.recurrent_module
    expected_class = CrossAttentionStateTrackingGatedDeltaNetCell if mode == "patches" else StateTrackingGatedDeltaNetCell
    assert type(cell) is expected_class
    assert (cell.num_heads, cell.key_dim, cell.value_dim, cell.conv_size) == (2, 12, 12, 3)
    assert cell.dropout_layer.p == 0.2
    # The variant has the same parameters and checkpoint layout as the existing GDN.
    cfg.model.model = "GDN"
    baseline = construct_rnn(cfg.model)
    baseline.recurrent_module.load_state_dict(cell.state_dict(), strict=True)

    inputs = torch.randn((5, 3, 4, 8) if mode == "patches" else (5, 3, 8), requires_grad=True)
    state = model.get_initial_state(batch_size=3)
    outputs, final_state = model(inputs, state)
    first, middle = model(inputs[:2], state)
    second, resumed = model(inputs[2:], middle)
    assert outputs.shape == (5, 3, 12)
    torch.testing.assert_close(outputs, torch.cat([first, second]))
    for full, chunked in zip(final_state, resumed, strict=True):
        torch.testing.assert_close(full, chunked)
    assert len(final_state) == (5 if mode == "patches" else 4)
    if mode == "patches":
        torch.testing.assert_close(final_state[-1], outputs[-1])
        assert model.cross_attn.q_proj.in_features == 12

    model.train()
    training_outputs, _ = model(inputs, model.get_initial_state(batch_size=3))
    training_outputs[-1].square().mean().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad[0].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_compiled_cell_matches_eager():
    cell = StateTrackingGatedDeltaNetCell(d_input=4, d_model=8, num_heads=2).eval()
    inputs = torch.randn(1, 4)
    _, state = cell(inputs, state=None)
    expected_output, expected_state = cell(inputs, state)
    compiled = torch.compile(cell, backend="eager", fullgraph=True)
    output, final_state = compiled(inputs, state)
    torch.testing.assert_close(output, expected_output)
    for actual, expected in zip(final_state, expected_state, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA Inductor")
@pytest.mark.parametrize("cell_cls", [GatedDeltaNetCell, StateTrackingGatedDeltaNetCell])
def test_cuda_inductor_sequence_training_and_inference_match_eager(cell_cls):
    """Exercise real code generation across the train/eval/train transition.

    Eight steps roll the convolution history over twice. The old slice/cat
    update raised 'Expected a number but got Add' when compiling inference;
    a single-cell test with backend="eager" cannot catch that failure.
    """
    torch.manual_seed(15)
    torch._dynamo.reset()
    reference = RecurrentModel(cell_cls(d_input=4, d_model=8, num_heads=2)).cuda()
    candidate = copy.deepcopy(reference)
    compiled = torch.compile(candidate, backend="inductor", mode="reduce-overhead", fullgraph=True)
    inputs = torch.randn(8, 2, 4, device="cuda")
    try:
        for training in (True, False, True):
            reference.train(training)
            compiled.train(training)
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.enable_grad() if training else torch.inference_mode():
                reference_inputs = inputs.clone().requires_grad_(training)
                candidate_inputs = inputs.clone().requires_grad_(training)
                expected, expected_state = reference(reference_inputs, reference.get_initial_state(2))
                actual, actual_state = compiled(candidate_inputs, candidate.get_initial_state(2))
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-5)
                if training:
                    (expected.square().mean() + expected_state[0].square().mean()).backward()
                    (actual.square().mean() + actual_state[0].square().mean()).backward()
                    torch.testing.assert_close(candidate_inputs.grad, reference_inputs.grad, rtol=2e-4, atol=2e-5)
                    for expected_param, actual_param in zip(reference.parameters(), candidate.parameters(), strict=True):
                        assert actual_param.grad is not None
                        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=2e-4, atol=2e-5)
    finally:
        torch._dynamo.reset()
