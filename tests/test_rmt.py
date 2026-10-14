import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from models import construct_rnn
from models.base import CrossAttentionRecurrentModel, RecurrentModel
from models.rmt import RecurrentMemoryTransformer, RMTCell
from tasks.image_classification.imagenet import ImageNet100Task
from tasks.recall.associative_recall import AssociativeRecallTask
from tasks.recall.copy import CopyTask
from tasks.recall.parity import ParityTask
from tasks.recall.priority_sort import PrioritySortTask
from tasks.utils import get_parameter_count, optimise_max_d_model


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/recurrent_memory_transformer/code/LM-RMT/pytorch"


def _model(**overrides):
    kwargs = dict(d_input=4, d_model=8, num_heads=2, num_layers=2,
                  segment_length=3, num_mem_tokens=2, dropout=0.0)
    return RecurrentMemoryTransformer(**(kwargs | overrides))


def test_rmt_is_causal_within_and_across_segments():
    torch.manual_seed(0)
    model = _model().eval()
    inputs = torch.randn(8, 2, 4)
    with torch.no_grad():
        expected, _ = model(inputs)
        # Cover the first segment, a boundary, and the final short segment.
        for prefix_length in (1, 3, 5, 7):
            changed = inputs.clone()
            changed[prefix_length:] += torch.randn_like(changed[prefix_length:]) * 10
            actual, _ = model(changed)
            torch.testing.assert_close(actual[:prefix_length], expected[:prefix_length])


def test_rmt_carries_memory_and_backpropagates_across_segments():
    torch.manual_seed(1)
    model = _model()
    inputs = torch.randn(8, 2, 4, requires_grad=True)
    initial_state = model.get_initial_state(2)
    initial_memory = initial_state[0].detach().clone()
    outputs, state = model(inputs, initial_state)
    assert outputs.shape == (8, 2, 8)
    assert state[0].shape == (2, 2, 8)
    torch.testing.assert_close(initial_state[0], initial_memory)
    assert not torch.allclose(state[0], initial_memory)

    # A final-token loss must reach inputs that can only be accessed via memory.
    outputs[-1, :, 0].sum().backward()
    assert inputs.grad[:3].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_rmt_segment_aligned_streaming_and_reset():
    torch.manual_seed(2)
    model = _model().eval()
    inputs = torch.randn(8, 2, 4)
    with torch.no_grad():
        whole, whole_state = model(inputs)
        first, state = model(inputs[:3])
        remaining, state = model(inputs[3:], state)
        torch.testing.assert_close(torch.cat((first, remaining)), whole)
        torch.testing.assert_close(state[0], whole_state[0])
        reset, _ = model(inputs[3:], model.get_initial_state(2))
        assert not torch.allclose(remaining, reset)
        repeated, _ = model(inputs)
        torch.testing.assert_close(repeated, whole)
        empty, unchanged = model(inputs[:0], state)
        assert empty.shape == (0, 2, 8)
        torch.testing.assert_close(unchanged[0], state[0])


def test_rmt_supports_lengths_beyond_training_segments_and_checkpoint_round_trip():
    torch.manual_seed(3)
    model = _model(segment_length=16).eval()
    restored = _model(segment_length=16).eval()
    restored.load_state_dict(model.state_dict())
    inputs = torch.randn(1001, 1, 4)
    with torch.no_grad():
        expected, state = model(inputs)
        actual, actual_state = restored(inputs)
    assert actual.shape == (1001, 1, 8)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state[0], state[0])


@pytest.mark.parametrize("overrides", [
    {"segment_length": 0}, {"num_mem_tokens": 0}, {"num_layers": 0},
    {"num_heads": 3}, {"d_model": 7}, {"attention_dropout": 1.0},
])
def test_rmt_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        _model(**overrides)


@pytest.mark.skipif(not (REFERENCE / "mem_transformer.py").exists(), reason="Local LM-RMT reference is unavailable")
@pytest.mark.parametrize("pre_lnorm", [False, True])
def test_rmt_matches_reference_outputs_memory_and_gradients(monkeypatch, pre_lnorm):
    # Run the authors' implementation directly; it is deliberately not a runtime
    # dependency. Compare multiple segments, including a short final segment.
    monkeypatch.syspath_prepend(str(REFERENCE / "utils"))
    spec = importlib.util.spec_from_file_location("lm_rmt_reference", REFERENCE / "mem_transformer.py")
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    torch.manual_seed(4)
    model = _model(pre_lnorm=pre_lnorm).double()
    reference = reference_module.MemTransformerLM(
        n_token=4, n_layer=2, n_head=2, d_model=8, d_head=4, d_inner=16,
        dropout=0.0, dropatt=0.0, pre_lnorm=pre_lnorm,
        tgt_len=3, ext_len=0, mem_len=0, num_mem_tokens=2,
        mem_at_end=True, attn_type=0,
    ).double()
    reference.layers.load_state_dict(model.layers.state_dict())
    with torch.no_grad():
        reference.r_w_bias.copy_(model.r_w_bias)
        reference.r_r_bias.copy_(model.r_r_bias)
        reference.mem_tokens.copy_(model.mem_tokens.transpose(0, 1))
    # Replace only the vocabulary embedding; both models see identical features.
    reference.word_emb = nn.Identity()
    projection = copy.deepcopy(model.input_proj)
    inputs = torch.randn(8, 2, 4, dtype=torch.float64, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    actual, state = model(inputs)
    memory = reference.mem_tokens.expand(-1, 2, -1)
    expected = []
    for segment in (projection(ref_inputs) * model.embedding_scale).split(3):
        hidden, _ = reference._forward(segment, mem_tokens=memory)
        expected.append(hidden[2:-2])
        memory = hidden[-2:]
    expected = torch.cat(expected)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(state[0], memory.transpose(0, 1), atol=1e-10, rtol=1e-8)

    actual[..., 0].sum().backward()
    expected[..., 0].sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(model.mem_tokens.grad, reference.mem_tokens.grad.transpose(0, 1), atol=1e-10, rtol=1e-8)
    for name, parameter in model.layers.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.layers.get_parameter(name).grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("task_name,task_cls,budget,segment_length,memory_tokens,overrides", [
    ("Copy", CopyTask, 500000, 10, 128, ["task.copy_delay=0", "task.copy_min_sequence_length=100", "task.copy_max_sequence_length=100", "task.copy_dim=10"]),
    ("PrioritySort", PrioritySortTask, 500000, 20, 128, []),
    ("Parity", ParityTask, 60000, 4, 4, []),
    ("AssociativeRecall", AssociativeRecallTask, 500000, 10, 32, []),
])
def test_rmt_task_integration_and_parameter_matching(task_name, task_cls, budget, segment_length, memory_tokens, overrides):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="base", overrides=[
            "model=RMT", f"task={task_name}", f"max_parameters={budget}",
            f"model.segment_length={segment_length}", f"model.num_mem_tokens={memory_tokens}",
            "model.num_layers=2", "device=-1", "training.compile=false", *overrides,
        ])
    # Build/search without constructing full datasets or optimizers.
    task = object.__new__(task_cls)
    optimise_max_d_model(task, cfg)
    with torch.device("meta"):
        count = get_parameter_count(task.build_model(cfg))
    assert count <= budget
    cfg.model.d_model += cfg.model.num_heads
    with torch.device("meta"):
        assert get_parameter_count(task.build_model(cfg)) > budget
    cfg.model.d_model -= cfg.model.num_heads

    model = task.build_model(cfg)
    inputs = torch.randn(2, segment_length + 1, task._pseudo_input_shape(cfg)[1])
    logits, state = model(inputs, model.get_initial_states(2))
    assert logits.shape == (2, segment_length + 1, task.d_output(cfg))
    assert state[0].shape == (2, memory_tokens, cfg.model.d_model)
    logits.square().mean().backward()
    assert model.rnn.mem_tokens.grad.abs().sum() > 0
    assert get_parameter_count(model) == count
    assert isinstance(construct_rnn(cfg.model), RecurrentMemoryTransformer)


def test_rmt_torch_compile_preserves_outputs_and_gradients():
    torch.manual_seed(5)
    model = _model()
    compiled_source = copy.deepcopy(model)
    compiled = torch.compile(compiled_source, backend="aot_eager", fullgraph=True)
    inputs = torch.randn(7, 2, 4)
    expected, expected_state = model(inputs)
    actual, actual_state = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state[0], expected_state[0])
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    for expected_parameter, actual_parameter in zip(model.parameters(), compiled_source.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("num_heads", [1, 3])
def test_rmt_parameter_search_respects_even_width_with_odd_head_counts(num_heads):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="base", overrides=[
            "model=RMT", "task=Parity", "max_parameters=60000",
            "model.num_layers=2", "model.num_mem_tokens=4",
            f"model.num_heads={num_heads}", "device=-1", "training.compile=false",
        ])
    task = object.__new__(ParityTask)
    optimise_max_d_model(task, cfg)
    assert cfg.model.d_model % 2 == 0
    assert cfg.model.d_model % num_heads == 0
    with torch.device("meta"):
        assert get_parameter_count(task.build_model(cfg)) <= cfg.max_parameters
        cfg.model.d_model += 2 * num_heads
        assert get_parameter_count(task.build_model(cfg)) > cfg.max_parameters


def test_rmt_cell_matches_single_token_segment_sequence_model():
    torch.manual_seed(6)
    sequence = _model(segment_length=1).eval()
    recurrent = RecurrentModel(RMTCell(copy.deepcopy(sequence))).eval()
    inputs = torch.randn(5, 2, 4)
    expected, (memory,) = sequence(inputs)
    actual, state = recurrent(inputs, recurrent.get_initial_state(2))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state[0], memory)
    torch.testing.assert_close(state[-1], expected[-1])


def _patch_rmt():
    return construct_rnn(OmegaConf.create({
        "model": "RMT", "d_input": 4, "d_model": 8, "num_heads": 2,
        "num_layers": 2, "num_mem_tokens": 2, "segment_length": 1,
        "dropout": 0.0, "backbone": {"mode": "patches", "num_heads": 2},
    }))


def test_patch_rmt_is_causal_streams_and_backpropagates_through_memory():
    torch.manual_seed(7)
    model = _patch_rmt().eval()
    assert isinstance(model, CrossAttentionRecurrentModel)
    inputs = torch.randn(5, 2, 6, 4, requires_grad=True)
    outputs, state = model(inputs, model.get_initial_state(2))
    assert outputs.shape == (5, 2, 8)
    assert state[0].shape == (2, 2, 8)
    torch.testing.assert_close(state[-1], outputs[-1])
    with torch.no_grad():
        first, carried = model(inputs[:2], model.get_initial_state(2))
        rest, carried = model(inputs[2:], carried)
        torch.testing.assert_close(torch.cat((first, rest)), outputs)
        torch.testing.assert_close(carried[0], state[0])
        changed = inputs.detach().clone()
        changed[3:] += 10 * torch.randn_like(changed[3:])
        changed_outputs, _ = model(changed, model.get_initial_state(2))
        torch.testing.assert_close(changed_outputs[:3], outputs[:3])
    outputs[-1, :, 0].sum().backward()
    assert inputs.grad[0].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_patch_rmt_requires_one_tick_segments():
    with pytest.raises(ValueError, match="model.segment_length=1"):
        RMTCell(_model(segment_length=2))


def test_patch_rmt_compiles_without_graph_breaks():
    torch.manual_seed(8)
    model = _patch_rmt().eval()
    inputs = torch.randn(3, 2, 6, 4)
    model(inputs, model.get_initial_state(2))  # Materialize lazy cross-attention projections.
    compiled_source = copy.deepcopy(model)
    compiled = torch.compile(compiled_source, backend="aot_eager", fullgraph=True)
    expected, expected_state = model(inputs, model.get_initial_state(2))
    actual, actual_state = compiled(inputs, compiled_source.get_initial_state(2))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state[0], expected_state[0])
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    for p, q in zip(model.parameters(), compiled_source.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_imagenet_rmt_parameter_budget_includes_cross_attention_and_classifier():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="base", overrides=[
            "model=RMT", "task=ImageNet100", "max_parameters=1000000",
            "model.num_layers=2", "model.segment_length=1", "model.num_mem_tokens=32",
            "model.backbone.pretrained=false", "device=-1", "training.compile=false",
        ])
    task = object.__new__(ImageNet100Task)
    optimise_max_d_model(task, cfg)
    assert cfg.model.d_model == 152
    with torch.device("meta"):
        model = task.build_model(cfg)
        assert get_parameter_count(model) == 997876
        assert all(not p.requires_grad for p in model.backbone.parameters())
        logits, state = model(torch.empty(2, 3, 224, 224), model.get_initial_states(2))
        assert logits.shape == (2, 100, 25)
        assert state[0].shape == (2, 32, 152)
        cfg.model.d_model += cfg.model.num_heads
        assert get_parameter_count(task.build_model(cfg)) > cfg.max_parameters
