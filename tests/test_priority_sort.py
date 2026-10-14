from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf

from tasks.recall.priority_sort import PrioritySortDataset, PrioritySortTask, VariableLengthBatchSampler


def test_priority_sort_dataset_accepts_length_from_sampler_index():
    dataset = PrioritySortDataset(value_width=8, num_input_items=20, num_output_items=16, num_samples=10)

    inputs, targets = dataset[(0, 24)]

    assert inputs.shape == (40, 11)
    assert targets.shape == (40, 8)
    assert inputs[:24, 9].eq(1).all()
    assert inputs[24:, 10].eq(1).all()


def test_variable_length_batch_sampler_is_homogeneous_balanced_and_deterministic():
    sampler = VariableLengthBatchSampler(
        dataset_size=18,
        batch_size=2,
        min_num_input_items=16,
        max_num_input_items=18,
        seed=7,
    )
    sampler.set_epoch(3)
    batches = list(sampler)

    lengths = []
    for batch in batches:
        batch_lengths = {index[1] for index in batch}
        assert len(batch_lengths) == 1
        lengths.append(batch[0][1])

    assert set(lengths) == {16, 17, 18}
    counts = Counter(lengths)
    assert max(counts.values()) - min(counts.values()) <= 1

    repeated = VariableLengthBatchSampler(18, 2, 16, 18, seed=7)
    repeated.set_epoch(3)
    assert list(repeated) == batches


def test_priority_sort_loss_uses_final_output_steps():
    task = object.__new__(PrioritySortTask)
    task.num_output_items = 2
    targets = torch.zeros(1, 5, 1)
    logits = torch.full_like(targets, 20.0)
    logits[:, -2:, :] = -20.0

    loss = task._compute_loss(logits, targets)
    metrics = task._compute_metrics(logits, targets, loss)

    assert loss.item() < 1e-6
    assert metrics["bit_accuracy"] == 1.0
    assert metrics["vector_accuracy"] == 1.0


def test_priority_sort_rejects_more_outputs_than_inputs():
    dataset = PrioritySortDataset(value_width=8, num_input_items=20, num_output_items=16, num_samples=10)

    with pytest.raises(ValueError, match="must not exceed"):
        dataset[(0, 15)]


def test_priority_sort_task_builds_variable_length_training_batches():
    cfg = OmegaConf.create(
        {
            "device": -1,
            "seed": 5,
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "use_scheduler": False,
                "max_grad_norm": 1.0,
                "compile": False,
            },
            "model": {
                "model": "RNN",
                "d_model": 8,
                "d_input": 4,
                "dropout": 0.0,
                "layer_norm": True,
                "backbone": {
                    "type": "linear",
                    "mode": "vector",
                    "pretrained": False,
                    "freeze": False,
                },
            },
            "task": {
                "value_width": 3,
                "num_input_items": 4,
                "train_min_num_input_items": 3,
                "train_max_num_input_items": 5,
                "num_output_items": 2,
                "num_samples": 6,
            },
        }
    )
    task = PrioritySortTask(cfg)

    sequence_lengths = {inputs.shape[1] for inputs, _ in task.train_dataloader}

    assert sequence_lengths == {5, 6, 7}
