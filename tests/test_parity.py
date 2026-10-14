import torch
from omegaconf import OmegaConf

from models.transformer import TransformerSequenceModel
from tasks.recall.parity import ParityDataset, ParityTask


def _task_config():
    return OmegaConf.create(
        {
            "device": -1,
            "seed": 3,
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
                "train_min_seq_length": 2,
                "train_max_seq_length": 4,
                "test_seq_lengths": [4, 7],
                "num_samples": 6,
                "eval_num_samples": 4,
            },
        }
    )


def test_parity_dataset_target_and_query_token():
    inputs, target = ParityDataset(seq_length=5, num_samples=1)[0]

    assert inputs.shape == (6, 2)
    assert inputs[:-1, 1].eq(0).all()
    assert inputs[-1].equal(torch.tensor([0.0, 1.0]))
    assert target.item() == inputs[:-1, 0].sum().remainder(2).item()


def test_parity_dataset_accepts_variable_length_sampler_index():
    inputs, _ = ParityDataset(seq_length=5, num_samples=1)[(0, 9)]
    assert inputs.shape == (10, 2)


def test_parity_task_uses_homogeneous_variable_length_batches():
    task = ParityTask(_task_config())
    observed_lengths = {inputs.shape[1] - 1 for inputs, _ in task.train_dataloader}
    assert observed_lengths == {2, 3, 4}


def test_parity_loss_and_accuracy_use_final_query_step():
    task = object.__new__(ParityTask)
    targets = torch.tensor([0.0, 1.0])
    logits = torch.zeros(2, 4, 1)
    logits[:, -1, 0] = torch.tensor([-20.0, 20.0])

    loss = task._compute_loss(logits, targets)
    metrics = task._compute_metrics(logits, targets, loss)

    assert loss.item() < 1e-6
    assert metrics["accuracy"] == 1.0


def test_causal_transformer_hides_future_tokens():
    torch.manual_seed(0)
    model = TransformerSequenceModel(
        d_input=4,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        max_seq_len=8,
        causal=True,
    ).eval()
    first = torch.randn(5, 1, 4)
    second = first.clone()
    second[3:] = torch.randn_like(second[3:])

    with torch.inference_mode():
        first_outputs, _ = model(first)
        second_outputs, _ = model(second)

    torch.testing.assert_close(first_outputs[:3], second_outputs[:3])
