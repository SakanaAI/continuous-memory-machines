
# Continuous Memory Machines

The Continuous Memory Machine (CMM) is a recurrent architecture with separate matrix-valued short- and long-term memories. Neuron-level models compute over recent activity, while a shared Transformer updates both memories and lets them exchange information. This supports computation and persistent storage across algorithmic, in-context learning, and recurrent reasoning tasks.

## Experiments

Install dependencies and run commands from the repository root:

```bash
uv sync
```

The scripts reproduce the paper experiments, including the appendix. Each script contains one experiment configuration and runs its specified seeds sequentially on the local machine. Choose a model's script from the experiment folder; ablations are in its `ablations/` subfolder.

| Experiment | Example command | All configurations |
|---|---|---|
| Copy (length 100) | `bash scripts/algorithmic/copy/cmm.sh` | [Copy](scripts/algorithmic/copy) |
| Priority Sort | `bash scripts/algorithmic/priority_sort/cmm.sh` | [Priority Sort](scripts/algorithmic/priority_sort) |
| Associative Recall | `bash scripts/algorithmic/associative_recall/cmm.sh` | [Associative Recall](scripts/algorithmic/associative_recall) |
| Parity | `bash scripts/algorithmic/parity/cmm.sh` | [Parity](scripts/algorithmic/parity) |
| Linear regression (K=20) | `bash scripts/few_shot_regression/linear/k20/cmm.sh` | [Linear K=20](scripts/few_shot_regression/linear/k20) |
| Linear regression (K=10) | `bash scripts/few_shot_regression/linear/k10/cmm.sh` | [Linear K=10](scripts/few_shot_regression/linear/k10) |
| ImageNet-100 | `bash scripts/imagenet100/cmm.sh` | [ImageNet-100](scripts/imagenet100) |
| Large mazes | `bash scripts/maze/cmm.sh` | [Mazes](scripts/maze) |

Appendix experiments: [Priority Sort memory/depth sweeps](scripts/algorithmic/priority_sort/sweeps) and [Priority Sort length generalization](scripts/algorithmic/priority_sort/generalization).

Before running ImageNet-100, follow the [ViT-5 checkpoint setup](scripts/imagenet100/README.md). See the [maze notes](scripts/maze/README.md) for training settings and configurations with failed prior runs.

Scripts accept Hydra overrides at the end of the command. For example, to run a single seed without Weights & Biases logging:

```bash
bash scripts/algorithmic/copy/cmm.sh seed=0 logging.wandb.enabled=false
```

GPU experiments default to device 0; use `device=-1` for CPU. For SLURM, append `hydra/launcher=submitit_slurm hydra.launcher.partition=YOUR_PARTITION` and any resource overrides required by your cluster.

## Analysis

See [analysis/README.md](analysis/README.md) for the paper figures, tables, checkpoint evaluations, and compute benchmark. Experiment manifests identify the recorded runs and can be changed to point at new runs. Training outputs, datasets, and checkpoints are not included in the repository.

## Tests

Run the tests that do not need external datasets or pretrained weights:

```bash
uv run python -m pytest tests/ --ignore=tests/test_vit_checkpoint.py
```

See [tests/README.md](tests/README.md) for the ViT integration tests.
