# Paper analysis

Run commands from the repository root with `uv run python -m ...`. These modules cover the experiments in the main paper and appendix. Importing a module does not run an analysis.

## Inputs and outputs

The YAML files in [configs/](configs/) identify the original Hydra runs. Training outputs and checkpoints are not distributed with the source code. To analyze new runs, edit a copy of the relevant manifest and pass `--config path/to/runs.yaml`. To read the recorded runs from another checkout or archive, pass `--runs-root /path/containing/outputs`. Use `--output-dir` to choose a destination; defaults are under `outputs/analysis/`.

| Paper result | Module | Default output |
|---|---|---|
| Algorithmic losses and linear regression tables | `analysis.results` | `results/summary.csv`, `results/per_seed.csv`, `results/regression_table.csv` |
| Four-task algorithmic figure | `analysis.algorithmic` | `algorithmic/combined_bar.pdf` |
| Priority Sort length generalization | `analysis.priority_sort.generalization` | `priority_sort/generalization/extrapolation_bit_accuracy_with_dnc.pdf` |
| Priority Sort capacity and depth ablations | `analysis.priority_sort.ablations` | `priority_sort/ablations/priority_sort_*.pdf` |
| Maze final results and learning curves | `analysis.maze.results` | `maze/summary.csv`, `maze/per_seed.csv` |
| ImageNet-100 top-1/top-5 table | `analysis.imagenet.evaluate` | `imagenet/top1_top5_summary.csv` |
| Maze attention example | `analysis.maze.attention` | PNG/PDF attention trajectories |
| ImageNet attention examples | `analysis.imagenet.attention` | PNG/PDF trajectories and an image gallery |
| Memory-attention figure | `analysis.attention.plot` | `attention/attention_analysis.pdf` |
| Full-sequence compute table | `analysis.compute.profile` | `compute/profile_copy100_full_sequence.csv` |

## Results from training logs

```bash
uv run python -m analysis.results
uv run python -m analysis.algorithmic
uv run python -m analysis.priority_sort.ablations
uv run python -m analysis.maze.results
```

The shared result collector reads the final epoch, exports the underlying training seed and source run, and reports arithmetic means and sample standard deviations (`ddof=1`). Missing or incomplete runs produce warnings; duplicate training seeds are rejected. Inspect `n_seeds` before reporting a result. Standard deviation is reported as zero for a single observation; it is not an uncertainty estimate.

Copy uses length 100; the combined figure uses Parity length 40. The result CSV also retains the other Parity evaluation lengths described in the appendix. Copy, Recall, Sort, and regression use epoch 1,000; Parity uses epoch 200. Regression uses linear target functions with 10 or 20 support examples, and reports logged query accuracy as a percentage. Missing query accuracy is never inferred from MSE.

The recorded paper results use `GDNStateTracking` for algorithmic tasks, ImageNet, and compute, but `GDN` for regression. The manifest preserves that distinction; result CSVs retain `source_model` and display both as GDN. Model identifiers `FSRNN`, `LSTM_CMM`, `CTM_Matched`, and `CTMTransformer` correspond to FS-LSTM, CMM-LSTM, CTM-Ext., and CMM-STM.

The ablations retain the two appendix sweeps: LTM capacity with STM length 5 and Transformer depth with the standard Sort memory sizes. Error bars use sample standard deviations across two seeds.

Maze tables use final available **current-model** accuracy, rather than EMA or each run's peak accuracy. The manifest records RMC seed 2's final epoch of 5,075; the other reported seeds end at 9,000. Full-precision metrics take precedence over rounded continuation logs. RMT, LSAM, and GDN have no completed Maze result in the paper.

## Checkpoint evaluations

```bash
uv run python -m analysis.priority_sort.generalization --device 0
uv run python -m analysis.imagenet.evaluate --device 0
```

Priority Sort evaluates the final checkpoints on 832 examples at each length `4, 8, 12, 16, 20, 30, 40, 50`, sharing evaluation examples across models and training seeds. `--plot-only` reuses an existing summary; `--models`, `--lengths`, and `--reuse-existing` support partial reevaluation. The recorded RMT generalization runs use training seeds 0, 1, and 3.

ImageNet evaluates all test images using the final iteration of epoch-25 checkpoints. `--models` reevaluates selected architectures while retaining cached rows for the others. Use `--device -1` for CPU. Dataset access or an existing Hugging Face cache is required.

## Attention

Export the attention data, then draw the combined figure:

```bash
uv run python -m analysis.attention.algorithmic --device 0
uv run python -m analysis.attention.visual --device 0
uv run python -m analysis.attention.plot
```

Algorithmic attention uses Copy training seed 1 and Sort training seed 0, with one test example from each, the first memory layer, and the mean over heads. Visual attention uses intact models on all 5,000 test images for each of three training seeds, retaining layers, heads, and iterations before aggregation. Batches are weighted by their example counts. The plot accepts `--algorithmic-csv` and `--visual-dir` for existing exports and requires complete visual evaluations. Hooks record attention without changing model outputs.

For individual visual examples:

```bash
uv run python -m analysis.maze.attention path/to/maze_checkpoint.pt --seed 110 --device 0
uv run python -m analysis.imagenet.attention path/to/imagenet_checkpoint.pt --seed 144 --head 3 --device 0
```

The ImageNet renderer supports `--num-images`, `--indices`, and `--all-heads`. Both renderers accept `--output-dir`. They produce source images and trajectories; the paper's assembled/cropped panels are separate presentation assets.

## Compute

```bash
uv run python -m analysis.compute.profile
```

Run on an otherwise idle CUDA GPU. Each architecture runs in a fresh process on complete Copy-100 sequences: 100 input plus 100 output steps, batch size 64, eager float32, five warmups, and twenty measurements. Counts include input/output projections. Forward timing disables autograd; forward/backward timing excludes optimizer updates. Per-step times divide full-sequence latency by 200; peak memory covers the complete forward/backward pass. JSON metadata records the hardware and protocol.
