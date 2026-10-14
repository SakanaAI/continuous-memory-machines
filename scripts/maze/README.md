# Large mazes

The maze dataset downloads automatically on first run. Run a model's script from the repository root, for example `bash scripts/maze/cmm.sh`. The CMM-LSTM and CMM-STM ablations are in `ablations/`.

All maze scripts train for 9,000 epochs, matching the default in [Maze.yaml](../../conf/task/Maze.yaml).

The previous LSAM, GDN, and RMT runs failed. Their configurations are retained for reference and have not been validated by a successful training run.
