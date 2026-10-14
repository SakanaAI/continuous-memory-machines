#!/usr/bin/env bash

# Large mazes: FSRNN
uv run python main.py -m \
  model=FSRNN \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.epochs=9000 \
  training.batch_size=1024 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  model.fast_layers=2 \
  "$@"
