#!/usr/bin/env bash

# Large mazes: GDN
# Previous run failed; configuration retained for reference.
uv run python main.py -m \
  model=GDN \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.epochs=9000 \
  training.batch_size=8 \
  training.learning_rate=1e-4 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  "$@"
