#!/usr/bin/env bash

# Large mazes: NAMTM
uv run python main.py -m \
  model=NAMTM \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.batch_size=256 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  max_parameters=10000000 \
  training.accumulation_steps=4 \
  training.epochs=9000 \
  "$@"
