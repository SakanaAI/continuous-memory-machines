#!/usr/bin/env bash

# Large mazes: RMT
# Previous run failed; configuration retained for reference.
uv run python main.py -m \
  model=RMT \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.batch_size=64 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  max_parameters=10000000 \
  model.num_layers=2 \
  model.segment_length=1 \
  model.num_mem_tokens=8 \
  training.epochs=9000 \
  training.accumulation_steps=16 \
  "$@"
