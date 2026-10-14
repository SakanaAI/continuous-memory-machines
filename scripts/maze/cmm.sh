#!/usr/bin/env bash

# Large mazes: CMM
uv run python main.py -m \
  model=CMM \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.epochs=9000 \
  training.batch_size=1024 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  model.memory_length=10 \
  model.num_long_term_memory_tokens=8 \
  model.num_heads=1 \
  model.memory_update_gate=true \
  model.memory_update_gate_init=0.0 \
  "$@"
