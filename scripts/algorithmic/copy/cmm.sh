#!/usr/bin/env bash

# Copy: CMM
uv run python main.py -m \
  model=CMM \
  task=Copy \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  task.copy_delay=0 \
  task.copy_min_sequence_length=100 \
  task.copy_max_sequence_length=100 \
  task.copy_dim=10 \
  model.memory_length=10 \
  model.num_long_term_memory_tokens=128 \
  "$@"
