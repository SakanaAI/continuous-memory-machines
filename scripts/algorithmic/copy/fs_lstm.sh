#!/usr/bin/env bash

# Copy: FSRNN
uv run python main.py -m \
  model=FSRNN \
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
  model.fast_layers=2 \
  "$@"
