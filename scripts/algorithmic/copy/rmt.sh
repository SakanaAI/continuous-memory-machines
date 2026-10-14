#!/usr/bin/env bash

# Copy: RMT
uv run python main.py -m \
  model=RMT \
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
  model.num_layers=2 \
  model.segment_length=10 \
  model.num_mem_tokens=128 \
  "$@"
