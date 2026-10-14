#!/usr/bin/env bash

# Copy: Transformer
uv run python main.py -m \
  model=Transformer \
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
  model.causal=true \
  model.num_layers=4 \
  model.nhead=4 \
  model.dim_feedforward=384 \
  model.max_seq_len=256 \
  model.dropout=0.0 \
  training.weight_decay=0.0 \
  "$@"
