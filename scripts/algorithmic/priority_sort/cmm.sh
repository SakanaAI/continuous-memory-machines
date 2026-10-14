#!/usr/bin/env bash

# PrioritySort: CMM
uv run python main.py -m \
  model=CMM \
  task=PrioritySort \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.memory_length=20 \
  model.num_long_term_memory_tokens=128 \
  model.memory_transformer_layers=2 \
  "$@"
