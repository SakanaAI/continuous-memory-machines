#!/usr/bin/env bash

# PrioritySort: CTM
uv run python main.py -m \
  model=CTM \
  task=PrioritySort \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.memory_length=20 \
  "$@"
