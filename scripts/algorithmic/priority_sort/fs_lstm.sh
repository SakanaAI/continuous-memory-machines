#!/usr/bin/env bash

# PrioritySort: FSRNN
uv run python main.py -m \
  model=FSRNN \
  task=PrioritySort \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.fast_layers=2 \
  "$@"
