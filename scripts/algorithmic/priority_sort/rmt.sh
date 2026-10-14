#!/usr/bin/env bash

# PrioritySort: RMT
uv run python main.py -m \
  model=RMT \
  task=PrioritySort \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.num_layers=2 \
  model.segment_length=20 \
  model.num_mem_tokens=128 \
  "$@"
