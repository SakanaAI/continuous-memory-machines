#!/usr/bin/env bash

# AssociativeRecall: RMT
uv run python main.py -m \
  model=RMT \
  task=AssociativeRecall \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.num_layers=2 \
  model.segment_length=10 \
  model.num_mem_tokens=32 \
  "$@"
