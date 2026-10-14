#!/usr/bin/env bash

# AssociativeRecall: CMM
uv run python main.py -m \
  model=CMM \
  task=AssociativeRecall \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.memory_length=10 \
  model.num_long_term_memory_tokens=32 \
  "$@"
