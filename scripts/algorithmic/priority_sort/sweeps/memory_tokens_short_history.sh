#!/usr/bin/env bash

# Priority Sort: sweep long-term memory tokens with a shorter history
uv run python main.py -m \
  model=CMM \
  task=PrioritySort \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1 \
  training.batch_size=32 \
  model.memory_length=5 \
  model.num_long_term_memory_tokens=2,4,8,16,32,64,128 \
  model.memory_transformer_layers=2 \
  "$@"
