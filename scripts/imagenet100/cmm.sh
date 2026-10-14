#!/usr/bin/env bash

# ImageNet-100: CMM
uv run python main.py -m \
  model=CMM \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.memory_length=10 \
  model.num_long_term_memory_tokens=32 \
  "$@"
