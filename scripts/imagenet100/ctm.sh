#!/usr/bin/env bash

# ImageNet-100: CTM
uv run python main.py -m \
  model=CTM \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.memory_length=10 \
  "$@"
