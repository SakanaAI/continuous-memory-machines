#!/usr/bin/env bash

# ImageNet-100: RMC
uv run python main.py -m \
  model=RMC \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.mem_slots=32 \
  "$@"
