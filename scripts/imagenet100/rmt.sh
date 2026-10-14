#!/usr/bin/env bash

# ImageNet-100: RMT
uv run python main.py -m \
  model=RMT \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.num_layers=2 \
  model.segment_length=1 \
  model.num_mem_tokens=32 \
  "$@"
