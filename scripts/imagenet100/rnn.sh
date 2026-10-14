#!/usr/bin/env bash

# ImageNet-100: RNN
uv run python main.py -m \
  model=RNN \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  "$@"
