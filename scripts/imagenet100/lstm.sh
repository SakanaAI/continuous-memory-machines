#!/usr/bin/env bash

# ImageNet-100: LSTM
uv run python main.py -m \
  model=LSTM \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  "$@"
