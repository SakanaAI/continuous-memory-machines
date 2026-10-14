#!/usr/bin/env bash

# ImageNet-100: Transformer
uv run python main.py -m \
  model=Transformer \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.causal=true \
  model.max_seq_len=25 \
  "$@"
