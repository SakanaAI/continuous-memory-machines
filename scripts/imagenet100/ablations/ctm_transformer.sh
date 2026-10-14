#!/usr/bin/env bash

# ImageNet-100: CTM-Transformer ablation
uv run python main.py -m \
  model=CTMTransformer \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.memory_length=42 \
  model.memory_transformer_layers=1 \
  model.num_heads=4 \
  model.memory_update_gate=false \
  "$@"
