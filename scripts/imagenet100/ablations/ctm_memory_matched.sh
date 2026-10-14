#!/usr/bin/env bash

# ImageNet-100: CTM-Ext (memory-capacity matched CTM)
uv run python main.py -m \
  model=CTM \
  task=ImageNet100 \
  max_parameters=1000000 \
  device=0 \
  seed=0,1,2 \
  model.memory_length=42 \
  training.epochs=25 \
  training.batch_size=64 \
  training.learning_rate=1e-4 \
  "$@"
