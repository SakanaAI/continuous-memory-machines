#!/usr/bin/env bash

# AssociativeRecall: CTM-Transformer ablation
uv run python main.py -m \
  model=CTMTransformer \
  task=AssociativeRecall \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  model.memory_length=42 \
  model.memory_transformer_layers=1 \
  model.num_heads=4 \
  training.batch_size=32 \
  "$@"
