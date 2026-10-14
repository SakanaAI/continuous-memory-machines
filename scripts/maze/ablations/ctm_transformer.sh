#!/usr/bin/env bash

# Large mazes: CTM-Transformer ablation
uv run python main.py -m \
  model=CTMTransformer \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.batch_size=1024 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  max_parameters=10000000 \
  task.iterations=50 \
  model.memory_length=18 \
  model.memory_transformer_layers=1 \
  model.num_heads=1 \
  model.synapse_type=unet \
  model.synapse_depth=16 \
  model.memory_update_gate=true \
  model.memory_update_gate_init=0.0 \
  training.epochs=9000 \
  "$@"
