#!/usr/bin/env bash

# Few-shot linear regression (K=10): RMT
uv run python main.py -m \
  model=RMT \
  task=FewShotRegression \
  task.task_mode=linear \
  task.train_length=10 \
  max_parameters=500000 \
  device=0 \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  evaluation.evaluation_interval=500 \
  model.num_layers=2 \
  model.num_heads=4 \
  model.segment_length=10 \
  model.num_mem_tokens=32 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  "$@"
