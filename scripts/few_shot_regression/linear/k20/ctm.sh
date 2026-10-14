#!/usr/bin/env bash

# Few-shot linear regression (K=20): CTM
uv run python main.py -m \
  model=CTM \
  task=FewShotRegression \
  task.task_mode=linear \
  task.train_length=20 \
  max_parameters=500000 \
  device=0 \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  evaluation.evaluation_interval=500 \
  model.memory_length=10 \
  "$@"
