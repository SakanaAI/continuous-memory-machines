#!/usr/bin/env bash

# Few-shot linear regression (K=10): CMM
uv run python main.py -m \
  model=CMM \
  task=FewShotRegression \
  task.task_mode=linear \
  task.train_length=10 \
  max_parameters=500000 \
  device=0 \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  evaluation.evaluation_interval=500 \
  model.memory_length=10 \
  model.num_long_term_memory_tokens=32 \
  "$@"
