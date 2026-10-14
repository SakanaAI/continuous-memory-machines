#!/usr/bin/env bash

# Few-shot linear regression (K=10): LSTM_CMM ablation
uv run python main.py -m \
  model=LSTM_CMM \
  task=FewShotRegression \
  task.task_mode=linear \
  task.train_length=10 \
  max_parameters=500000 \
  device=0 \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  evaluation.evaluation_interval=500 \
  model.num_long_term_memory_tokens=32 \
  "$@"
