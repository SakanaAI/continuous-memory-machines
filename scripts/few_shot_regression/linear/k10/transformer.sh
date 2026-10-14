#!/usr/bin/env bash

# Few-shot linear regression (K=10): Transformer
uv run python main.py -m \
  model=Transformer \
  task=FewShotRegression \
  task.task_mode=linear \
  task.train_length=10 \
  max_parameters=500000 \
  device=0 \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  evaluation.evaluation_interval=500 \
  model.causal=true \
  model.num_layers=4 \
  model.nhead=4 \
  model.dim_feedforward=384 \
  model.max_seq_len=64 \
  model.dropout=0.0 \
  training.learning_rate=1e-4 \
  training.weight_decay=0.0 \
  training.compile=false \
  logging.wandb.enabled=false \
  "$@"
