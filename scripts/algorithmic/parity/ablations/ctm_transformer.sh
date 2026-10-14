#!/usr/bin/env bash

# Parity: CTM-Transformer ablation
uv run python main.py -m \
  model=CTMTransformer \
  task=Parity \
  device=0 \
  max_parameters=60000 \
  model.dropout=0.0 \
  task.train_min_seq_length=1 \
  task.train_max_seq_length=40 \
  'task.test_seq_lengths=[40,50,100,200,500,1000]' \
  task.num_samples=8192 \
  task.eval_num_samples=512 \
  training.batch_size=64 \
  training.epochs=200 \
  training.learning_rate=1e-4 \
  training.weight_decay=0.0 \
  training.compile=false \
  evaluation.evaluation_interval=25 \
  logging.wandb.enabled=false \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=50 \
  seed=0,1,2 \
  model.memory_length=8 \
  model.memory_transformer_layers=1 \
  model.num_heads=4 \
  model.memory_update_gate=true \
  model.memory_update_gate_init=0.0 \
  "$@"
