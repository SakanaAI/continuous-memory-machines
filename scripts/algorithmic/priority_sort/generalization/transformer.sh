#!/usr/bin/env bash

# Priority Sort length generalization: Transformer
uv run python main.py -m \
  model=Transformer \
  task=PrioritySort \
  device=-1 \
  max_parameters=60000 \
  model.dropout=0.1 \
  task.value_width=4 \
  task.num_output_items=4 \
  task.train_min_num_input_items=4 \
  task.train_max_num_input_items=16 \
  task.num_input_items=20 \
  task.num_samples=832 \
  training.batch_size=64 \
  training.epochs=3000 \
  training.learning_rate=1e-3 \
  training.weight_decay=1e-4 \
  training.compile=false \
  evaluation.evaluation_interval=50 \
  logging.wandb.enabled=false \
  checkpoint.enabled=true \
  checkpoint.checkpoint_every=500 \
  seed=0,1,2 \
  model.causal=true \
  model.num_layers=2 \
  model.nhead=4 \
  model.dim_feedforward=128 \
  model.max_seq_len=2048 \
  model.learn_pos_emb=false \
  "$@"
