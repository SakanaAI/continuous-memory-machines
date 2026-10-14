#!/usr/bin/env bash

# Large mazes: Transformer
uv run python main.py -m \
  model=Transformer \
  task=Maze \
  task.dataset=SakanaAI/mazes-large \
  training.batch_size=1024 \
  training.learning_rate=1e-3 \
  training.compile=false \
  device=0 \
  seed=0,1,2 \
  training.num_workers=7 \
  max_parameters=10000000 \
  model.causal=true \
  model.max_seq_len=50 \
  training.epochs=9000 \
  "$@"
