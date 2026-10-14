#!/usr/bin/env bash

# AssociativeRecall: GDNStateTracking
uv run python main.py -m \
  model=GDNStateTracking \
  task=AssociativeRecall \
  max_parameters=500000 \
  device=0 \
  training.epochs=1000 \
  training.learning_rate=1e-4 \
  seed=0,1,2 \
  "$@"
