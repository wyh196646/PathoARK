#!/usr/bin/env bash

python run.py \
  --experiment_type linprobe \
  --model_name titan \
  # --model_kwargs_yaml "configs/abmil_kwargs.yaml" \  # Additional kwargs for model initialization (required for ABMIL)
  --tasks_yaml configs/tasks.yaml \
  --combine_slides_per_patient False \
  --saveto ../artifacts/example_runs/titan_linprobe \
  --hyperparams_yaml "configs/linprobe/linprobe.yaml" \
  --pooled_dirs_yaml "configs/pooled_embeddings_paths.yaml" \
  --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
  --splits_root "../artifacts/splits" \
  --conda_venv pathobench \
  --delay_interval 5                    # Controls how much each pane is delayed by. Pane i will start after (i*delay_interval) seconds