#!/bin/bash

GPU=0

python cifar100_eval.py --gpu $GPU --mclip_model_path /experiment/Low/main_data1.0_koTrue_20250730_1e-05/best_model.pt --tomclip_model_path /experiment/Low/main_data1.0_koTrue_20250730_1e-05_dm_loss_0.01_swd_1_0.01_0.0/best_model.pt
python cifar100_eval.py --gpu $GPU --mclip_model_path /experiment/Full/main_run_1_data0.01_koTrue_20250730_1e-05/best_model.pt --tomclip_model_path /experiment/Full/main_run_1_data0.01_koTrue_20250730_1e-05_dm_loss_0.01_swd_1_0.01_0.0/best_model.pt

python evaluate_xflickrco.py \
  --gpu 0 \
  --clip_model_name ViT-B/32 \
  --mclip_model_path /experiment/Low/main_data1.0_koTrue_20250730_1e-05/best_model.pt \
  --tomclip_model_path /experiment/Low/main_data1.0_koTrue_20250730_1e-05_dm_loss_0.01_swd_1_0.01_0.0/best_model.pt \
  --split test \
  --bootstrap 10000 \
  --out /experiment/Low/xflickrco_results_full.csv

python evaluate_xflickrco.py \
  --gpu 0 \
  --clip_model_name ViT-B/32 \
  --mclip_model_path /experiment/Full/main_run_1_data0.01_koTrue_20250730_1e-05/best_model.pt \
  --tomclip_model_path /experiment/Full/main_run_1_data0.01_koTrue_20250730_1e-05_dm_loss_0.01_swd_1_0.01_0.0/best_model.pt \
  --split test \
  --bootstrap 10000 \
  --out /experiment/Full/xflickrco_results_low.csv
