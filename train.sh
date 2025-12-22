#!/bin/bash

GPU=0
LR=1e-5


### Low setting ###
NUM_EPOCHS=500
EXPERIMENT="main_low_mclip"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 20 \
    --datause 0.01 \
    --data_korean 'true' 

EXPERIMENT="main_low_tomclip_dm"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 20 \
    --dm_loss 'yes' \
    --alpha 0.01 \
    --datause 0.01 \
    --data_korean 'true' 

EXPERIMENT="main_low_tomclip_ta"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 20 \
    --dm_loss 'yes' \
    --alpha 0.0 \
    --tda_loss "swd" \
    --beta 0.01 \
    --gamma 0.0 \
    --datause 0.01 \
    --data_korean 'true' 

EXPERIMENT="main_low_tomclip"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 20 \
    --dm_loss 'yes' \
    --alpha 0.01 \
    --tda_loss "swd" \
    --beta 0.01 \
    --gamma 0.0 \
    --datause 0.01 \
    --data_korean 'true' 


### Full setting ###
NUM_EPOCHS=100
EXPERIMENT="main_mclip"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 5 \
    --data_korean 'true' 

EXPERIMENT="main_tomclip_dm"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 5 \
    --dm_loss 'yes' \
    --alpha 0.01 \
    --data_korean 'true' 

EXPERIMENT="main_tomclip_ta"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 5 \
    --dm_loss 'yes' \
    --alpha 0.0 \
    --tda_loss "swd" \
    --beta 0.01 \
    --gamma 0.0 \
    --data_korean 'true' 

EXPERIMENT="main_tomclip"
python pt_Training_with_tdaloss.py \
    --gpu "$GPU" \
    --experiment "$EXPERIMENT" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --patience 5 \
    --dm_loss 'yes' \
    --alpha 0.01 \
    --tda_loss "swd" \
    --beta 0.01 \
    --gamma 0.0 \
    --data_korean 'true' 



