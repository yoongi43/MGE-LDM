#!/bin/bash

# EXPNAME=$1
# GPU=$2
# GPU=5
GPU="6,7"

## Load from mix-only pretrained model
# MIX_CKPT_DIR="/data2/yoongi/MGE_LDM/default_dit_mix_pretrain/checkpoints/"
# MIX_CKPT_PATH=$MIX_CKPT_DIR"unwrapped_last.ckpt"
# MIX_CKPT_PATH=null ## Can be null if you want to train from scratch

### Load from resume checkpoint
# CKPT_DIR="/data2/yoongi/MGE_LDM/default_dit/checkpoints/"
# CKPT_PATH=$CKPT_DIR"last.ckpt"

SAVE_DIR="/data2/yoongi/MGE_LDM"
CONFIG_NAME="dit"

AE_CKPT_PATH="/data2/yoongi/MGE_LDM/default_ae/checkpoints/unwrapped_last.ckpt"
# AE_CKPT_PATH=".../unwrapped_AE.ckpt"


# CUDA_VISIBLE_DEVICES=$GPU \
# taskset -c 16-79 \
# python train_dit.py \
# --config-name $CONFIG_NAME \
# save_dir=$SAVE_DIR \
# autoencoder_ckpt_path=$AE_CKPT_PATH
# # # pretrained_ckpt_path=$MIX_CKPT_PATH
# # # ckpt_path=$CKPT_PATH


## Mutlti-GPU Training
CUDA_VISIBLE_DEVICES=$GPU \
torchrun --nproc_per_node gpu train_dit.py \
--config-name $CONFIG_NAME \
save_dir=$SAVE_DIR \
autoencoder_ckpt_path=$AE_CKPT_PATH
