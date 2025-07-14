#!/bin/bash

# EXPNAME=$1
# GPU=$2

## in configs/
# GPU="0,1"

## For resuming
# CKPT_DIR="/data2/yoongi/MGE_LDM/default_ae/checkpoints/"
# CKPT_PATH=$CKPT_DIR"last.ckpt"

SAVE_DIR="/data2/yoongi/MGE_LDM"
CONFIG_NAME="default_ae_temp"
# CONFIG_NAME="default_ae"
GPU=0
CUDA_VISIBLE_DEVICES=$GPU \
taskset -c 64-79 \
python train_ae.py \
--config-name $CONFIG_NAME \
save_dir=$SAVE_DIR \
ckpt_path=$CKPT_PATH
# python train_ae.py