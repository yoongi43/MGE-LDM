#!/bin/bash

# GPU=$1
GPU=7
CKPT_DIR="/data2/yoongi/MGE_LDM/default_ae/checkpoints/"
CKPT_PATH=$CKPT_DIR"last.ckpt"

# OUTPUT_PATH=$CKPT_DIR"unwrapped_last"
OUTPUT_PATH=$CKPT_DIR"unwrapped_AE"

CUDA_VISIBLE_DEVICES=$GPU \
python unwrap_model.py \
    --config-name default_ae \
    +type=autoencoder \
    ckpt_path=${CKPT_PATH} \
    +use_safetensors=false \
    +output_name=${OUTPUT_PATH}