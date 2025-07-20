#!/bin/bash

GPU=2
# CONFIG_NAME="default_dit_mix_pretrain"
# CONFIG_NAME="default_dit"
# CONFIG_NAME="default_dit_scratch"
CONFIG_NAME="dit"

CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"last.ckpt"

# OUTPUT_PATH=$CKPT_DIR"unwrapped_last"
# OUTPUT_PATH=$CKPT_DIR"unwrapped_DiT"
OUTPUT_PATH=$CKPT_DIR"unwrapped_DiT_31"

CUDA_VISIBLE_DEVICES=$GPU \
python unwrap_model.py \
    --config-name $CONFIG_NAME \
    +type=mgeldm \
    ckpt_path=${CKPT_PATH} \
    +use_safetensors=false \
    +output_name=${OUTPUT_PATH}
