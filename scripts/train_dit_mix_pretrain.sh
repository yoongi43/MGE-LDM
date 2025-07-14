#!/bin/bash

# EXPNAME=$1
# GPU=$2

## in configs/
# GPU="0,1"
# CKPT_DIR="/home/yoongi43/gen_extract/stable_latent/results/AE/fl741ymc/checkpoints/"
# CKPT_DIR="/data/yoongidata/genext/result/AE/stable_audio_2_genext_vae_training_single_dataset_lqis/checkpoints/"
# CKPT_DIR="/data2/yoongi/MGE_LDM/default_ae/checkpoints/"
# CKPT_PATH=$CKPT_DIR"last.ckpt"

## Rseume
CKPT_DIR="/data2/yoongi/MGE_LDM/default_dit_mix_pretrain/checkpoints/"
# CKPT_PATH=$CKPT_DIR"last.ckpt"
# CKPT_PATH=$CKPT_DIR"epoch=4-step=100000.ckpt"
CKPT_PATH="${CKPT_DIR}/epoch-4-step-100000.ckpt"

SAVE_DIR="/data2/yoongi/MGE_LDM"
GPU=7
CUDA_VISIBLE_DEVICES=$GPU \
taskset -c 32-63 \
python train_dit.py \
--config-name default_dit_mix_pretrain \
save_dir=$SAVE_DIR \
ckpt_path=$CKPT_PATH
# ckpt_path=$CKPT_PATH
# python train_ae.py