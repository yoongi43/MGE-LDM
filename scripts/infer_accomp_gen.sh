#!/bin/bash

GPU=4
# CONFIG_NAME="default_dit"
CONFIG_NAME="default_dit_scratch"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="accomp_gen"
# GEN_AUDIO_DUR=30.0
GIVEN_WAV_PATH="/home/yoongi43/gen_extract/MGE-LDM/data_sample/musdb_AlJames/vocals.wav"
# TEXT_PROMPT="The piano accompaniment without vocals"
TEXT_PROMPT="Accompaniment containing piano, drums, and bass, and without vocals"

## GEN / Inpaint Condition
NUM_STEPS=20
CFG_SCALE=2.0
OVERLAP_DUR=3.0
REPAINT_N=2
 


CUDA_VISIBLE_DEVICES=$GPU \
python infer.py \
    --config-name $CONFIG_NAME \
    +task=$TASK \
    ckpt_path=${CKPT_PATH} \
    +given_wav_path=${GIVEN_WAV_PATH} \
    "+text_prompt='${TEXT_PROMPT}'" \
    +num_steps=${NUM_STEPS} \
    +cfg_scale=${CFG_SCALE} \
    +overlap_dur=${OVERLAP_DUR} \
    +repaint_n=${REPAINT_N} \
    +output_dir=${OUTPUT_DIR}
