#!/bin/bash

GPU=4
# CONFIG_NAME="default_dit"
CONFIG_NAME="default_dit_scratch"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="partial_gen"
# GEN_AUDIO_DUR=30.0
# GIVEN_WAV_PATH="/home/yoongi43/gen_extract/MGE-LDM/data_sample/sample1/mixture.wav"
# GIVEN_WAV_PATH="data_sample/sakanaction_music/sakanaction_music_seg.wav"
GIVEN_WAV_PATH="data_sample/bruno_24kmagic/bruno_24kmagic_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie/charlie_attention_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie/charlie_wedont_seg.wav"


# TEXT_PROMPT="Jazz piano improvisation"
TEXT_PROMPT="The sound of the distorted guitar"
# TEXT_PROMPT="Guitar solo"
# TEXT_PROMPT="The violin track"
# TEXT_PROMPT="The sound of string instruments"
# TEXT_PROMPT="Dynamic EDM synth melody"

## GEN / Inpaint Condition
NUM_STEPS=100
CFG_SCALE=2.0
OVERLAP_DUR=5.0
REPAINT_N=1
 


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
