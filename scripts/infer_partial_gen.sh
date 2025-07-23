#!/bin/bash

GPU=2
# CONFIG_NAME="default_dit"
CONFIG_NAME="dit"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt" ## Set your checkpoint path here

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="partial_gen"
GIVEN_WAV_PATH="data_sample/sakanaction_music_seg.wav"
# GIVEN_WAV_PATH="data_sample/haruharu_seg.wav"
# GIVEN_WAV_PATH="data_sample/dontwannacry_seg.wav"
# GIVEN_WAV_PATH="data_sample/bruno_24kmagic_seg.wav"
# GIVEN_WAV_PATH="data_sample/iu_lilac_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie_attention_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie_wedont_seg.wav"
# GIVEN_WAV_PATH="data_sample/aot_seg.wav"

# TEXT_PROMPT="The sound of overdrive guitar"
# TEXT_PROMPT="Funky guitar strumming"
TEXT_PROMPT="Jazzy guitar improvisation"

## GEN / Inpaint Condition
NUM_STEPS=100
# CFG_SCALE=8.0
# CFG_SCALE=14.0
CFG_SCALE=5.0
OVERLAP_DUR=6.0
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
