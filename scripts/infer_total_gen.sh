#!/bin/bash

GPU=1
# CONFIG_NAME="default_dit"
# CONFIG_NAME="default_dit_mix_pretrain"
CONFIG_NAME="default_dit_scratch"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="total_gen"
GEN_AUDIO_DUR=30.0
GIVEN_WAV_PATH=null


#### Set text prompt here
TEXT_PROMPT="Funky upbeat jazz with guitar, saxophone and piano"
# TEXT_PROMPT="Lo-fi hip hop beat with mellow jazzy chords and a smooth bassline"
# TEXT_PROMPT="Relaxing acoustic guitar instrumental with soft percussion"
# TEXT_PROMPT="Metal guitar riff with heavy distortion and fast-paced drums"
# TEXT_PROMPT="Upbeat electronic dance music with catchy synth melodies and driving bass"


## Generation Configuration
NUM_STEPS=50
CFG_SCALE=3.0
OVERLAP_DUR=5.0
REPAINT_N=8 ## Multiple repainting steps for better quality



CUDA_VISIBLE_DEVICES=$GPU \
python infer.py \
    --config-name $CONFIG_NAME \
    +task=$TASK \
    ckpt_path=${CKPT_PATH} \
    +gen_audio_dur=${GEN_AUDIO_DUR} \
    +given_wav_path=${GIVEN_WAV_PATH} \
    "+text_prompt='${TEXT_PROMPT}'" \
    +num_steps=${NUM_STEPS} \
    +cfg_scale=${CFG_SCALE} \
    +overlap_dur=${OVERLAP_DUR} \
    +repaint_n=${REPAINT_N} \
    +output_dir=${OUTPUT_DIR}