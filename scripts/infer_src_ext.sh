#!/bin/bash

GPU=0
# CONFIG_NAME="default_dit"
CONFIG_NAME="default_dit_scratch"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="source_extract"

GIVEN_WAV_PATH="data_sample/sakanaction_music/sakanaction_music_seg.wav"
# GIVEN_WAV_PATH="data_sample/bruno_24kmagic/bruno_24kmagic_seg.wav"
# GIVEN_WAV_PATH="data_sample/youseebiggirl/youseebiggirl_seg.wav"
# GIVEN_WAV_PATH="data_sample/vaundy_kaiju/vaundy_kaiju_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie/charlie_attention_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie/charlie_wedont_seg.wav"

# TEXT_PROMPT="The sound of vocals"
TEXT_PROMPT="The vocals stem"
# TEXT_PROMPT="The sound of the bass"
# TEXT_PROMPT="The sound of drums"
# TEXT_PROMPT="The sound of the synthesizer"
# TEXT_PROMPT="The synthesizer sounds like a siren"
# TEXT_PROMPT="The sound of the ocilating synthesizer, like a siren"
# TEXT_PROMPT="The ocillating synthesizer, sounds like a siren"
# TEXT_PROMPT="The acoustic guitar sound"

# TEXT_PROMPT="The sound of the distorted guitar"
# TEXT_PROMPT="The sound of the chorus vocals"
# TEXT_PROMPT="The percussive sound"
# TEXT_PROMPT="Vocals track"
# TEXT_PROMPT="Drums track"
# TEXT_PROMPT="Synth track"
# TEXT_PROMPT="The sound of the guitar"
# TEXT_PROMPT="The lead electric guitar sound"
# TEXT_PROMPT="The sound of the lead melodic distorted guitar"
# TEXT_PROMPT="The sound of the backing guitar"
# TEXT_PROMPT="The backing guitar track"
# TEXT_PROMPT="Second guitar track"


## GEN / Inpaint Condition
NUM_STEPS=200
CFG_SCALE=1.0
OVERLAP_DUR=4.0
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