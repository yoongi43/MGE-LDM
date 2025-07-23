#!/bin/bash


EXPNAME="default_ae" ## Set your experiment name here
DATASET=$1  ## Dataset to encode. Set to "all" to encode all datasets
GPU=$2 ## GPU ID to use

CKPT_DIR="/data2/yoongi/MGE_LDM/default_ae/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

CLAP_CKPT_PATH="/data2/yoongi/dataset/pre_trained/music_audioset_epoch_15_esc_90.14.pt"

SAVE_ROOT_DIR="/data2/yoongi/dataset/pre_extracted_latents/"

if [ "$DATASET" = "all" ]; then
    ## You should set valid paths to your dataset in pre_encode.py
    echo "No dataset specified. Using all datasets"
    DATASET_LIST=(
        slakh2100 
        musdb18hq 
        moisesdb 
        # medleydbV2
        # mtgjamendo 
        # other_tracks
        )
else
    echo "Using specified dataset list: $DATASET"
    IFS=',' read -r -a DATASET_LIST <<< "$DATASET"
    echo "Extracting from datasets: ${DATASET_LIST[*]}"
fi

echo "Number of datasets: ${#DATASET_LIST[@]}"
# exit 0

JOINED=$(IFS=,; echo "${DATASET_LIST[*]}")    # "slakh2100,musdb18hq,moisesdb,..."
LIST_OVERRIDE="[$JOINED]"

CUDA_VISIBLE_DEVICES=$GPU \
python pre_encode.py \
    --config-name $EXPNAME \
    +clap_ckpt_path=${CLAP_CKPT_PATH} \
    ckpt_path=${CKPT_PATH} \
    +save_root_dir=${SAVE_ROOT_DIR} \
    +extract_dataset=$LIST_OVERRIDE