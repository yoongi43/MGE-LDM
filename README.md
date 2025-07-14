# 🎶 MGE-LDM

&#x20;

> **Official implementation of the paper Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction.**

This repository is being updated.

---

## 📋 Table of Contents

<!-- 1. [✨ Features](#✨-features) -->
1. [📖 Paper & Samples](#📖-paper--samples)
2. [⚙️ Installation](#⚙️-installation)
3. [💾 Model Checkpoints](#💾-model-checkpoints)
4. [🛠️ Process Overview](#🛠️-process-overview)
6. [🚀 Inference](#🚀-inference)
7. [🔗 Reference](#🔗-reference)
8. [📚 Citation](#📚-citation)

---

## 📖 Paper & Samples

* **Paper**: [arXiv](https://arxiv.org/abs/2505.23305)
* **Sample Page**: [link](https://yoongi43.github.io/MGELDM_Samples/)

---

## ⚙️ Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yoongi43/MGE-LDM.git
   cd MGE-LDM
   ```
2. **Create environment**:

   ```bash
   conda env create -n mgeldm python=3.9
   conda activate mge-ldm
   ```
3. **Install dependencies**:

   ```bash
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
   ```
   pip install -r requirements.txt
   ```
---

## 💾 Model Checkpoints

> **Note:** This checkpoint differs from the one used in the original paper; it was trained with additional data (MTG Jamendo, MedleyDB, etc.) and alternative hyperparameters.

Pretrained checkpoint will be available soon.

**AutoEncoder**: [autoencoder_checkpoint.ckpt](#)

**LDM (DiT)**: [dit_checkpoint.ckpt](#)

**CLAP Checkpoint**: download ```music_audioset_epoch_15_esc_90.14.pt``` from [laion_clap](https://github.com/LAION-AI/CLAP) repository.

---

## 🛠️ Process Overview

### 1. Download datasets
- [Slakh2100](https://zenodo.org/records/4599666)
- [MUSDB18](https://zenodo.org/records/3338373)
- [Moises](https://music.ai/research/)

### 2. Train AutoEncoder
Run the following script: ```bash scripts/train_ae.sh```


``` bash 
## See scripts/train_ae.sh

SAVE_DIR="/data2/yoongi/MGE_LDM"
CONFIG_NAME="default_ae" # See configs/default_ae.yaml for details.
GPU=0 # Set GPU ID
CKPT_PATH="" # Optional: Path to resume training

## Set CPU cores with `taskset -c` command.
CUDA_VISIBLE_DEVICES=$GPU \
taskset -c 64-79 \
python train_ae.py \
--config-name $CONFIG_NAME \
save_dir=$SAVE_DIR
# ckpt_path=$CKPT_PATH ## Add if resuming training.
```

After training, unwrap the AutoEncoder from pytorch lightning trainer by running the following script:

```bash scripts/unwrap_ae_script.sh```

```bash
## See scripts/unwrap_ae_script.sh
GPU=0 # Set GPU ID
CKPT_DIR="...MGE_LDM/default_ae/checkpoints/"
CKPT_PATH=$CKPT_DIR"last.ckpt"

OUTPUT_PATH=$CKPT_DIR"unwrapped_last"

CUDA_VISIBLE_DEVICES=$GPU \
python unwrap_model.py \
    --config-name default_ae \
    +type=autoencoder \
    ckpt_path=${CKPT_PATH} \
    +use_safetensors=false \
    +output_name=${OUTPUT_PATH}
```

### 3. Data Prepration for LDM training
- Encode audio into latent representations and compute CLAP latents.
- Precompute and save all latents to ```pre_extracted_latents/``` to avoid on-the-fly computation overhead.
- See ```scripts/pre_encode_script.sh``` for details.


```
Structure of pre_extracted_latents:

pre_extracted_latents/
├── zero_latent.npy           # For salient segment loading
├── slakh2100/
│   ├── train/
│   │   ├──track00000/
│   │   │   ├── mix.npy
│   │   │   ├── mix_clap.npy
│   │   │   ├──comb0/
│   │   │   │   ├── src.npy
│   │   │   │   ├── src_clap.npy
│   │   │   │   ├── submix.npy
│   │   │   │   ├── submix_clap.npy
│   │   │   │   └── comb_info.json
│   │   │   ├──comb1/
│   │   │   │   ├── src.npy
│   │   │   │   ├── src_clap.npy
│   │   │   │   ├── submix.npy
│   │   │   │   ├── submix_clap.npy
│   │   │   │   └── comb_info.json
│   │   │   ...
│   │   ├──track00001/
│   │   ...
│   │
│   ├── valid/... 
│   ...
│   ├── test/...  
│
├── musdb18/...  # optional datasets
...
├── moisesdb/...  # optional datasets
...
```
---
### 4. Train Latent Diffusion Model

Run the following script to train MGE-LDM:
```bash scripts/train_dit.sh```
```bash
## See scripts/train_dit.sh
## Set CPU cores with `taskset -c` command.
GPU=0 # Set GPU ID
# GPU="6,7" # For multi-GPU training

SAVE_DIR="/data2/yoongi/MGE_LDM"
CONFIG_NAME="dit" # See configs/dit.yaml for details.
CKPT_PATH="" # Optional: Path to resume training

CUDA_VISIBLE_DEVICES=$GPU \
taskset -c 16-79 \
python train_dit.py \
--config-name $CONFIG_NAME \
save_dir=$SAVE_DIR
# ckpt_path=$CKPT_PATH ## Add if resuming training.


# ## Mutlti-GPU Training
# CUDA_VISIBLE_DEVICES=$GPU \
# torchrun --nproc_per_node gpu train_dit.py \
# --config-name $CONFIG_NAME \
# save_dir=$SAVE_DIR
```

After training, unwrap DiT from pytorch lightning trainer by running the following script:

```bash scripts/unwrap_dit_script.sh```

```bash
## See scripts/unwrap_dit_script.sh

GPU=0 # Set GPU ID
## Set config name / checkpoint path to unwrap.
CONFIG_NAME="dit"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"last.ckpt"

OUTPUT_PATH=$CKPT_DIR"unwrapped_last"

CUDA_VISIBLE_DEVICES=$GPU \
python unwrap_model.py \
    --config-name $CONFIG_NAME \
    +type=mgeldm \
    ckpt_path=${CKPT_PATH} \
    +use_safetensors=false \
    +output_name=${OUTPUT_PATH}
```


## 🚀 Inference
### Total Generation
Run ```bash scripts/infer_total_gen.sh```
```bash
## See scripts/infer_total_gen.sh

GPU=0 # Set GPU ID

CONFIG_NAME="dit"
# Path to the trained LDM checkpoint. Can be downloaded from the model checkpoint section.
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt" 

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="total_gen" # Total Generation
GEN_AUDIO_DUR=30.0 # Set the duration of the generated audio in seconds.
GIVEN_WAV_PATH=null

## Set text prompt here
TEXT_PROMPT="Funky upbeat jazz with guitar, saxophone and piano"
# TEXT_PROMPT="Lo-fi hip hop beat with mellow jazzy chords and a smooth bassline"
# TEXT_PROMPT="Relaxing acoustic guitar instrumental with soft percussion"
# TEXT_PROMPT="Metal guitar riff with heavy distortion and fast-paced drums"
# TEXT_PROMPT="Upbeat electronic dance music with catchy synth melodies and driving bass"

## Generation Configuration
NUM_STEPS=50
CFG_SCALE=3.0
OVERLAP_DUR=5.0 ## Overlap duration in seconds for continuation.
REPAINT_N=4 ## Multiple repainting steps for better quality (for continuation)


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
```

### Source Extraction
Run ```bash scripts/infer_src_ext.sh```
```bash
## See scripts/infer_src_ext.sh
#!/bin/bash

GPU=0 # Set GPU ID
## Set checkpoint path
CONFIG_NAME="dit"
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

TEXT_PROMPT="The sound of vocals"
# TEXT_PROMPT="The sound of the bass"
# TEXT_PROMPT="The sound of drums"
# TEXT_PROMPT="The sound of guitars"
# TEXT_PROMPT="The sound of the synthesizer"

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
```
### Partial Generation (Source Imputation)
Run ```bash scripts/infer_partial_gen.sh```
```bash
## See scripts/infer_partial_gen.sh

GPU=0 # Set GPU ID
## Set checkpoint path
CONFIG_NAME="dit"
CKPT_DIR="/data2/yoongi/MGE_LDM/${CONFIG_NAME}/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_last.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="partial_gen"
GIVEN_WAV_PATH="data_sample/sakanaction_music/sakanaction_music_seg.wav"
# GIVEN_WAV_PATH="data_sample/bruno_24kmagic/bruno_24kmagic_seg.wav"
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
```



## 🔗 Reference

* **Stable Audio Tools**: [stable-audio-tools](https://github.com/yourusername/stable-audio-tools)
* **Friendly Stable Audio Tools**: [friendly-stable-audio-tools](https://github.com/yourusername/friendly-stable-audio-tools)

---

## 📚 Citation

```bibtex
@article{chae2025mge,
  title={MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction},
  author={Chae, Yunkee and Lee, Kyogu},
  journal={arXiv preprint arXiv:2505.23305},
  year={2025}
}
```
