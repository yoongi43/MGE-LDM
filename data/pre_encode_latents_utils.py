import json
import yaml
import os; opj=os.path.join
from glob import glob
from tqdm import tqdm
import numpy as np

import torch
import torchaudio 
from torchaudio.transforms import Resample

import typing as tp

"""
This code pre-extracts latents from multi-track datasets with a pre-trained autoencoder.
Supports following dataset:
    - Slakh2100
    - MUSDB18HQ
    - MoisesDB
    - MedleyDB2.0

Saved latents will be saved in the specified directory.
+ Saved_Latents
    - zero_latent.npy
    + Slakh2100
        + train
            + track00000
                - mix.npy
                - mix_clap.npy
                + comb0
                    - submix.npy, src.npy, submix_clap.npy, src_clap.npy, comb_info.json
                + comb1
            + track00001
        + valid
"""

# AE_CKPT_PATH = "/data2/yoongi/MGE_LDM/default_ae/checkpoints/unwrapped_last.ckpt"
# SR_AE = 16000
# SR_CLAP = 48000
# CLAP_CKPT_PATH = "/data2/yoongi/dataset/pre_trained/music_audioset_epoch_15_esc_90.14.pt"
# clap_kwargs= {
#     "output_dim": 512,
#     "clap_ckpt_path": CLAP_CKPT_PATH,
#     "audio_model_type": "HTSAT-base",
#     "enable_fusion": False,
#     "project_out": False,
#     "finetune": False
# }

def to_npy(tensor):
    return tensor.detach().cpu().numpy()

def check_len(waveforms):
    """
    Check if all waveforms have the same length.
    """
    if isinstance(waveforms, dict):
        waveforms = list(waveforms.values())
    
    lengths = [wav.shape[-1] for wav in waveforms]
    if len(set(lengths)) > 1:
        # raise ValueError("Waveforms must have the same length.")
        return False
    return lengths[0]  # Return the length of the first waveform

def adjust_seq_len(wav_tensor, length):
    tlen = wav_tensor.shape[-1]
    if tlen > length:
        wav_tensor = wav_tensor[:, :length]
    elif tlen < length:
        wav_tensor = torch.nn.functional.pad(wav_tensor, (0, length - tlen))
    return wav_tensor

def get_latent_embs_from_wav(
    wavs: torch.Tensor, # tensor, (B, 1, len)
    sr_ori: int,
    
    model_ae: torch.nn.Module, # pre-trained autoencoder model
    sr_ae: int, # sample rate of autoencoder (e.g., 16000)
    
    model_clap: torch.nn.Module, # pre-trained CLAP model
    sr_clap: int, # sample rate of CLAP (e.g., 48000)
    clap_seg_dur: float
):
    """
    We save all clap embedding for each second of the audio.
    The window length is determined by clap_seg_dur.
    """
    assert sr_clap == 48000, "CLAP sample rate must be 48000"
    # device = wavs.device
    device = model_clap.device
    wavs = wavs.to(device)  # wavs: (B, 1, len)
    resampler_ae = Resample(sr_ori, sr_ae).to(device)
    wavs_ae = resampler_ae(wavs)  # wavs_ae: (B, 1, len_ae)
    wav_len_adj = wavs_ae.shape[-1] - (wavs_ae.shape[-1] % model_ae.downsampling_ratio)
    wavs_ae = wavs_ae[:, :, :wav_len_adj]  # (B, 1, len_ae)
    
    ## Make AE latents
    with torch.no_grad():
        encoded = model_ae.encode(wavs_ae, return_vae_mean=True)  # (B, 64, len_ae)
    
    ## Make CLAP latent
    clap_emb_list = []
    resampler_clap = Resample(sr_ori, sr_clap).to(device)
    wavs_clap = resampler_clap(wavs)  # wavs_clap: (B, 1, len_clap)
    dur = wavs_clap.shape[-1] / sr_clap ## dur in seconds
    
    for i in range(0, int(dur - clap_seg_dur)):
        offset_st = int(i * sr_clap)
        offset_end = int(offset_st + clap_seg_dur * sr_clap)
        clap_input = wavs_clap[:, :, offset_st:offset_end]  # (B, 1, len_clap)
        
        clap_emb, mask = model_clap(clap_input)  # (B, 1, 512)
        clap_emb = clap_emb.squeeze(1)  # (B, 512)
        clap_emb_list.append(clap_emb)
    clap_embs = torch.stack(clap_emb_list, dim=-1)  # (B, 512, num_clap_frames)
    
    ## clap_embs:(B, 512, num_clap_frames)
    ## encoded: (B, 64, len_ae)
    # return clap_embs, encoded  
    return encoded, clap_embs
    

@torch.no_grad()
def save_combinations(save_dir, 
                      model_ae,
                      sample_rate_ae,
                      model_clap,
                      sample_rate_clap,
                      source_wavs: tp.Union[tp.Dict[str, torch.Tensor], tp.List[torch.Tensor]], # Dict of tensors, each (1, len)
                      mix_wav: torch.Tensor, # Can be none => make mix by summation
                      wav_sr_ori: int # original sample rate of wavs (e.g., 44100, 48000, etc.)
                      ):
    
    ## check_len
    # length = check_len(list(source_wavs.values())) if isinstance(source_wavs, dict) else check_len(source_wavs)
    validity = check_len(source_wavs)  # Ensure all source_wavs have the same length
    if not validity:
        print(f"Error: source_wavs must have the same length. Skipping saving combinations.")
        print(f"save_dir: {save_dir}")
        return None
    
    """
    Save combinations of source wavs and mix wavs as latents.
    """
    # assert len(source_wavs) >= 2, "source_wavs must contain at least two wavs."
    if len(source_wavs) < 2:
        print("Warning: source_wavs must contain at least two wavs.")
        print(f"save_dir: {save_dir}")
        print("source_wavs must contain at least two wavs. Skipping saving combinations.")
        return None
    
    if isinstance(source_wavs, list):
        inst_names = None
        wav_list = source_wavs
    elif isinstance(source_wavs, dict):
        inst_names = list(source_wavs.keys())
        wav_list = list(source_wavs.values())
    else:
        raise ValueError("source_wavs must be a list or a dict.")

    ## Adjust lengths: assure all wavs have the same length
    if mix_wav is not None:
        seqlen = mix_wav.shape[-1]
    else:
        seqlen = max([wav.shape[-1] for wav in wav_list])
    
    wav_list = [adjust_seq_len(wav, seqlen) for wav in wav_list]
    if mix_wav is None:
        mix_wav = sum(wav_list)
        
    ## save mixture
    mix = mix_wav.unsqueeze(0) # (1, 1, len)
    try:
        mix_latent, mix_clap = get_latent_embs_from_wav(
            wavs=mix,
            sr_ori=wav_sr_ori,
            model_ae=model_ae,
            sr_ae=sample_rate_ae,
            model_clap=model_clap,
            sr_clap=sample_rate_clap,
            clap_seg_dur=10.0,  # 10 seconds for CLAP segmentation
        )
    except Exception as e:
        print(f"Error processing mix wav: {e}")
        print("Mix wav shape:", mix.shape)
        print("wav_sr_ori:", wav_sr_ori)
        print(f"savedir: {save_dir}")
        print(f"Skipping saving combinations for this mix.")
        
        return None
        
    # mix_latent: # (1, 64, len_ae)
    # mix_clap: (1, 512, num_clap_frames)
    mix_latent = mix_latent.squeeze(0)  # (64, len_ae)
    mix_clap = mix_clap.squeeze(0)  # (512, num_clap_frames)
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(opj(save_dir, "mix.npy"), to_npy(mix_latent))
    np.save(opj(save_dir, "mix_clap.npy"), to_npy(mix_clap))
        
    ## make combinations
    for ii, label in enumerate(inst_names):
        save_comb_dir = opj(save_dir, f"comb{ii}")
        os.makedirs(save_comb_dir, exist_ok=True)
        
        wav_src = wav_list[ii]  # (1, len)
        wav_rest = sum(wav_list[:ii] + wav_list[ii+1:])  # (1, len)
        rest_label_list = [inst_names[k] for k in range(len(inst_names)) if k != ii]
        
        
        """
        in-batch, OOM error
        """
        # wavs = torch.stack([wav_src, wav_rest], dim=0)  # (2, 1, len)
        # latents, claps = get_latent_embs_from_wav(
        #     wavs=wavs,
        #     sr_ori=wav_sr_ori,
        #     model_ae=model_ae,
        #     sr_ae=sample_rate_ae,
        #     model_clap=model_clap,
        #     sr_clap=sample_rate_clap,
        #     clap_seg_dur=10.0,  # 10 seconds for CLAP segmentation
        # )
        # ## latents: (2, 64, len_ae)
        # ## claps: (2, 512, num_clap_frames)
        # latents = to_npy(latents)  # (2, 64, len_ae)
        # claps = to_npy(claps)  # (2, 512, num_clap_frames)
        # src_latent, src_clap = latents[0], claps[0]  # (64, len_ae), (512, num_clap_frames)
        # submix_latent, submix_clap = latents[1], claps[1]  # (64, len_ae), (512, num_clap_frames)
        """ separately"""
        latent_src, clap_src = get_latent_embs_from_wav(
            wavs=wav_src.unsqueeze(0),  # (1, 1, len)
            sr_ori=wav_sr_ori,
            model_ae=model_ae,
            sr_ae=sample_rate_ae,
            model_clap=model_clap,
            sr_clap=sample_rate_clap,
            clap_seg_dur=10.0,  # 10 seconds for CLAP segmentation
        )
        latent_rest, clap_rest = get_latent_embs_from_wav(
            wavs=wav_rest.unsqueeze(0),  # (1, 1, len)
            sr_ori=wav_sr_ori,
            model_ae=model_ae,
            sr_ae=sample_rate_ae,
            model_clap=model_clap,
            sr_clap=sample_rate_clap,
            clap_seg_dur=10.0,  # 10 seconds for CLAP segmentation
        )
        src_latent, src_clap = to_npy(latent_src.squeeze(0)), to_npy(clap_src.squeeze(0))  # (64, len_ae), (512, num_clap_frames)
        submix_latent, submix_clap = to_npy(latent_rest.squeeze(0)), to_npy(clap_rest.squeeze(0))  # (64, len_ae), (512, num_clap_frames)
        
        os.makedirs(save_comb_dir, exist_ok=True)
        np.save(opj(save_comb_dir, "src.npy"), src_latent)  # (64, len_ae)
        np.save(opj(save_comb_dir, "submix.npy"), submix_latent)  # (64, len_ae)
        np.save(opj(save_comb_dir, "src_clap.npy"), src_clap)  # (512, num_clap_frames)
        np.save(opj(save_comb_dir, "submix_clap.npy"), submix_clap)  # (512, num_clap_frames)
        

        comb_info = {
            "src_label": label,
            "submix_label_list": rest_label_list,
            "wav_sr_ori": wav_sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.)
            "sample_rate_ae": sample_rate_ae,
            "sample_rate_clap": sample_rate_clap,
        }
        with open(opj(save_comb_dir, "comb_info.json"), "w") as f:
            json.dump(comb_info, f, indent=4)
        

# def load_ae_model(ckpt_path, cfg_path):
#     with open(cfg_path, "r") as f:
#         cfg = json.load(f)
        
#     cfg_pre = cfg["model"]["pretransform"]
#     sr = 16000
#     assert cfg["sample_rate"] == sr

#     model = create_pretransform_from_config(cfg_pre, sr)
#     model.load_state_dict(
#         load_ckpt_state_dict(ckpt_path), strict=True
#     )
#     return model


def extract_zero_latent(
    save_root_dir,
    model_ae,
    sample_rate_ae,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    
    wavlen = int(sample_rate_ae * 20)
    wavlen = wavlen - (wavlen % model_ae.downsampling_ratio)  # Adjust to AE downsampling ratio
    wavesilence = torch.zeros((1, 1, wavlen)).to(device)
    with torch.no_grad():
        encoded = model_ae.encode(wavesilence, return_vae_mean=True)  # (1, 64, len_ae)
    # encoded: (1, 64, len_ae)
    zero_latent = encoded[0, :, 100] # (64, )
    zero_latent = to_npy(zero_latent)  # Convert to numpy array
    os.makedirs(save_root_dir, exist_ok=True)
    np.save(opj(save_root_dir, "zero_latent.npy"), zero_latent)    


def extract_latents_slakh2100(
    data_dir, # dataset/slakh2100_inst/slakh2100_flac_redux_wav/
    save_root_dir, ## 'saved_latents"
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    
    save_name = "slakh2100"
    save_dir = opj(save_root_dir, save_name)
    
    split_list = ["train", "validation", "test"]
    # split_list = ["validation", "test"]
    for split in split_list:
        split_dir = opj(data_dir, split)
        if split == "validation":
            split = "valid"
        save_split_dir = opj(save_dir, split)
        # os.makedirs(save_split_dir, exist_ok=True)
        
        track_list = sorted(os.listdir(split_dir)) 
        # for track in tqdm(track_list, desc="Processing Slakh2100", unit="track"):
        for idx, track in enumerate(tqdm(track_list, desc="Processing Slakh2100", unit="track")):
            # if idx < 760:
            #     continue
            track_dir = opj(split_dir, track)
            save_track_dir = opj(save_split_dir, f"track{idx:05d}")
            # os.makedirs(save_track_dir, exist_ok=True)
            
            # with open(opj(track_dir, "metadata.json"), "r") as f:
            #     metadata = json.load(f)
            with open(opj(track_dir, "metadata.yaml"), "r") as f:
                metadata = yaml.safe_load(f)
            
            # wav_mix, sr_ori = torchaudio.load(opj(track_dir, "mix.wav"))
            max_length = 0
            waveform_dict = {}
            stem_dict = metadata["stems"]
            for stemname in stem_dict.keys():
                ## stemname: S00, S01,...
                if stem_dict[stemname]["audio_rendered"] is False:
                    continue
                inst_class = stem_dict[stemname]["inst_class"]
                wav_stem, sr_ori = torchaudio.load(opj(track_dir, "stems", f"{stemname}.wav"))
                
                if wav_stem.shape[-1] > max_length:
                    max_length = wav_stem.shape[-1]
                ## sr_ori = 44100
                if inst_class == "Strings (continued)":
                    inst_class = "Strings"
                    # import pdb; pdb.set_trace()
                if inst_class not in waveform_dict.keys():
                    waveform_dict[inst_class] = [wav_stem]
                else:
                    waveform_dict[inst_class].append(wav_stem)
            
            ## Adjust lengths of waveforms to the maximum length      
            for stemname in waveform_dict.keys():
                wav_list =  waveform_dict[stemname]
                wav_list = [adjust_seq_len(wav, max_length) for wav in wav_list]
                wav_sum = sum(wav_list)  # Sum all wavs of the same instrument
                waveform_dict[stemname] = wav_sum # (1, len)
                
                                

            ## Save combination
            save_combinations(
                save_dir=save_track_dir,
                model_ae=model_ae,
                sample_rate_ae=sample_rate_ae,
                model_clap=model_clap,
                sample_rate_clap=sample_rate_clap,
                source_wavs=waveform_dict,  # Dict of tensors, each (1, len)
                mix_wav=None,  #
                wav_sr_ori=sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.
            )
            

def extract_latents_musdb18hq(
    data_dir, # dataset/musdb18hq_from_zenodo/
    save_root_dir, ## 'saved_latents'
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    
    save_name = "musdb18"
    save_dir = opj(save_root_dir, save_name)
    
    split_list = ["train", "test"]
    for split in split_list:
        split_dir = opj(data_dir, split)
        save_split_dir = opj(save_dir, split)
        # os.makedirs(save_split_dir, exist_ok=True)
        
        track_list = sorted(os.listdir(split_dir)) 
        # for track in tqdm(track_list, desc="Processing MUSDB18HQ", unit="track"):
        for idx, track in enumerate(tqdm(track_list, desc="Processing MUSDB18HQ", unit="track")):
            track_dir = opj(split_dir, track)
            save_track_dir = opj(save_split_dir, f"track{idx:05d}")
            # os.makedirs(save_track_dir, exist_ok=True)
            
            stem_list = ["vocals", "drums", "bass", "other"]
            
            waveform_dict = {}
            
            for stemname in stem_list:
                wav_stem, sr_ori = torchaudio.load(
                    opj(track_dir, f"{stemname}.wav")
                )
                wav_stem = torch.mean(wav_stem, dim=0, keepdim=True) # (1, len)
                waveform_dict[stemname] = wav_stem
                
            ## Save combination
            save_combinations(
                save_dir=save_track_dir,
                model_ae=model_ae,
                sample_rate_ae=sample_rate_ae,
                model_clap=model_clap,
                sample_rate_clap=sample_rate_clap,
                source_wavs=waveform_dict,  # Dict of tensors, each (1, len)
                mix_wav=None, 
                wav_sr_ori=sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.
            )
            
            
def extract_latents_moisesdb(
    data_dir, # "dataset/moisesdb/moisesdb_v0.1/"
    save_root_dir, ## 'saved_latents'
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
    split_seed = 42, # random seed for train/test split
):
    from sklearn.model_selection import train_test_split
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    
    save_name = "moisesdb"
    save_dir = opj(save_root_dir, save_name)
    
    track_list = sorted(os.listdir(data_dir))
    train_list, test_list = train_test_split(
        track_list, test_size=0.1, random_state=split_seed
    )
    
    split_list = ["train", "test"]
    for split in split_list:
        if split == "train":
            track_list = train_list
        else:
            track_list = test_list
        save_split_dir = opj(save_dir, split)
        # os.makedirs(save_split_dir, exist_ok=True)
        for idx, track in enumerate(tqdm(track_list, desc="Processing MoisesDB", unit="track")):
            track_dir = opj(data_dir, track)
            save_track_dir = opj(save_split_dir, f"track{idx:05d}")
            # os.makedirs(save_track_dir, exist_ok=True)
            
            waveform_dict ={}
            stem_list = os.listdir(track_dir)
            stem_list.remove("data.json")
            
            # seglen_ori = None
            max_length = 0
            
            for stemname in stem_list:
                stemdir = opj(track_dir, stemname)
                files = glob(opj(stemdir, "*.wav"))
                wav_stem_list = []
                
                for f in files:
                    wav_stem, sr_ori = torchaudio.load(f)
                    # if seglen_ori is None:
                    #     seglen_ori = wav_stem.shape[-1]
                    if wav_stem.shape[-1] > max_length:
                        max_length = wav_stem.shape[-1]
                    wav_stem = torch.mean(wav_stem, dim=0, keepdim=True)
                    # wav_stem = adjust_seq_len(wav_stem, seglen_ori)
                    # wav_stem = adjust_seq_len(wav_stem, max_length)  # Adjust to the maximum length
                    wav_stem_list.append(wav_stem)
                max_in_file = max([wav.shape[-1] for wav in wav_stem_list])
                wav_stem_list = [adjust_seq_len(wav, max_in_file) for wav in wav_stem_list]
                wav_stem = sum(wav_stem_list)
                if stemname == "other_keys":
                    stemname = "synths"
                waveform_dict[stemname] = wav_stem
            
            for stemname in waveform_dict.keys():
                wav_stem = waveform_dict[stemname]
                if wav_stem.shape[-1] < max_length:
                    wav_stem = adjust_seq_len(wav_stem, max_length)
                    waveform_dict[stemname] = wav_stem
                
            ## Save combination
            save_combinations(
                save_dir=save_track_dir,
                model_ae=model_ae,
                sample_rate_ae=sample_rate_ae,
                model_clap=model_clap,
                sample_rate_clap=sample_rate_clap,
                source_wavs=waveform_dict,  # Dict of tensors, each (1, len)
                mix_wav=None, 
                wav_sr_ori=sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.
            )
        
def extract_latents_medleydbV2(
    data_dir, # dataset/MedleyDB_V2
    save_root_dir, ## 'saved_latents'
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    save_name = "medleydbv2"
    save_dir = opj(save_root_dir, save_name)
    
    track_list = sorted(os.listdir(data_dir))
    for idx, track in enumerate(tqdm(track_list, desc="Processing MedleyDB V2", unit="track")):
        track_dir = opj(data_dir, track)
        save_track_dir = opj(save_dir, f"track{idx:05d}")
        # os.makedirs(save_track_dir, exist_ok=True)
        
        # with open(opj(track_dir, "metadata.yaml"), "r") as f:
        #     metadata = yaml.safe_load(f)
        with open(opj(track_dir, f"{track}_METADATA.yaml"), "r") as f:
            metadata = yaml.safe_load(f)
        
        waveform_dict = {}

        stem_dict = metadata["stems"] # {"S01": {...}, "S02": {...}, ...}
        stems_dir = opj(track_dir, metadata["stem_dir"])
        
        
        data_len = None
        for stemname in stem_dict.keys():
            instrument = stem_dict[stemname]["instrument"]
            instrument = instrument.replace(" ", "_")  
            
            filename = stem_dict[stemname]["filename"]
            
            wav_stem, sr_ori = torchaudio.load(opj(stems_dir, filename))
            wav_stem = torch.mean(wav_stem, dim=0, keepdim=True) # (1, len)
            if instrument not in waveform_dict.keys():
                waveform_dict[instrument] = wav_stem
            else:
                try:
                    waveform_dict[instrument] += wav_stem
                except:
                    print(f"Error adding wavs for {instrument} in {track}. Skipping this instrument.")
                    print(f"Wav shape: {wav_stem.shape}")
                    print(f"prev_wav shape: {waveform_dict[instrument].shape}")
                    print("Skipping saving combinations for this track.")
                    continue
        ## all stems in waveform_dict shoul have the same length
        check_len = [wav.shape[-1] for wav in waveform_dict.values()]
        check_len = set(check_len)
        if len(check_len) > 1:
            print(f"Error: Stems in {track} have different lengths. Skipping this track.")
            print(f"Lengths: {check_len}")
            continue
        
        ## Save combination
        save_combinations(
            save_dir=save_track_dir,
            model_ae=model_ae,
            sample_rate_ae=sample_rate_ae,
            model_clap=model_clap,
            sample_rate_clap=sample_rate_clap,
            source_wavs=waveform_dict,  # Dict of tensors, each (1, len)
            mix_wav=None, 
            wav_sr_ori=sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.
        )
        

def extract_latents_mtgjamendo(
    data_dir, # dataset/mtg_jamendo/
    save_root_dir, 
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    save_name = "mtg_jamendo"
    save_dir = opj(save_root_dir, save_name)
    
    
    subset0 = range(0, 100)
    subset1 = range(0, 27)
    
    subset2 = range(27, 50)
    subset3 = range(50, 75)
    subset4 = range(75, 100)
    
    subset = subset0
    
    subdir_list = sorted(os.listdir(data_dir)) #00, 01, ...
    for idx, subdir in enumerate(tqdm(subdir_list, desc="Processing MTG Jamendo", unit="subdir")):
        if idx not in subset:
            continue
        
        
        subdir = opj(data_dir, subdir)
        data_list = os.listdir(subdir)
        # data_list = [d for d in data_list if not os.path.
        data_list = [d for d in data_list if (d.endswith(".wav") or d.endswith(".mp3"))]  # Only wav or flac files
        for data in tqdm(data_list, desc=f"Processing {subdir}", unit="data"):
            wav, sr = torchaudio.load(opj(subdir, data))
            wav = torch.mean(wav, dim=0, keepdim=True)
            dataname = data.split(".")[0]
            subdir_idx = subdir.split("/")[-1]  # e.g., "00", "01", ...
            save_track_dir = opj(save_dir, subdir_idx, dataname)
            # import pdb; pdb.set_trace()
            # os.makedirs(save_track_dir, exist_ok=True)
            
            mix = wav.unsqueeze(0)
            try:
                mix_latent, mix_clap = get_latent_embs_from_wav(
                    wavs=mix,
                    sr_ori=sr,
                    model_ae=model_ae,
                    sr_ae=sample_rate_ae,
                    model_clap=model_clap,
                    sr_clap=sample_rate_clap,
                    clap_seg_dur=10.0,  # 10 seconds for CLAP segmentation
                )
            except Exception as e:
                print(f"Error processing wav {data} in {subdir}: {e}")
                print("Wav shape:", wav.shape)
                print("sr:", sr)
                print(f"Skipping saving combinations for this wav.")
                continue
            # mix_latent: # (1, 64, len_ae)
            # mix_clap: (1, 512, num_clap_frames)
            mix_latent = mix_latent.squeeze(0)
            mix_clap = mix_clap.squeeze(0)
            os.makedirs(save_track_dir, exist_ok=True)
            np.save(opj(save_track_dir, "mix.npy"), to_npy(mix_latent))
            np.save(opj(save_track_dir, "mix_clap.npy"), to_npy(mix_clap))
            

def extract_latents_othertracks(
    data_dir, # 
    save_root_dir, ## 'saved_latents'
    model_ae,
    sample_rate_ae,
    model_clap,
    sample_rate_clap,
    device,
):
    model_ae.to(device)
    model_ae.eval()
    model_clap.set_device(device)
    savename = "other_tracks"
    save_dir = opj(save_root_dir, savename)
    
    track_list = sorted(os.listdir(data_dir))
    for idx, track in enumerate(tqdm(track_list, desc="Processing Other Tracks", unit="track")):
        track_dir = opj(data_dir, track)
        save_track_dir = opj(save_dir, f"track{idx:05d}")
        # os.makedirs(save_track_dir, exist_ok=True)
        
        wavs = glob(opj(track_dir, "*.wav"))
        wavs = [w for w in wavs if "mix.wav" not in w]  # Exclude mix.wav if exists
        if len(wavs) == 0:
            print(f"No wav files found in {track_dir}, skipping...")
            continue
        
        # stem_list = [os.path.basename(wav).split(".")[0] for wav in wavs]
        waveform_dict = {}
        for wav in wavs:
            wav_stem, sr_ori = torchaudio.load(wav)
            wav_stem = torch.mean(wav_stem, dim=0, keepdim=True)
            stemname = os.path.basename(wav).split(".")[0]
            waveform_dict[stemname] = wav_stem
        ## Save combination
        
        save_combinations(
            save_dir=save_track_dir,
            model_ae=model_ae,
            sample_rate_ae=sample_rate_ae,
            model_clap=model_clap,
            sample_rate_clap=sample_rate_clap,
            source_wavs=waveform_dict,  # Dict of tensors, each (1, len)
            mix_wav=None, 
            wav_sr_ori=sr_ori,  # original sample rate of wavs (e.g., 44100, 48000, etc.
        )