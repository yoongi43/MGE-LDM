import torch
from torch.utils.data import Dataset, DataLoader


import numpy as np
import os; opj = os.path.join
from glob import glob
from tqdm import tqdm
from time import time
import yaml
import json

import typing as tp

NUM_TRIES=15

from .dataset_single import ConcatDataset


def get_salient_latents(
    *,
    latent: tp.Union[np.ndarray, str], ## path or latent (64, length)
    segment_len: int, # 80
    num_tries,
    latent_zero, #(64)
    threshold, # (0.07)
    state: np.random.RandomState,
):
    """
    Load non-silent latent as possible
    """
    if isinstance(latent, str):
        latent = np.load(latent)
    
    if latent_zero is not None:
        latent_zero = latent_zero.reshape(-1, 1) # (64, 1)
    else:
        assert threshold is None, "If latent_zero is None, threshold must be None"
    
    total_len = latent.shape[-1]
    lower_bound = 0
    upper_bound = max(total_len - segment_len, 0)
    
    if threshold is None:
        offset_st = state.randint(lower_bound, upper_bound)
        offset_end = offset_st + segment_len
        latent_excerpt = latent[..., offset_st:offset_end]
    else:
        diff_abs = -np.inf
        num_try = 0
        while diff_abs <= threshold:
            offset_st = state.randint(lower_bound, upper_bound)
            offset_end = offset_st + segment_len
            latent_excerpt = latent[..., offset_st:offset_end] # (64, 80)
            diff_abs = np.abs(np.mean(latent_excerpt-latent_zero)).item()
            num_try += 1
            if num_tries is not None and num_try >= num_tries:
                break
    
    return latent_excerpt, offset_st, offset_end


def get_prompt_from_labels(label_list, state=None):
    """
    Convert a list of labels into a single natural-language prompt.
    - Replace '_' with space, lowercase everything.
    - Join multiple labels with commas and 'and'.
    - Pick one of several templates at random, using the provided RNG state.

    Args:
        label_list (list of str): e.g. ["vocals", "drums", "electric_guitar"]
        state (np.random.RandomState or None): if None, a new RandomState is created.
    Returns:
        str: a prompt string.
    """
    # 1) RNG setup
    rng = state if state is not None else np.random.RandomState()

    # 2) Normalize labels
    labels = [lab.replace('_', ' ').lower() for lab in label_list]

    # 3) Join into a natural list string
    if len(labels) == 0:
        raise ValueError("Label list is empty, cannot generate prompt.")
    elif len(labels) == 1:
        joined = labels[0]
    elif len(labels) == 2:
        joined = f"{labels[0]} and {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", and {labels[-1]}"

    # 4) Prompt templates (instrument/stem only, no verbs)
    
    templates = [
        "The sound of {labels}.",
        "The instrumental sound of {labels}.",
    ]
    
    templates += [
        "Only {labels}.",
        "{labels} sounds.",
        "Instrumental {labels}.",
        "Stem of {labels}.",
        "Pure {labels}.",
        "Audio of {labels}.",
        "{labels} part.",
        "{labels} stem.",
        "{labels} segment."
    ]
    
    templates += [
        "{labels}",
        "Only {labels}",
        "{labels} stem",
        "{labels} track",
        "{labels} audio",
        "Stem: {labels}",
        "Track: {labels}",
        "Audio: {labels}",
    ]
    templates += [
        "The music of {labels}.",
        "The performance of {labels}.",
        "The track of {labels}.",
        "The audio track of {labels}.",
        "The {labels}",
        "Audio elements: {labels}.",
    ]

    # 5) Select one template at random
    template = rng.choice(templates)
    return template.format(labels=joined)

class MixSubmixSourceLatentDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        segment_length,
        split="train",
        num_examples=10000,
        zero_diff_threshold=0.1,
        zero_latent_path=None,
        num_tracks_debug=None,
        state_id=None,
        # return_src_txt=True,
        return_mix_only=False, ## For mixture only pre-training
        ## For duration calculation
        ae_comp_ratio=2048,
        ae_sample_rate=16000,
    ):
        super().__init__()
        assert zero_latent_path is not None, "zero_latent_path must be provided"
        
        self.dataset_name = os.path.basename(data_dir)
        
        if split is not None:
            self.data_dir = opj(data_dir, split)
        else:
            self.data_dir = data_dir
        self.segment_length = segment_length
        self.num_examples = num_examples
        self.zero_diff_threshold = zero_diff_threshold
        self.zero_latent_path = zero_latent_path
        self.num_tracks_debug = num_tracks_debug
        self.state_id = state_id
        # self.return_src_txt = return_src_txt
        self.return_mix_only = return_mix_only
        self.ae_comp_ratio = ae_comp_ratio
        self.ae_sample_rate = ae_sample_rate
        
        if self.state_id is not None:
            self.state = np.random.RandomState(self.state_id)
        else:
            self.state = None
            
        self.latent_zero = np.load(self.zero_latent_path) if self.zero_latent_path else None

        self.tracks = os.listdir(self.data_dir)
        self.tracks = sorted(self.tracks)
        if self.num_tracks_debug is not None:
            self.tracks = self.tracks[:self.num_tracks_debug]
        
        self._preload_dataset()
        
        if len(self.data_dict) == 0:
            raise ValueError(
                f"No data for dataset dir: {data_dir}. "
            )
            
        self.latent_to_dur_ratio = ae_comp_ratio / ae_sample_rate
        
        

        

    def _preload_dataset(self):
        """
        track00000
        - mix_clap.npy: (512, num_sec)
        - mix.npy: (64, T)
        + comb0
            - src.npy: (64, T)
            - src_clap.npy: (512, num_sec)
            - submix.npy: (64, T)
            - submix_clap.npy: (512, num_sec)
            - comb_info.json:
                {
                    "src_label": "drums",
                    "submix_label_list": [
                        "bass",
                        "piano",
                        "electric_guitar"
                    ],
                    "wav_sr_ori": 44100,
                    "sample_rate_ae": 16000,
                    "sample_rate_clap": 48000
                }
        + comb1
            - ...
        """
        """
        We load all the data into memory.
        """
        self.data_dict = {}
        pbar = tqdm(self.tracks, desc=f"Preloading {self.dataset_name} dataset", unit="track")
        
        for track in pbar:
            self.data_dict[track] = {}
            track_dir = opj(self.data_dir, track)
            zmix = np.load(opj(track_dir, "mix.npy"))  
            cmix = np.load(opj(track_dir, "mix_clap.npy"))
            if np.isnan(zmix).any() or np.isnan(cmix).any():
                print(f"NaN found in {track}. Skipping this track.")
                continue
            
            self.data_dict[track]["mix"] = zmix
            self.data_dict[track]["mix_clap"] = cmix
            
            if self.return_mix_only is False:
                comb_dir_list = glob(opj(track_dir, "comb*"))
                for comb_dir in comb_dir_list:
                    comb_name = os.path.basename(comb_dir) 
                    
                    with open(opj(comb_dir, "comb_info.json"), "r") as f:
                        comb_info = json.load(f)
                    ## If 'other' is in the src label, we don't use this combination.
                    if "other" in comb_info["src_label"].lower():
                        continue
                    
                    self.data_dict[track][comb_name] = {}
                    
                    zsrc = np.load(opj(comb_dir, "src.npy"))
                    csrc = np.load(opj(comb_dir, "src_clap.npy"))
                    zsubmix = np.load(opj(comb_dir, "submix.npy"))
                    csubmix = np.load(opj(comb_dir, "submix_clap.npy"))
                    
                    if np.isnan(zsrc).any() or np.isnan(csrc).any() or \
                       np.isnan(zsubmix).any() or np.isnan(csubmix).any():
                        print(f"NaN found in {comb_name} of {track}. Skipping this combination.")
                        continue
                    
                    src_inst = comb_info["src_label"]
                    submix_inst = comb_info["submix_label_list"]
                    
                    self.data_dict[track][comb_name]["src"] = zsrc
                    self.data_dict[track][comb_name]["src_clap"] = csrc
                    self.data_dict[track][comb_name]["submix"] = zsubmix
                    self.data_dict[track][comb_name]["submix_clap"] = csubmix
                    self.data_dict[track][comb_name]["src_inst"] = src_inst
                    self.data_dict[track][comb_name]["submix_inst"] = submix_inst
        
        return self.data_dict
        
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.state:
            state = self.state
        else:
            state = np.random.RandomState((int(time()) % (2 ** 32)-1)+idx)
        
        track = state.choice(self.tracks)
        track_data = self.data_dict[track]
        
        
        ## Load mix
        zmix = track_data["mix"]
        cmix = track_data["mix_clap"]
        
        ## Load src, submix
        if self.return_mix_only:
            ## Random segment from mix
            zmix, offset_st, offset_end = get_salient_latents(
                latent=zmix,
                segment_len=self.segment_length,
                num_tries=NUM_TRIES,
                latent_zero=self.latent_zero,
                threshold=self.zero_diff_threshold,
                state=state
            )
            clap_len = cmix.shape[-1]
            offset_clap = int(offset_st * self.latent_to_dur_ratio)
            offset_clap = min(offset_clap, clap_len - 1)
            cmix = cmix[:, offset_clap]
        
            zsrc = np.zeros_like(zmix)
            csrc = np.zeros_like(cmix)
            src_prompt = ""
            
            zsubmix = zmix
            csubmix = cmix
            submix_prompt = ""
        else:
            comb_name_list = [c for c in track_data.keys() if c.startswith("comb")]
            comb_name = state.choice(comb_name_list)
            comb_data = track_data[comb_name] 
            
            zsrc = comb_data["src"]
            csrc = comb_data["src_clap"]
            src_inst = comb_data["src_inst"]
            
            zsubmix = comb_data["submix"]
            csubmix = comb_data["submix_clap"]
            submix_inst = comb_data["submix_inst"]
            
            ## sanity check
            assert zsrc.shape == zsubmix.shape == zmix.shape
            assert csrc.shape == csubmix.shape == cmix.shape
            
            ## Salient latent for src
            zsrc, offset_st, offset_end = get_salient_latents(
                latent=zsrc,
                segment_len=self.segment_length,
                num_tries=NUM_TRIES,
                latent_zero=self.latent_zero,
                threshold=self.zero_diff_threshold,
                state=state
            )
            
            clap_len = csrc.shape[-1]
            offset_clap = int(offset_st * self.latent_to_dur_ratio) ## => "seconds"
            offset_clap = min(offset_clap, clap_len - 1)
            csrc = csrc[:, offset_clap]
            
            zsubmix = zsubmix[..., offset_st:offset_end]
            csubmix = csubmix[:, offset_clap]
            
            zmix = zmix[..., offset_st:offset_end]
            cmix = cmix[:, offset_clap]
            
            ## Get prompts
            src_prompt = get_prompt_from_labels([src_inst], state=state)
            submix_prompt = get_prompt_from_labels(submix_inst, state=state)
            
        items = {
            "mix_latent": zmix,
            "mix_clap": cmix,
            
            "src_latent": zsrc,
            "src_clap": csrc,
            "src_prompt": src_prompt,
            
            "submix_latent": zsubmix,
            "submix_clap": csubmix,
            "submix_prompt": submix_prompt,
        }
            
        return items
    

class MTGJamendoLatentDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        segment_length,
        split="train",
        num_examples=10000,
        zero_diff_threshold=0.1,
        zero_latent_path=None,
        num_tracks_debug=None,
        state_id=None,
        # return_src_txt=True,
        return_mix_only=True, ## For mixture only pre-training
        ## For duration calculation
        ae_comp_ratio=2048,
        ae_sample_rate=16000,
    ):
        super().__init__()
        assert zero_latent_path is not None, "zero_latent_path must be provided"
        assert return_mix_only is  True
        
        self.dataset_name = os.path.basename(data_dir)
        
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.num_examples = num_examples
        self.zero_diff_threshold = zero_diff_threshold
        self.zero_latent_path = zero_latent_path
        self.num_tracks_debug = num_tracks_debug
        self.state_id = state_id
        # self.return_src_txt = return_src_txt
        self.return_mix_only = return_mix_only
        self.ae_comp_ratio = ae_comp_ratio
        self.ae_sample_rate = ae_sample_rate
        
        if self.state_id is not None:
            self.state = np.random.RandomState(self.state_id)
        else:
            self.state = None
            
        self.latent_zero = np.load(self.zero_latent_path) if self.zero_latent_path else None
        
        self.idx_dirs = sorted(os.listdir(data_dir)) ## 00~99
        assert len(self.idx_dirs) == 100
        if split == "train":
            self.idx_dirs = self.idx_dirs[:98]
        elif split in ["valid", "validation"]:
            self.idx_dirs = self.idx_dirs[98:]
        else: 
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'valid'.")

        if num_tracks_debug is not None:
            self.idx_dirs = self.idx_dirs[:num_tracks_debug]

        # self.tracks = os.listdir(self.data_dir)
        # self.tracks = sorted(self.tracks)
        # if self.num_tracks_debug is not None:
        #     self.tracks = self.tracks[:self.num_tracks_debug]
        
        self._preload_dataset()
        
        if len(self.data_dict) == 0:
            raise ValueError(
                f"No data for dataset dir: {data_dir}. "
            )
            
        self.latent_to_dur_ratio = ae_comp_ratio / ae_sample_rate
        
        
    def _preload_dataset(self):
        """
        track00000
        - mix_clap.npy: (512, num_sec)
        - mix.npy: (64, T)
        """
        """
        We load all the data into memory.
        """
        self.data_dict = {}
        
        pbar = tqdm(self.idx_dirs, desc=f"Preloading {self.dataset_name} dataset", unit="directory")
        # pbar = tqdm(self.tracks, desc=f"Preloading {self.dataset_name} dataset", unit="track")
        data_count = 0
        for idx_dir in pbar:
            trackdir_list = os.listdir(opj(self.data_dir, idx_dir))
            for trackdir in trackdir_list:
                zmix = np.load(opj(self.data_dir, idx_dir, trackdir, "mix.npy"))
                cmix = np.load(opj(self.data_dir, idx_dir, trackdir, "mix_clap.npy"))
                
                ## IF there is NaN in the mix, we skip this track
                if np.isnan(zmix).any() or np.isnan(cmix).any():
                    print(f"NaN found in {idx_dir}/{trackdir}. Skipping this track.")
                    continue
                
                self.data_dict[f"{idx_dir}_{trackdir}"] = {}
                self.data_dict[f"{idx_dir}_{trackdir}"]["mix"] = zmix
                self.data_dict[f"{idx_dir}_{trackdir}"]["mix_clap"] = cmix
                data_count += 1
        return self.data_dict            
        
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.state:
            state = self.state
        else:
            state = np.random.RandomState((int(time()) % (2 ** 32)-1)+idx)
        
        data_key = state.choice(list(self.data_dict.keys()))
        data = self.data_dict[data_key]
        
        ## Load mix
        zmix = data["mix"]
        cmix = data["mix_clap"]
        
        # print(""f"Mix shape: {zmix.shape}, Clap shape: {cmix.shape}")
        
        ## Random segment from mix
        zmix, offset_st, offset_end = get_salient_latents(
            latent=zmix,
            segment_len=self.segment_length,
            num_tries=NUM_TRIES,
            latent_zero=self.latent_zero,
            threshold=self.zero_diff_threshold,
            state=state
        )
        clap_len = cmix.shape[-1]
        offset_clap = int(offset_st * self.latent_to_dur_ratio)
        offset_clap = min(offset_clap, clap_len - 1)
        cmix = cmix[:, offset_clap]
    
        zsrc = np.zeros_like(zmix)
        csrc = np.zeros_like(cmix)
        src_prompt = ""
        
        zsubmix = zmix
        csubmix = cmix
        submix_prompt = ""
        
        ## Get mix prompt
        items = {
            "mix_latent": zmix,
            "mix_clap": cmix,
            
            "src_latent": zsrc,
            "src_clap": csrc,
            "src_prompt": src_prompt,
            
            "submix_latent": zsubmix,
            "submix_clap": csubmix,
            "submix_prompt": submix_prompt,
        }
        
        return items
    
            

def create_multi_latent_dataloader_from_config(
    dataset_configs,
    split: str,
    batch_size: int,
    sample_rate: int,
    segment_length: int,
    mixture_only_pretraining: bool = False,
    
    ae_comp_ratio: int = None,
    ## loader params
    num_workers: int = 8,
    shuffle: bool = True,
):
    dataset_list = []
    dataset_configs = dataset_configs[split]
    
    for dataset_name, dataset_cfg in dataset_configs.items():
        if dataset_name == "mtg_jamendo":
            assert mixture_only_pretraining, "MTG Jamendo dataset is only for mixture-only pretraining"
            dset_cls = MTGJamendoLatentDataset
        else:
            dset_cls = MixSubmixSourceLatentDataset
            
        dataset = dset_cls(
            data_dir=dataset_cfg["data_dir"],
            segment_length=segment_length,
            split=dataset_cfg.get("split", None),
            num_examples=dataset_cfg["num_examples"],
            zero_diff_threshold=dataset_cfg["zero_diff_threshold"],
            zero_latent_path=dataset_cfg["zero_latent_path"],
            num_tracks_debug=dataset_cfg.get("num_tracks_debug", None),
            state_id=dataset_cfg.get("state_id", None),
            return_mix_only=mixture_only_pretraining,
            ae_comp_ratio=ae_comp_ratio,
            ae_sample_rate=sample_rate,
        )
        
        dataset_list.append(dataset)
    
    dataset_cat = ConcatDataset(dataset_list, shuffle=shuffle)
    dataloader = DataLoader(
        dataset_cat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Ensure consistent batch size
    )
    return dataloader