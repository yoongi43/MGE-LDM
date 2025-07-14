import hydra
import torch
from functools import partial
import os; opj = os.path.join

from multi_track_stable_audio.models.factory import create_autoencoder_from_config
from multi_track_stable_audio.models.conditioners import CLAPAudioConditioner
from multi_track_stable_audio.utils import load_ckpt_state_dict
from data.pre_encode_latents_utils import (
    extract_zero_latent,
    extract_latents_slakh2100,
    extract_latents_musdb18hq,
    extract_latents_moisesdb,
    extract_latents_medleydbV2,
    extract_latents_mtgjamendo,
    extract_latents_othertracks
)

BASEDIR="/data2/yoongi/dataset/"
DATA_DIRS = {
    "slakh2100": opj(BASEDIR, "slakh2100_inst/slakh2100_flac_redux_wav/"),
    "musdb18hq": opj(BASEDIR, "musdb18hq_from_zenodo/"),
    "moisesdb": opj(BASEDIR, "moisesdb/moisesdb_v0.1/"),
    "medleydbV2": opj(BASEDIR, "medleydb/"),
    "mtgjamendo": opj(BASEDIR, "mtg_jamendo/"),
    "other_tracks": opj(BASEDIR, "other_tracks/"),
}

@hydra.main(version_base=None, config_path="configs", config_name="default_ae")
def pre_encode(config):
    
    model_ae = create_autoencoder_from_config(config.model)
    model_ae.load_state_dict(
        load_ckpt_state_dict(config.ckpt_path), strict=True
    )
    
    clap_audio_kwargs= {
    "output_dim": 512,
    "clap_ckpt_path": config.clap_ckpt_path,
    "audio_model_type": "HTSAT-base",
    "enable_fusion": False,
    "project_out": False,
    "finetune": False
    }
    SR_CLAP = 48000
    SR_AE = config.model.sample_rate
    device = torch.device(f"cuda:0")
    model_clap = CLAPAudioConditioner(**clap_audio_kwargs)
    
    save_root_dir = config.save_root_dir
    ## Extract latents 
    extract_zero_latent(
        save_root_dir=save_root_dir,
        model_ae=model_ae,
        sample_rate_ae=SR_AE,
        device=device,
    )
    
    extract_func_dict = {
        "slakh2100": extract_latents_slakh2100,
        "musdb18hq": extract_latents_musdb18hq,
        "moisesdb": partial(extract_latents_moisesdb, split_seed=42),
        "medleydbV2": extract_latents_medleydbV2,
        "mtgjamendo": extract_latents_mtgjamendo,
        "other_tracks": extract_latents_othertracks
    }
    
    # if config.extract_dataset is None:
    print("Extracting latents for datasets: ", config.extract_dataset)
    # assert False
    for datakey in extract_func_dict.keys():
        if datakey not in config.extract_dataset:
            continue
        func = extract_func_dict[datakey]
        func(
            data_dir=DATA_DIRS[datakey],
            save_root_dir=save_root_dir,
            model_ae=model_ae,
            sample_rate_ae=SR_AE,
            model_clap=model_clap,
            sample_rate_clap=SR_CLAP,
            device=device,
        )
        
        
    # for func in extract_func_list:
    #     func(
    #         save_root_dir=save_root_dir,
    #         model_ae=model_ae,
    #         sample_rate_ae=SR_AE,
    #         model_clap=model_clap,
    #         sample_rate_clap=SR_CLAP,
    #         device=device
    #     )
    
    
if __name__ == "__main__":
    pre_encode()