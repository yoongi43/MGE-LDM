import os; opj = os.path.join
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from multi_track_stable_audio.models.factory import create_mgeldm_from_config
from multi_track_stable_audio.utils import load_ckpt_state_dict, to_numpy

from multi_track_stable_audio.inference.task_wrapper import InferenceTaskWrapper

import warnings
import torch
import torchaudio
import soundfile as sf
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
warnings.filterwarnings("ignore")

# from sweetdebug import sweetdebug; sweetdebug()


# def generate_filename(output_dir):
#     """
#     Generate a unique filename for the output audio file.
#     """
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)
#     # return opj(output_dir, f"output_{len(os.listdir(output_dir)) + 1}.wav")
#     return f"output_{len(os.listdir(output_dir)) + 1}"

def save_spectrogram(wav: np.ndarray, 
                     sample_rate: int,
                     output_path):
    if wav.ndim > 1:
        sig = wav.mean(axis=0)  # Average across channels if multi-channel
    else:
        sig = wav
    D = librosa.stft(sig, n_fft=1024, hop_length=256, win_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sample_rate, x_axis='time', y_axis='linear', cmap='magma', ax=ax,
        hop_length=256,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    


def proc_audio(wav, downsample_ratio):
    assert wav.dim() == 2, "Expected wav to be a 2D tensor (channels, time)."
    wav = wav.mean(dim=0, keepdim=True)  # Average across channels
    ## wav len must be divisible by downsample_ratio
    wav_len = wav.shape[-1]
    if wav_len % downsample_ratio != 0:
        new_len = (wav_len // downsample_ratio) * downsample_ratio
        wav = wav[:, :new_len]
    assert wav.shape[-1] % downsample_ratio == 0, "Wav length must be divisible by downsample_ratio."
    return wav


@hydra.main(version_base=None, config_path="configs", config_name="default_dit")
def inference(config):
    """
    We need:
    config.ckpt_path: unwrapped DiT checkpoint path
    config.task: Task to run (e.g., 'generate', 'extract', 'partial_generation', 'accompaniment_generation')
    
    config.given_wav_path: Path to the input audio file for inference, can be None for total generation
    config.text_prompt: Text prompt for the model, can be None for unconditional generation
    config.output_dir: Directory to save the output files
    
    ## For generaiton,
        config.gen_audio_dur: Duration of the audio to generate (in seconds)
    
    config.num_steps: Number of diffusion steps
    config.cfg_scale: CFG scale for inference
    config.overlap_dur: Overlap duration for outpainting (in seconds)
    config.repaint_n: RePaint steps for continuous generation (i.e., for outpainting given overlapped region.)
    """
    print(f"Running task: {config.task}")
    
    model = create_mgeldm_from_config(config.model)
    # ckpt = load_ckpt_state_dict(config.ckpt_path, adjust_key_names=True)
    ckpt = load_ckpt_state_dict(config.ckpt_path)
    model.load_state_dict(ckpt, strict=True)
    print(f"Loaded model from {config.ckpt_path}")
    
    
    task_wrapper = InferenceTaskWrapper(
        model=model,
        segment_length_trained=config.model.segment_length,
        timestep_eps=config.model.timestep_eps,
        clap_ckpt_path=None, ## MGE-LDM already has CLAP text conditioner.
        load_clap_audio=False, ## Audio conditioning is not implemented yet
        device=torch.device(f"cuda:0"),
    )
    # os.makedirs(config.output_dir, exist_ok=True)
    
    func_args = {
        "overlap_dur": config.overlap_dur,
        "cfg_scale": config.cfg_scale,
        "num_timesteps": config.num_steps,
        "repaint_n": config.repaint_n,
        "verbose": True,
    }
    
    if config.task == "total_gen":
        output_dir = opj(config.output_dir, "total_gen")
        
        """
        output: dict(
            "gen_mix": torch.Tensor, (B, 1, T)
            "gen_submix": torch.Tensor, (B, 1, T)
            "gen_src": torch.Tensor, (B, 1, T)
        )
        """
        assert config.given_wav_path is None, "For total generation, given_wav_path should be None."
        text_conds_mix = [config.text_prompt]
        
        output = \
            task_wrapper.total_mixture_generation(
                text_conds_mix=text_conds_mix, # only generated single output if length of list = 1
                audio_dur= config.gen_audio_dur,
                return_submix_src=True,
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, verbose
            )
        
        
        num_samples = len(text_conds_mix)
        given_wav = None

                
    elif config.task == "partial_gen":
        """
        task_wrapper.partial_generation supports batch processing,
        but this code is designed for single input generation.
        """
        output_dir = opj(config.output_dir, "partial_gen_single")
        
        assert config.given_wav_path is not None, "For partial generation, given_wav_path should be provided."
        given_wav, sr_ori = torchaudio.load(config.given_wav_path)
        given_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_wav)
        given_wav = proc_audio(given_wav, downsample_ratio=2048) ## Downsample ratio of the autoencoder
        # given_wav: (1, T) 
        # if given_wav.shape[0] > 1:
        #     print("Warning: Multichannel audio detected. Only the first channel will be used.")
        #     given_wav = torch.mean(given_wav, dim=0, keepdim=True)  # Average across channels
        assert given_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        given_wav = given_wav.unsqueeze(0)  # (1, 1, T) # Temporary code. 
        
        text_conds_src = [config.text_prompt]
        assert given_wav.shape[0] == len(text_conds_src), "The number of given_wav and text_conds_src must be the same."
        
        output = \
            task_wrapper.partial_generation(
                given_wav=given_wav,
                text_conds_src=text_conds_src,  # only generated single output if length of list = 1
                overlap_dur=config.overlap_dur,
                cfg_scale=config.cfg_scale,
                num_timesteps=config.num_steps,
                repaint_n=config.repaint_n,  # RePaint steps for continuous generation.
                verbose=True,
                return_full_output=True,
            )
        num_samples = given_wav.shape[0]

    elif config.task == "source_extract":
        assert config.given_wav_path is not None, "For source extraction, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "src_extract")
        mix_wav, sr_ori = torchaudio.load(config.given_wav_path)
        mix_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(mix_wav)
        mix_wav = proc_audio(mix_wav, downsample_ratio=2048)  # Downsample ratio of the autoencoder
        # given_wav: (1, T)
        assert mix_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        mix_wav = mix_wav.unsqueeze(0)  # (1, 1, T)
        text_conds_src = [config.text_prompt]
        assert mix_wav.shape[0] == len(text_conds_src), "The number of mix_wav and text_conds_src must be the same."
        output = \
            task_wrapper.source_extraction(
                mix_wav=mix_wav,
                text_conds_src=text_conds_src,  # only generated single output if length of list = 1
                return_full_output=True,
                **func_args, 
            )
        num_samples = mix_wav.shape[0]
        given_wav = mix_wav  # Use mix_wav as given_wav for saving
        
    elif config.task == "accomp_gen":
        """
        given_wav must be single-source audio files.
        """
        assert config.given_wav_path is not None, "For accompaniment generation, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "accomp_gen")
        given_src, sr_ori = torchaudio.load(config.given_wav_path)
        given_src = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_src)
        given_src = proc_audio(given_src, downsample_ratio=2048)  # Downsample ratio of the autoencoder
        # given_src: (1, T)
        assert given_src.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        given_src = given_src.unsqueeze(0)  # (1, 1, T) 
        text_conds_submix = [config.text_prompt]
        assert given_src.shape[0] == len(text_conds_submix), "The number of given_src and text_conds_submix must be the same."
        output = \
            task_wrapper.accompaniments_generation(
                given_src=given_src,
                text_conds_submix=text_conds_submix,  # only generated single output if length of list = 1
                return_full_output=True,
                **func_args,  #
            )
        num_samples = len(text_conds_submix)
        given_wav = given_src  # Use given_src as given_wav for saving
        
    elif config.task == "partial_gen_iter":
        """
        task_wrapper.partial_generation_iter does not support batch processing.
        """
        assert config.given_wav_path is not None, "For partial generation iteration, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "partial_gen_iter")
        ordered_text_cond = config.text_prompt.split(";")  # e.g., "text1; text2; text3"
        assert len(ordered_text_cond) >= 2, "For partial generation iteration, at least two text prompts are required."
        
        given_wav, sr_ori = torchaudio.load(config.given_wav_path)
        given_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_wav)
        given_wav = proc_audio(given_wav, downsample_ratio=2048)  # Downsample ratio of the autoencoder
        assert given_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        # given_wav = given_wav.unsqueeze(0)  # (1, 1, T)
        ordered_text_cond_src = [text.strip() for text in ordered_text_cond]
        output = \
            task_wrapper.partial_generation_iterative(
                given_wav=given_wav,
                ordered_text_cond_src=ordered_text_cond_src,  # e.g., ["text1", "text2", "text3"]
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, verbose
            )
        ## output: "generated_mixture", "given_wav", "generated_sources"(list)
        given_wav = output["given_wav"]  # Use given_wav from output for saving
        output_new = {
            "gen_mix": output["generated_mixture"],
        }
        for ii, text_cond in enumerate(ordered_text_cond_src):
            output_new[f"gen_src_{ii}_{text_cond}"] = output["generated_sources"][ii]
        output = output_new
        num_samples = 1
        
    else:
        raise ValueError(f"Unknown task: {config.task}. Supported tasks are 'total_gen', 'partial_gen', 'source_extract', 'accomp_gen', 'partial_gen_iter'.")


    os.makedirs(output_dir, exist_ok=True)
    ## Save the output audio files
    for bb in range(num_samples):
        ## Temporary directory to save the output.
        # os.makedirs(output_dir, exist_ok=True)
        # output_subdir = f"output_{len(os.listdir(output_dir)) + 1:04d}"
        # os.makedirs(opj(output_dir, output_subdir), exist_ok=True)
        pattern = re.compile(r"^output_(\d{4})$")
        existing = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(opj(output_dir, d)) and pattern.match(d)
        ]
        numbers = sorted(int(pattern.match(d).group(1)) for d in existing)
        next_idx = numbers[-1] + 1 if numbers else 1
        output_subdir = f"output_{next_idx:04d}"
        os.makedirs(opj(output_dir, output_subdir), exist_ok=True)
        
        # Save the generated audio files
        for key, tensor in output.items():
            filename = f"{key}.wav"
            filepath = opj(output_dir, output_subdir, filename)
            # Save the tensor as a wav file
            wav = to_numpy(tensor[bb])
            assert wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
            sf.write(filepath, wav[0], config.model.sample_rate)
            print(f"Saved {key} to {filepath}")
            save_spectrogram(wav[0], config.model.sample_rate, filepath.replace('.wav', '.png'))
            
            
        if given_wav is not None:
            if config.task == "source_extract":
                save_path = opj(output_dir, output_subdir, "given_mix.wav")
            else:
                save_path = opj(output_dir, output_subdir, "given_wav.wav")
            # Save the given wav file
            given_wav = to_numpy(given_wav[bb])
            sf.write(save_path, given_wav[0], config.model.sample_rate)
            print(f"Saved given wav to {save_path}")
            save_spectrogram(given_wav[0], config.model.sample_rate, save_path.replace('.wav', '.png'))

    with open(opj(output_dir, output_subdir, "prompt.txt"), "w") as f:
        if config.task == "partial_gen_iter":
            f.write(f"Task: {config.task}\n")
            f.write(f"Given wav: {config.given_wav_path}\n")
            f.write(f"Ordered text conditions: {config.text_prompt}\n")
        else:
            f.write(f"Task: {config.task}\n")
            f.write(f"Given wav: {config.given_wav_path}\n")
            f.write(f"Text prompt: {config.text_prompt}\n")
        f.write(f"Overlap duration: {config.overlap_dur} seconds\n")
        f.write(f"CFG scale: {config.cfg_scale}\n")
        f.write(f"Number of diffusion steps: {config.num_steps}\n")
        f.write(f"RePaint steps for time-axis inpainting: {config.repaint_n}\n")

if __name__=="__main__":
    inference()