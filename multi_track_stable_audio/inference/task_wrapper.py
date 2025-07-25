import torch
import torch.nn as nn

import typing as tp
import math

from .sampling import sample, sample_given_track
from ..models.diffusion import MGELDM
from ..models.conditioners import CLAPTextConditioner, CLAPAudioConditioner
from ..utils import to_numpy
from tqdm import tqdm


class InferenceTaskWrapper:
    def __init__(
        self,
        model: MGELDM,
        segment_length_trained: int=80, 
        timestep_eps: float=0.02,
        clap_ckpt_path: str=None, 
        load_clap_audio: bool=False,
        device: torch.device=torch.device("cpu"),
    ):
        """
        forward: num_timesteps, repaint_n, audio_length,
        TODO: Implement audio conditioning
        """
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        
        self.timestep_eps = timestep_eps
        self.device = device
        
        self.downsample_ratio = model.pretransform.downsampling_ratio
        self.dim_in = model.num_tracks * model.io_channels ## 3 * 64
        self.sr = model.sample_rate
        
        self.frame_per_dur = self.sr / self.downsample_ratio # 16000 / 2048 = 7.8125
        self.dur_per_frame = self.downsample_ratio / self.sr # 2048 / 16000 = 0.128
        self.seg_len_model = segment_length_trained # 80 frames corresponds to 10.24 seconds
        
        ## conditioners
        try:
            self.text_conditioner = model.conditioner.conditioners["prompt_cond"]
        except KeyError:
            """
            Manually initialize CLAPTextConditioner
            """
            # clap_ckpt_path = "/data2/yoongi/dataset/pre_trained/music_audioset_epoch_15_esc_90.14.pt"
            self.text_conditioner = CLAPTextConditioner(
                output_dim=512,
                clap_ckpt_path=clap_ckpt_path,
                use_text_features=False,
                feature_layer_ix=None,
                audio_model_type="HTSAT-base",
                enable_fusion=False,
                project_out=False,
                finetune=False,
            ).to(self.device)
        
        if load_clap_audio:
            try:
                self.audio_conditioner = model.conditioner.conditioners["audio_cond"]
            except KeyError:
                """
                Manually initialize CLAPAudioConditioner
                """
                # clap_ckpt_path = "/data2/yoongi/dataset/pre_trained/music_audioset_epoch_15_esc_90.14.pt"
                self.audio_conditioner = CLAPAudioConditioner(
                    output_dim=512,
                    clap_ckpt_path=clap_ckpt_path,
                    audio_model_type="HTSAT-base",
                    enable_fusion=False,
                    project_out=False,
                    finetune=False,
                )
                
        assert self.model.diffusion_objective == "v", \
            "Currently, only v-objective is supported in MGE-LDM."
        
    def _get_emb_from_cond(
        self,
        conditions: tp.Union[tp.List, torch.Tensor],
        branch="text",
    ):
        num_cond = len(conditions)
        conditioning_list = []
        if branch == "text":
            for bb in range(num_cond):
                if conditions[bb] is not None:
                    # with torch.cuda.amp.autocast():
                    cond_inputs, mask_txt = self.text_conditioner([conditions[bb]])
                    conditioning_list.append(cond_inputs.float())
                else:
                    conditioning_list.append(torch.zeros((1, 1, 512), device=self.device, dtype=torch.float32))
        elif branch == "audio":
            raise NotImplementedError("Audio conditioning is not implemented yet.")
        else:
            raise ValueError(f"Unknown branch: {branch}. Choose from 'text' or 'audio'.")
        
        conditioning = torch.cat(conditioning_list, dim=0) # (B, 1, 512)
        return conditioning
    
    @torch.no_grad()
    def total_mixture_generation(
        self,
        text_conds_mix: tp.List[str],
        audio_dur: float=10.24, # in seconds
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
        return_submix_src: bool=False,
    ):
        """
        Generate a mixture track from text conditions.
        To generate a mixture longer than 10.24 seconds (T=80 in latent space), 
        We use repainting strategy for outpainting so far.
        (We didn't train adaptive timestep conditioning for time-axis so far.)
        (We'll implement this in the future.)
        
        By default, 
        MGE-LDM is trained on 10.24 seconds of audio, 
        which corresponds to 80 timeframes in the latent space.
        (sr = 16000, downsampling_ratio = 2048)
        """
        bs = len(text_conds_mix)
        conditioning = self._get_emb_from_cond(text_conds_mix, branch="text")
        global_cond = torch.zeros([bs, 3, 512], device=self.device, dtype=torch.float32)
        global_cond[:, 0:1, :] = conditioning ## condition on mixture track
        
        
        total_frame = math.ceil(audio_dur * self.frame_per_dur)  # Total frames to generate
        overlap_frame = math.ceil(overlap_dur * self.frame_per_dur)
        assert total_frame > overlap_frame

        if total_frame <= self.seg_len_model:
            noise = torch.randn([bs, self.dim_in, self.seg_len_model], device=self.device, dtype=torch.float32)
            latent = \
                sample(
                    model=self.model,
                    noise=noise,
                    overlap_z=None,
                    steps=num_timesteps,
                    eta=0.0,
                    t_min=self.timestep_eps,
                    verbose=verbose,
                    global_embed=global_cond,
                    cfg_scale=cfg_scale,
                )
        else:
            ## sliding window for outpainting (with RePainting strategy)
            n_segs = math.ceil((total_frame - self.seg_len_model) / (self.seg_len_model - overlap_frame)) + 1
            latents = []
            pbar = tqdm(range(n_segs), desc="Generating mixture track")
            for ii in pbar:
                if ii == 0:
                    ## First segment
                    noise = torch.randn([bs, self.dim_in, self.seg_len_model], device=self.device, dtype=torch.float32)
                    latent = \
                        sample(
                            model=self.model,
                            noise=noise,
                            overlap_z=None,
                            steps=num_timesteps,
                            eta=0.0,
                            repaint_n=1,
                            t_min=self.timestep_eps,
                            verbose=verbose,
                            global_embed=global_cond,
                            cfg_scale=cfg_scale,
                        )
                else:
                    ## 
                    prev_z = latents[-1][:, :, -overlap_frame:]   # (B, C, overlap_frame)
                    noise = torch.randn([bs, self.dim_in, self.seg_len_model - overlap_frame], device=self.device, dtype=torch.float32)
                    latent = \
                        sample(
                            model=self.model,
                            noise=noise,
                            overlap_z=prev_z,
                            steps=num_timesteps,
                            eta=0.0,
                            repaint_n=repaint_n,
                            t_min=self.timestep_eps,
                            verbose=verbose,
                            global_embed=global_cond,
                            cfg_scale=cfg_scale,
                        )
                    latent = latent[:, :, overlap_frame:]  # Remove the overlap part
                    
                latents.append(latent)
            
            latent = torch.cat(latents, dim=-1)[:, :, :total_frame]
            
        mix_z, submix_z, src_z = latent.chunk(3, dim=1)
        
        mix_wav = self.model.pretransform.decode(mix_z)
        item = {"gen_mix": mix_wav}
        if return_submix_src:
            submix_wav = self.model.pretransform.decode(submix_z)
            src_wav = self.model.pretransform.decode(src_z)
            item["gen_submix"] = submix_wav
            item["gen_src"] = src_wav
        return item
    
    def partial_generation(
        self,
        given_wav: torch.Tensor, # (B, 1, T_audio)
        text_conds_src: tp.List[str],
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
        return_full_output: bool=False, ## If True, return full output.
    ):
        given_wav = given_wav.to(self.device)
        
        return self.generate_given_track(
            given_wav=given_wav,
            given_track_idx=1, # 1: submix => partial generation (source imputation)
            text_conds=text_conds_src,
            overlap_dur=overlap_dur,
            cfg_scale=cfg_scale,
            num_timesteps=num_timesteps,
            repaint_n=repaint_n,
            verbose=verbose,
            return_full_output=return_full_output,
        )
        
    def partial_generation_iterative(
        self,
        given_wav: torch.Tensor, # (1, T_audio)
        ordered_text_cond_src: tp.List[str], # Model generates iteratively with this ordered text conditions
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
    ):
        """
        Batch process not supported.
        Iteratively generate sources from the given wav and ordered text conditions.
        """
        assert given_wav.dim() == 2
        given_wav = given_wav.to(self.device)
        given_wav = given_wav.unsqueeze(0)
        given_wav_ori = given_wav.clone()  # Keep original wav for final output
        wav_len = given_wav.shape[-1]
        
        gen_src_list = []
        
        for iter_idx, text_prompt in enumerate(ordered_text_cond_src):
            print(f"Iteration {iter_idx+1}/{len(ordered_text_cond_src)}: {text_prompt}")
            output = self.partial_generation(
                given_wav=given_wav,
                text_conds_src=[text_prompt], # Single text condition for this iteration
                overlap_dur=overlap_dur,
                cfg_scale=cfg_scale,
                num_timesteps=num_timesteps,
                repaint_n=repaint_n,
                verbose=verbose,
                return_full_output=False, # Return full output to get submix and source track
            )
            gen_src = output["gen_src"]  # (1, 1, T_audio)
            wav_len = min(wav_len, gen_src.shape[-1])  # Update wav_len to the minimum length
            given_wav = given_wav[:, :, :wav_len] + gen_src[:, :, :wav_len]  # Update given_wav with the generated source
            gen_src_list.append(gen_src[:, :, :wav_len])  # Append the generated source
        
        wav_len = min([gen_src.shape[-1] for gen_src in gen_src_list])
        gen_src_list = [src[:, :, :wav_len] for src in gen_src_list]  # Trim all generated sources to the same length
        mixture = sum(gen_src_list) + given_wav_ori[:, :, :wav_len]  # Sum of all generated sources and original wav
    
        return {
            "generated_mixture": mixture,  # (1, 1, T_audio)
            "given_wav": given_wav_ori[:, :, :wav_len],  # (1, 1, T_audio)
            "generated_sources": gen_src_list,  # List of generated sources for each iteration
        }
    
    def source_extraction(
        self,
        mix_wav: torch.Tensor,
        text_conds_src: tp.List[str],
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
        return_full_output: bool=False, ## If True, return full output.
    ):
        mix_wav = mix_wav.to(self.device)
        return self.generate_given_track(
            given_wav=mix_wav,
            given_track_idx=0, # 0: mix => source extraction
            text_conds=text_conds_src,
            overlap_dur=overlap_dur,
            cfg_scale=cfg_scale,
            num_timesteps=num_timesteps,
            repaint_n=repaint_n,
            verbose=verbose,
            return_full_output=return_full_output,
        )
        
    def accompaniments_generation(
        self,
        given_src: torch.Tensor,
        text_conds_submix: tp.List[str],
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
        return_full_output: bool=False, ## If True, return full output.
    ):
        """
        Generate submix track given source track.
        This task can used for the task such as "accomponiment generation given vocal track", 
        the task such as in the paper
        "SingSong: Generating musical accompaniments from singing", Donahue et al, 2023
        (https://arxiv.org/abs/2301.12662)
        """
        given_src = given_src.to(self.device)
        return self.generate_given_track(
            given_wav=given_src,
            given_track_idx=2, # 2: source track => generate submix
            text_conds=text_conds_submix,
            overlap_dur=overlap_dur,
            cfg_scale=cfg_scale,
            num_timesteps=num_timesteps,
            repaint_n=repaint_n,
            verbose=verbose,
            return_full_output=return_full_output,
    )
                    
                    
    @torch.no_grad()
    def generate_given_track(
        self,
        given_wav: torch.Tensor,
        given_track_idx: int, # 0: mix, 1: submix, 2: source track
        text_conds: tp.List[str],
        overlap_dur: float=3.0, # in seconds
        cfg_scale: float=2.0,
        num_timesteps: int=100,
        repaint_n: int=1, ## Reapinting strategy for outpainting
        verbose: bool=False,
        return_full_output: bool=False, ## If True, return full output.
    ):
        """
        given_wav: (B, 1, T_audio)
        text_conds: List of text conditions. 
        ex) [None, "The sound of guitar", "drums", None, ...]
        """
        assert given_track_idx in [0, 1, 2], \
            f"given_track_idx must be 0 (mix), 1 (submix), or 2 (source track). Got {given_track_idx}."
            
        bs = given_wav.shape[0]
        assert bs == len(text_conds), \
            f"Batch size of given_wav ({bs}) must match the number of text conditions ({len(text_conds)})."
            
        # assert bs == 1, "Currently, only batch size of 1 is supported for given_wav."
            
        conditioning = self._get_emb_from_cond(text_conds, branch="text")
        global_cond = torch.zeros([bs, 3, 512], device=self.device, dtype=torch.float32)
        # global_cond[:, 2:3, :] = conditioning ## condition on source track
        if given_track_idx == 0:
            ## given track = mixture => source extraction
            global_cond[:, 2:3, :] = conditioning ## condition on source track
        elif given_track_idx == 1:
            ## given track = submix => partial generation (source imputation)
            global_cond[:, 2:3, :] = conditioning ## condition on source track
        elif given_track_idx == 2:
            ## given track = source => generate submix
            global_cond[:, 1:2, :] = conditioning ## condition on submix track
        
        ## Get starting point
        given_z = self.model.pretransform.encode(given_wav) # (B, 64, T)
        noise = torch.randn_like(given_z) # (B, 64, T)
        
        total_frame = given_z.shape[-1]
        overlap_frame = math.ceil(overlap_dur * self.frame_per_dur)
        assert total_frame > overlap_frame
        
        if total_frame <= self.seg_len_model:
            noise = torch.randn([bs, self.dim_in, self.seg_len_model], device=self.device, dtype=torch.float32)
            latent = \
                sample_given_track(
                    model=self.model,
                    noise=noise,
                    overlap_z=None,
                    given_z=given_z,
                    given_track_idx=given_track_idx,
                    steps=num_timesteps,
                    eta=0.0,
                    repaint_n=repaint_n,
                    t_min=self.timestep_eps,
                    verbose=verbose,
                    global_embed=global_cond,
                    cfg_scale=cfg_scale,
                )
        else:
            n_segs = math.ceil((total_frame - self.seg_len_model) / (self.seg_len_model - overlap_frame)) + 1
            latents = []
            pbar = tqdm(range(n_segs), desc=f"Generating given track index {given_track_idx}")
            for ii in pbar:
                if ii == 0:
                    ## First segment
                    noise = torch.randn([bs, self.dim_in, self.seg_len_model], device=self.device, dtype=torch.float32)
                    given_z_seg = given_z[:, :, :self.seg_len_model]  # (B, 64, T_seg)
                    latent = \
                        sample_given_track(
                            model=self.model,
                            noise=noise,
                            overlap_z=None,
                            given_z=given_z_seg,
                            given_track_idx=given_track_idx,
                            steps=num_timesteps,
                            eta=0.0,
                            repaint_n=repaint_n,
                            t_min=self.timestep_eps,
                            verbose=verbose,
                            global_embed=global_cond,
                            cfg_scale=cfg_scale,
                        )
                    ## given_track_idx of "generated latent" is same as given_z.
                elif ii == n_segs - 1:
                    ## Last segment: 
                    # prev_last_frame = ii * (self.seg_len_model - overlap_frame)
                    # start = ii * (self.seg_len_model - overlap_frame)
                    start = total_frame - self.seg_len_model
                    overlap_last = ii * (self.seg_len_model - overlap_frame) - start + overlap_frame
                    if overlap_last <= latents[-1].shape[-1]:
                        prev_z = latents[-1][:, :, -overlap_last:]  # (B, 64, overlap_last)
                    else:
                        prev_z1 = latents[-1]
                        prev_z2 = latents[-2][:, :, -(overlap_last - prev_z1.shape[-1]):]
                        prev_z = torch.cat([prev_z2, prev_z1], dim=-1)
                    # prev_z = latents[-1][:, :, -overlap_last:]  # (B, 64, overlap_last)
                    noise = torch.randn([bs, self.dim_in, self.seg_len_model - overlap_last], device=self.device, dtype=torch.float32)
                    given_z_seg = \
                        given_z[:, :, -(self.seg_len_model - overlap_last):]  # (B, 64, T_seg)
                    # import pdb; pdb.set_trace()
                    latent = \
                        sample_given_track(
                            model=self.model,
                            noise=noise,
                            overlap_z=prev_z,
                            given_z=given_z_seg,
                            given_track_idx=given_track_idx,
                            steps=num_timesteps,
                            eta=0.0,
                            repaint_n=repaint_n,
                            t_min=self.timestep_eps,
                            verbose=verbose,
                            global_embed=global_cond,
                            cfg_scale=cfg_scale,
                        )
                    # import pdb; pdb.set_trace()
                    latent = latent[:, :, overlap_last:]  # Remove the overlap part
                    
                else:
                    prev_z = latents[-1][:, :, -overlap_frame:]
                    noise = torch.randn([bs, self.dim_in, self.seg_len_model - overlap_frame], device=self.device, dtype=torch.float32)
                    given_z_seg = \
                        given_z[:, :, ii*(self.seg_len_model-overlap_frame)+overlap_frame:ii*(self.seg_len_model-overlap_frame) + self.seg_len_model]
                        # given_z[:, :, ii * (self.seg_len_model - overlap_frame) :(ii + 1) * (self.seg_len_model - overlap_frame)]
                    # import pdb; pdb.set_trace()
                    latent = \
                        sample_given_track(
                            model=self.model,
                            noise=noise,
                            overlap_z=prev_z,
                            given_z=given_z_seg,
                            given_track_idx=given_track_idx,
                            steps=num_timesteps,
                            eta=0.0,
                            repaint_n=repaint_n,
                            t_min=self.timestep_eps,
                            verbose=verbose,
                            global_embed=global_cond,
                            cfg_scale=cfg_scale,
                        )
                    latent = latent[:, :, overlap_frame:]
                latents.append(latent)
            latent = torch.cat(latents, dim=-1)[:, :, :total_frame]
        # import pdb; pdb.set_trace()
        mix_z, submix_z, src_z = latent.chunk(3, dim=1)
        output_item = {}
        if given_track_idx == 0:
            ## given track = mixture => source extraction
            src_wav = self.model.pretransform.decode(src_z)
            output_item["gen_src"] = src_wav
            if return_full_output:
                submix_wav = self.model.pretransform.decode(submix_z)
                output_item["gen_submix"] = submix_wav
        elif given_track_idx == 1:
            ## given track = submix => partial generation (source imputation)
            src_wav = self.model.pretransform.decode(src_z)
            output_item["gen_src"] = src_wav
            if return_full_output:
                mix_wav = self.model.pretransform.decode(mix_z)
                output_item["gen_mix"] = mix_wav
                output_item["gen_mix_sum"] = given_wav + src_wav
        elif given_track_idx == 2:
            """
            Source given, generate submix.
            This task can used for the task such as "accomponiment generation given vocal track", 
            the task such as in the paper
            "SingSong: Generating musical accompaniments from singing", Donahue et al, 2023
            (https://arxiv.org/abs/2301.12662)
            """
            submix_wav = self.model.pretransform.decode(submix_z)
            output_item["gen_submix"] = submix_wav
            if return_full_output:
                mix_wav = self.model.pretransform.decode(mix_z)
                output_item["gen_mix"] = mix_wav
        else:
            raise ValueError(f"Unknown given_track_idx: {given_track_idx}. Must be 0, 1, or 2.")
        
        return output_item