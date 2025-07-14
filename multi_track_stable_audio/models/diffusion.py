import torch
import torch.nn as nn

from .dit import DiffusionTransformer
from .conditioners import MultiConditioner
from .pretransforms import Pretransform
# from ..inference.generation import generate_diffusion_cond

import typing as tp

class MGELDM(nn.Module):
    # def __init__(self, *args, **kwargs):
    def __init__(
        self,
        ## DiffusionTransformer Configs
        io_channels:int, # 64
        num_tracks:int, # 3
        embed_dim: int,
        global_cond_dim: int,
        global_cond_grouped: bool, # True
        timestep_features_dim: int, # 256
        depth: int,
        num_heads: int, 
        norm_type: tp.Literal["layernorm", "groupnorm"], 
        use_t_emb_trackwise: bool, # True
        ## MGE-LDM Configs
        conditioner: MultiConditioner,
        sample_rate: int,
        # min_input_length: int,
        diffusion_objective: tp.Literal["v", "rectified_flow"], ## Implmented only "v" for now
        pretransform: tp.Optional[Pretransform],
        global_cond_keys: tp.List[str], # ["audio_cond", "prompt_cond"]
        **kwargs
    ):
        super().__init__()
        self.model_dit = DiffusionTransformer(
            io_channels=io_channels,
            num_tracks=num_tracks,
            patch_size=1,
            embed_dim=embed_dim,
            cross_cond_token_dim=0,
            project_cross_cond_tokens=False, # dummy
            global_cond_dim=global_cond_dim,
            global_cond_grouped=global_cond_grouped,
            input_concat_dim=0,
            timestep_features_dim=timestep_features_dim,
            prepend_cond_dim=0,
            depth=depth,
            num_heads=num_heads,
            transformer_type="continuous_transformer",
            norm_type=norm_type,
            use_t_emb_trackwise=use_t_emb_trackwise,
            **kwargs
        )
        
        with torch.no_grad():
            for param in self.model_dit.parameters():
                param *= 0.5
                
        ## Load conditioner, 
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.global_cond_keys = global_cond_keys## ["audio_cond", "prompt_cond"]
        # self.min_input_length = min_input_length
        self.num_tracks = num_tracks
        
    def get_conditioning_inputs(
        self,
        conditioning_tensors: tp.Dict[str, tp.Any],
        specify_global_key: str = None,
        mask_tracks: tp.List[int] = []
    ):
        """
        conditioning_tensors:
        {
            "audio_cond": [(B, 1, 512), (B, 1, 512), (B, 1, 512)],
            "prompt_cond": [None, (B, 1, 512), (B, 1, 512)],
            or 
            "prompt_cond": [None, None, None]
            or 
            ...
        }
        At training, we only use prompt condition in source track, with linear interpolation between audio_cond.
        At inference, we use prompt condition only.
        
        mask_tracks: 
            List of indices to mask in the global conditioning input.
            
        specify_global_key:
            If not none, use this key to specify the global conditioning input.
            
        * null condition is zero
        """
                
        if len(self.global_cond_keys) > 0:
            ## ex) ["audio_cond"]
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds_dict = {}
            if specify_global_key:
                mix_emb, submix_emb, src_emb = conditioning_tensors[specify_global_key]
                if mix_emb is None:
                    mix_emb = torch.zeros_like(src_emb)
                if submix_emb is None:
                    submix_emb = torch.zeros_like(src_emb)
                global_conds_dict[specify_global_key] = (mix_emb, submix_emb, src_emb)
            else:
                for key in self.global_cond_keys: ##["audio_cond", "prompt"]
                    mix_emb, submix_emb, src_emb = conditioning_tensors[key]
                    global_conds_dict[key] = (mix_emb, submix_emb, src_emb)
                # => global_conds = [(B,3,512), (B,3,512)] if len(global_cond_ids) == 2 w/ text encoder

            ### Concatenate over the channel dimension
            # global_cond = torch.cat(global_conds, dim=-1)
            ## hard_coded: 
            num_cond = len(global_conds_dict)
            assert num_cond > 0, "No global conditioning inputs found."
            
            if num_cond == 1: ## "audio_cond" or "prompt_cond" only
                global_cond_tup = list(global_conds_dict.values())[0]
                global_cond = torch.cat(global_cond_tup, dim=1) # (B, 3, 512)
            elif num_cond == 2: ## "audio_cond" and "prompt_cond"
                mix_audemb, submix_audemb, src_audemb = global_conds_dict["audio_cond"]
                mix_textemb, submix_textemb, src_textemb = global_conds_dict["prompt_cond"]
                if src_textemb is not None:
                    comb_weights = torch.rand(num_cond) + 1e-6
                    comb_weights /= comb_weights.sum()
                    src_comb_emb = comb_weights[0] * src_audemb + comb_weights[1] * src_textemb
                else:
                    src_comb_emb = src_audemb
                global_cond = torch.cat([mix_audemb, submix_audemb, src_comb_emb], dim=1) # (B, 3, 512)
            else:
                raise ValueError(f"Unexpected number of global conditioning inputs: {num_cond}. Expected 1 or 2.")
                
            ## global_cond = (B, 3, 512)
            if len(mask_tracks) > 0:
                ## Null condition is zero
                global_cond[:, mask_tracks] = 0.0
        else:
            raise NotImplementedError("Global conditioning keys are not defined.")
            global_cond = None
          
          
        return {
            "cross_attn_cond": None,
            "cross_attn_cond_mask": None,
            "global_embed": global_cond, # (B, 3, 512)
            "input_concat_cond": None,
            "prepend_cond": None,
            "prepend_cond_mask": None
        }

    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: tp.Dict[str, tp.Any],
        **kwargs
    ):
        return self.model_dit(
            x=x,
            t=t,
            **self.get_conditioning_inputs(cond),
            **kwargs
        )
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Use generate_diffusion_cond instead.")
        # return generate_diffusion_cond(self, *args, **kwargs)
    