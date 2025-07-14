import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# from x_transformers import ContinuousTransformerWrapper, Encoder

from .layers import FourierFeatures
# from .layers import WNConv1d
from .transformer import ContinuousTransformer
from ..utils import exists


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 *,
                 io_channels: int,
                 num_tracks: int = 3, ## (Mix, Submix, Source) Track
                 patch_size: int = 1,
                 embed_dim: int,
                 cross_cond_token_dim: int = 0, ## Cross attention cond, MGE-LDM do not support cross attention so far
                 project_cross_cond_tokens: bool = True, ## valid if cross cond_token_dim > 0...
                 global_cond_dim: int, ## Global conditioning by AdaGN / AdaLN
                 global_cond_grouped: bool = True, # If true, the global CLAP conditioning is grouped into {num_tracks} groups
                 input_concat_dim: int = 0, # Not used, set 0
                 # We prepend timestep features to the input sequence for conditioning
                 timestep_features_dim: int = 256,
                 prepend_cond_dim: int = 0, ## addtiional prepend conditions, we do not use.
                 depth: int,
                 num_heads: int,
                 transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "continuous_transformer",
                 norm_type: tp.Literal["layernorm", "groupnorm"] = "groupnorm",
                 use_t_emb_trackwise: bool = True, # If ture, use the track-wise timestep embedding projection 
                 **kwargs
                 ):
        super().__init__()
        assert transformer_type == "continuous_transformer", "Only continuous_transformer is supported now."
        assert patch_size == 1, "Only patch_size=1 is supported now."
        
        self.patch_size = patch_size
        self.cross_cond_token_dim = cross_cond_token_dim
        self.input_concat_dim = input_concat_dim
        self.num_tracks = num_tracks
        self.io_channels = io_channels
        self.use_t_emb_trackwise = use_t_emb_trackwise
        
        dim_in = num_tracks * io_channels + input_concat_dim
        dim_out = num_tracks * io_channels
        
        embed_dim_mult = embed_dim * num_tracks
        
        if use_t_emb_trackwise:
            t_embed_dim = embed_dim
        else:
            t_embed_dim = embed_dim_mult
        
        self.global_cond_grouped = global_cond_grouped
        
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        
        conv_cls = nn.Conv1d
        
        if use_t_emb_trackwise:
            ## input: (B, 3, 256), 3=num_tracks
            ## (B, 3, 256) -> (B, 3, t_embed_dim)
            ## b: batch, n: num_tracks, e: embed_dim
            self.to_timestep_embed = nn.Sequential(
                Rearrange("b n e -> b (n e) 1"),
                conv_cls(
                    timestep_features_dim*num_tracks,
                    t_embed_dim*num_tracks,
                    kernel_size=1,
                    groups=num_tracks,
                    bias=True
                ),
                nn.SiLU(),
                conv_cls(
                    t_embed_dim*num_tracks,
                    t_embed_dim*num_tracks,
                    kernel_size=1,
                    groups=num_tracks,
                    bias=True
                ),
                Rearrange("b ne 1 -> b ne") # (B, 3*t_embed_dim)
            )
        else:
            ## input: (B, 256) -> (B, t_embed_dim)
            self.to_timestep_embed = nn.Sequential(
                nn.Linear(timestep_features_dim, t_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(t_embed_dim, t_embed_dim, bias=True)
            )
        
        if cross_cond_token_dim > 0:
            """
            This condition is for conditioning token sequences.
            Not supported in MGE-LDM yet.
            """
            ## Cross attention conditioning tokens
            raise NotImplementedError("Cross attention conditioning is not supported in MGE-LDM yet.")
            cross_cond_embed_dim = cross_cond_token_dim if not project_cross_cond_tokens \
                else embed_dim_mult
            self.to_cross_cond_embed = nn.Sequential(
                nn.Linear(cross_cond_token_dim, cross_cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cross_cond_embed_dim, cross_cond_embed_dim, bias=False)
            )
        else:
            cross_cond_embed_dim = 0
        
        if global_cond_dim > 0:
            ## Global conditioning, CLAP.
            ## project_global_cond is True
            ## input cond: (B, 3, clap_emb)
            # global_embed_dim = embed_dim_mult
            if global_cond_grouped is False: ## Not grouped : original projection
                ## input: (B, 3, 512), output: (B, 3*512)
                ## global embed_dim: 512 * 3
                ## b: batch, n: num_tracks, e: embed_dim
                self.to_global_embed = nn.Sequential(
                    Rearrange('b n e -> b (n e)'),
                    nn.Linear(global_cond_dim*num_tracks, global_cond_dim*num_tracks, bias=False),
                    nn.SiLU(),
                    nn.Linear(global_cond_dim*num_tracks, embed_dim_mult, bias=False)
                )
            else: ## Grouped: project each group separately
                ## input: (B, 3, 512), output: (B, 3*512)
                self.to_global_embed = nn.Sequential(
                    Rearrange('b n e -> b (n e) 1'),
                    conv_cls(
                        global_cond_dim*num_tracks,
                        global_cond_dim*num_tracks,
                        kernel_size=1,
                        groups=num_tracks,
                        bias=False
                    ),
                    nn.SiLU(),
                    conv_cls(
                        global_cond_dim*num_tracks,
                        embed_dim_mult,
                        kernel_size=1,
                        groups=num_tracks,
                        bias=False
                    ),
                    Rearrange('b ne 1 -> b ne') # (B, 3*embed_dim)
                )
        else:
            pass
        
        if prepend_cond_dim > 0:
            ## prepend condition, other than timestep features.
            raise NotImplementedError("Prepend condition is not supported in MGE-LDM yet.")
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim_mult, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim_mult, embed_dim_mult, bias=False)
            )
        
        # global_dim = embed_dim_mult
        ## Define Transformer Layers
        norm_kwargs = {"num_groups": num_tracks} if norm_type == "groupnorm" else {}
        
        self.transformer = ContinuousTransformer(
            dim=embed_dim_mult,
            depth=depth,
            dim_heads=embed_dim_mult // num_heads,
            dim_in=dim_in * patch_size,
            dim_out=dim_out * patch_size,
            cross_attend=cross_cond_token_dim > 0,
            cond_token_dim=cross_cond_embed_dim,
            global_cond_dim=embed_dim_mult,
            norm_type=norm_type,
            global_cond_group=num_tracks if global_cond_grouped else 1,
            norm_kwargs=norm_kwargs,
            **kwargs
        )
        
        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(dim_out, dim_out, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)
    
    
    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor, 
        mask: tp.Optional[torch.Tensor] = None,
        cross_attn_cond: tp.Optional[torch.Tensor] = None,
        cross_attn_cond_mask: tp.Optional[torch.Tensor] = None,
        input_concat_cond: tp.Optional[torch.Tensor] = None,
        global_embed: tp.Optional[torch.Tensor] = None,
        prepend_cond: tp.Optional[torch.Tensor] = None,
        prepend_cond_mask: tp.Optional[torch.Tensor] = None,
        return_info: bool = False,
        **kwargs
    ):
        ## MGE-LDM does not support following conditions:
        assert prepend_cond == None ## Prepend condition is only for time embedding
        assert prepend_cond_mask == None
        assert cross_attn_cond == None ## cross attention is not used.
        assert cross_attn_cond_mask == None 
        assert input_concat_cond == None 
        
        """
        x: (B, 3C, T)
        CLAP conditioning : global embedding => AdaLN
            single emb = (512, ) embedding
            embs = [mix_emb, submix_emb, src_emb] : (3, 512)
            batch_embs = (B, 3, 512)
            
        Time conditionin: prepend
        t: (B, ) or (B, 3)
        ## (B, ) => same t for all tracks
        ## (B, 3) => different t for all tracks
        ## (B, 3, T) => TODO: different t for all tracks, all frames.
        """
        
        if exists(cross_attn_cond):
            # raise NotImplementedError("Cross attention is not supported in MGE-LDM yet.")
            cross_attn_cond = self.to_cross_cond_embed(cross_attn_cond)
        
        if exists(global_embed):
            ## Project the global conditioning to the embedding dimension
            ## global_embed: (B, 3, 512) -> (B, 3*512)
            global_embed = self.to_global_embed(global_embed)
        else:
            raise NotImplementedError("Global conditioning (CLAP) should be provided in MGE-LDM")
        
        if exists(prepend_cond):
            raise NotImplementedError("Prepend condition is not supported in MGE-LDM yet.")
            prepend_cond = self.to_prepend_embed(prepend_cond)
            if exists(prepend_cond_mask):
                prepend_mask = prepend_cond_mask
        
        if exists(input_concat_cond):
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')
            x = torch.cat([x, input_concat_cond], dim=1)
        
        if t.dim() == 1:
            if self.use_t_emb_trackwise:
                t = t.unsqueeze(1).repeat(1, self.num_tracks) # (B, 3)
                timestep_feat = self.timestep_features(t[:, :, None]) # (B, 3, 256)
                timestep_embed = self.to_timestep_embed(timestep_feat) # (B, 3*t_embed_dim)
            else:
                timestep_feat = self.timestep_features(t[:, None]) # (B, 256)
                timestep_embed = self.to_timestep_embed(timestep_feat) # (B, t_embed_dim)
        elif t.dim() == 2:
            ## t: (B, 3)
            assert self.use_t_emb_trackwise is True
            timestep_feat = self.timestep_features(t[:, :, None]) # (B, 3, 256)
            timestep_embed = self.to_timestep_embed(timestep_feat) # (B, 3*t_embed_dim)
            
        ## Prepend timestep features to the input sequence
        prepend_inputs = timestep_embed.unsqueeze(1) # (B, 1, 3*t_embed_dim) = (B, 1, embed_dim)
        prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
        prepend_length = 1
        
        
        ## Process
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")  # (B, T, 3C)
        
        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        
        info = None
        output = self.transformer(
            x, 
            mask=mask,
            prepend_embeds=prepend_inputs,
            prepend_mask=prepend_mask,
            global_cond=global_embed,
            return_info=return_info,
            context=cross_attn_cond,
            context_mask=cross_attn_cond_mask,
            **kwargs
        )
        if return_info:
            output, info = output
            
        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]  
        
        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)
            
        output = self.postprocess_conv(output) + output 
        
        return (output, info) if return_info else output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None, ## x
        cross_attn_cond_mask=None, ## x
        negative_cross_attn_cond=None, ## x
        negative_cross_attn_mask=None, ## x
        input_concat_cond=None, ## x
        global_embed=None, ## The only condition we use in MGE-LDM
        prepend_cond=None, ## x
        prepend_cond_mask=None, ## x
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False, ## x
        scale_phi=0.0,
        mask=None, ## x
        return_info=False,
        **kwargs
    ):
        """
        MGE-LDM setting
        """
        assert cross_attn_cond is None
        assert cross_attn_cond_mask is None
        assert negative_cross_attn_cond is None
        assert negative_cross_attn_mask is None
        assert input_concat_cond is None
        assert prepend_cond is None
        assert prepend_cond_mask is None
        assert causal is False ## default setting for DiffusionTransformer
        assert mask is None
        
        assert exists(global_embed)
        
        """
        please refer to:
        https://github.com/yukara-ikemiya/friendly-stable-audio-tools/blob/8bcff7280ad372e99d79b18c5b395a5b6d9c5284/stable_audio_tools/models/dit.py#L258
        """
        if self.training and cfg_dropout_prob > 0.0:
            if exists(cross_attn_cond):
                raise NotImplementedError
            if exists(prepend_cond):
                raise NotImplementedError
            if exists(global_embed):
                # global_embed: (B, 3, 512)
                null_embed = torch.zeros_like(global_embed, device=global_embed.device)
                batch_size, num_tracks, clap_dim = global_embed.shape
                dropout_mask = torch.bernoulli(
                    torch.full((batch_size, num_tracks, 1), cfg_dropout_prob, device=global_embed.device)).to(torch.bool)
                global_embed = torch.where(dropout_mask, null_embed, global_embed)
                
        if cfg_scale != 1.0 and exists(global_embed):
            ## Classifier-free guidance
            ## Concatenate conditioned and unconditioned inputs on the batch dimension
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)
            
            if exists(input_concat_cond):
                raise NotImplementedError
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None
            
            if exists(cross_attn_cond):
                raise NotImplementedError
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if exists(negative_cross_attn_cond):
                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if exists(negative_cross_attn_mask):
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)

                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if exists(cross_attn_cond_mask):
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
            else:
                batch_cond = None
                batch_cond_masks = None
            
            if exists(prepend_cond):
                raise NotImplementedError
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)

                if exists(prepend_cond_mask):
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
            else:
                batch_prepend_cond = None
                batch_prepend_cond_mask = None
            
            ## CLAP condition
            if exists(global_embed):
                null_embed = torch.zeros_like(global_embed, device=global_embed.device)
                batch_global_cond = torch.cat([global_embed, null_embed], dim=0)
            
            if exists(mask):
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
                
            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                input_concat_cond=batch_input_concat_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
                prepend_cond_mask=batch_prepend_cond_mask,
                mask=batch_masks,
                return_info=return_info,
                **kwargs
            )
            
            if return_info:
                batch_output, info = batch_output
            
            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + cfg_scale * (cond_output - uncond_output)
            
            # CFG rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std / out_cfg_std)) + (1 - scale_phi) * cfg_output
            else:
                output = cfg_output
            
            return (output, info) if return_info else output

        else: ## No CFG
            return self._forward(
                x,
                t, 
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )

        
        