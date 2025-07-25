import math

import torch
from ..utils import print_once, exists
from ..models.diffusion import MGELDM



def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    ##t=0 => alpha=1, sigma=0
    ##t=1 => alpha=0, sigma=1
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)



@torch.no_grad()
def sample_simple(model, x, steps, eta, t_min=None, verbose: bool = True,  **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    if t_min is not None:
        t = torch.linspace(1, t_min, steps)
    else:
        t = torch.linspace(1, 0, steps + 1)[:-1]
    ## ex) t = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    alphas, sigmas = get_alphas_sigmas(t)

    if verbose:
        itv = 50
        t_s = torch.cuda.Event(enable_timing=True)
        t_e = torch.cuda.Event(enable_timing=True)
        t_s.record()

    # The sampling loop
    for i in range(steps):

        # Get the model output (v, the predicted velocity)
        # with torch.cuda.amp.autocast():
        ## t[i]: 현재 timestep에 대한 t값, ts * t[i]: 현재 timestep에 대한 t값을 배치 크기만큼 확장한 텐서
        v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # # If we are not on the last timestep, compute the noisy image for the
        # # next timestep.
        # if i < steps - 1:
        #     # If eta > 0, adjust the scaling factor for the predicted noise
        #     # downward according to the amount of additional noise to add
        #     ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
        #         (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
        #     adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

        #     # Recombine the predicted noise and predicted denoised image in the
        #     # correct proportions for the next step
        #     x = pred * alphas[i + 1] + eps * adjusted_sigma

        #     # Add the correct amount of fresh noise
        #     if eta:
        #         x += torch.randn_like(x) * ddim_sigma

        if i < steps - 1:
            if eta > 0:
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()
                x = pred * alphas[i + 1] + eps * adjusted_sigma
                x += torch.randn_like(x) * ddim_sigma
            else:
                x = pred * alphas[i + 1] + eps * sigmas[i + 1]

        if verbose and (i + 1) % itv == 0:
            t_e.record()
            torch.cuda.synchronize()
            proc_time = t_s.elapsed_time(t_e) / 1000.
            print_once(f"{i + 1}\t / {steps}  [{itv / proc_time:.2f} iter/sec]")
            t_s.record()

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def sample_extraction_simple(model, x, steps, eta, verbose:bool = True, t_min=None, **extra_args):
    """ 
    Sample submix and source given mix_0 
    input x has clean mixture in the first C channels.
    """
    bs = x.shape[0]
    # ts = x.new_ones([bs]) ## (B, )

    if t_min is not None:
        t = torch.linspace(1, t_min, steps)
    else:
        t = torch.linspace(1, 0, steps + 1)[:-1]

    ## t: (steps, )    
    alphas, sigmas = get_alphas_sigmas(t) ## t: (steps, )

    if verbose:
        itv = 50
        t_s = torch.cuda.Event(enable_timing=True)
        t_e = torch.cuda.Event(enable_timing=True)
        t_s.record()
    
    # The sampling loop
    for i in range(steps):
        
        # Get the model output (v, the predicted velocity)
        # with torch.cuda.amp.autocast():
        # cur_ts: (B, 3)
        # x: (B, 3C, T)
        # v: (B, 3C, T)
        cur_ts = torch.ones(bs, 3).to(x.device) * t[i]
        cur_ts[:, 0] = 0.0

        v = model(x, cur_ts, **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i] ## first C channels are dummy.
        eps = x * sigmas[i] + v * alphas[i]

        if i < steps - 1:
            if eta > 0:
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()
                x = pred * alphas[i + 1] + eps * adjusted_sigma
                x += torch.randn_like(x) * ddim_sigma
            else:
                x = pred * alphas[i + 1] + eps * sigmas[i + 1]

        if verbose and (i + 1) % itv == 0:
            t_e.record()
            torch.cuda.synchronize()
            proc_time = t_s.elapsed_time(t_e) / 1000.
            print_once(f"{i + 1}\t / {steps}  [{itv / proc_time:.2f} iter/sec]")
            t_s.record()

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def sample(
    model:MGELDM,
    noise, # (B, C, T_2) # 
    overlap_z, # (B, C, T_1) # Can be None
    steps,
    eta,
    t_min=None,
    repaint_n:int=1,
    verbose:bool=True,
    **extra_args
):
    bs = noise.shape[0]
    ts = noise.new_ones([bs])
    
    latent_dim = noise.shape[1] # (B, C, T_2)
    
    if t_min is not None:
        t = torch.linspace(1, t_min, steps)
    else:
        t = torch.linspace(1, 0, steps + 1)[:-1]
    
    ## t: (steps, ), alpha: (steps, ), sigma: (steps, )
    alphas, sigmas = get_alphas_sigmas(t)
    
    if verbose:
        itv = 50
        t_s = torch.cuda.Event(enable_timing=True)
        t_e = torch.cuda.Event(enable_timing=True)
        t_s.record()
    
    if overlap_z is None:
        assert repaint_n == 1, "Repaint is not needed when overlap_z is None."
        x = noise
    else:
        x = torch.cat([torch.randn_like(overlap_z), noise], dim=-1)  # (B, C, T_1 + T_2)
    # noise_full = torch.cat([torch.randn_like(overlap_z), noise], dim=-1)  # (B, C, T_1 + T_2)
    # x = noise_full
    
    for i in range(steps):
        for jj in range(repaint_n):
            # with torch.cuda.amp.autocast():
            # v = model(x, ts * t[i], **extra_args).float()
            v = model.model_dit(x, ts * t[i], **extra_args).float()
            pred = x * alphas[i] - v * sigmas[i]
            eps = x * sigmas[i] + v * alphas[i]
            
            if i < steps - 1:
                if eta > 0:
                    ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                        (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                    adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()
                    x = pred * alphas[i + 1] + eps * adjusted_sigma
                    x += torch.randn_like(x) * ddim_sigma
                else:
                    x = pred * alphas[i + 1] + eps * sigmas[i + 1]
                
                if overlap_z is not None:
                    overlap_noised = overlap_z * alphas[i+1] + torch.randn_like(overlap_z) * sigmas[i+1] 
                    x[:, :, :overlap_z.shape[-1]] = overlap_noised ## Overlap region is noised.
            
            if jj < repaint_n -1 and i < steps - 1:
                ## one-step forward: re-corrupt
                noise_step = torch.randn_like(x)
                alpha_t_sqrt = alphas[i] / alphas[i+1]
                alpha_t = alpha_t_sqrt ** 2
                beta_t = 1 - alpha_t
                beta_t_sqrt = beta_t.sqrt()
                x = x * alpha_t_sqrt + noise_step * beta_t_sqrt
            
        if verbose and (i+1) % itv == 0:
            t_e.record()
            torch.cuda.synchronize()
            proc_time = t_s.elapsed_time(t_e) / 1000.
            print_once(f"{i + 1}\t / {steps}  [{itv / proc_time:.2f} iter/sec]")
            t_s.record()
        
    # return pred[:, :, overlap_z.shape[-1]:]  # Return the generated part only (T_2)
    return pred
                
    
    
@torch.no_grad()
def sample_given_track(
    model,
    noise, # (B, 3C, T_2) # latent.
    overlap_z, # (B, 3C, T_1) # Can be None. Generated latent of the previous chunk.
    given_z, # (B, C, T_2)
    given_track_idx: int,
    steps,
    eta,
    repaint_n: int = 1, ## RePaint for outpaint regions
    t_min=None,
    verbose: bool = True,
    **extra_args
):
    """
    Sample remaining tracks given the specified track.
    - given_track_idx: Index of the track to be used as a condition.
    """
    assert given_track_idx in [0, 1, 2]
    # bs = x.shape[0]
    bs = noise.shape[0]
    latent_dim = given_z.shape[1]
    # assert x.shape[1] == 3 * latent_dim
    assert noise.shape[1] == 3 * latent_dim
    
    ## Set schedules
    if t_min is not None:
        t = torch.linspace(1, t_min, steps)
    else:
        t = torch.linspace(1, 0, steps + 1)[:-1]
        
    alphas, sigmas = get_alphas_sigmas(t)
    
    if verbose:
        itv = 50
        t_s = torch.cuda.Event(enable_timing=True)
        t_e = torch.cuda.Event(enable_timing=True)
        t_s.record()
    
    tidx1 = given_track_idx*latent_dim
    tidx2 = (given_track_idx+1)*latent_dim
    
    if overlap_z is None:
        # assert repaint_n == 1, "Repaint is not needed when overlap_z is None."
        ## First segment can be overlap_z == None
        x = noise
    else:
        x = torch.cat([torch.randn_like(overlap_z), noise], dim=-1)  # (B, 3C, T_1 + T_2)
        t_over = overlap_z.shape[-1]
        
    for ii in range(steps):
        for jj in range(repaint_n):
            cur_ts = torch.ones(bs, 3).to(x.device) * t[ii]
            cur_ts[:, given_track_idx] = 0.0
            # import pdb; pdb.set_trace()
            if overlap_z is None:
                x[:, tidx1:tidx2, :] = given_z
            else:
                x[:, tidx1:tidx2, :t_over] = overlap_z[:, tidx1:tidx2, :]
                x[:, tidx1:tidx2, t_over:] = given_z
            # x[:, given_track_idx*latent_dim:(given_track_idx+1)*latent_dim, :] = given_z
            # with torch.cuda.amp.autocast():
            ## cur_ts: (B, 3)
            ## x: (B, 3C, T)
            ## v: (B, 3C, T)
            # v = model(x, cur_ts, **extra_args).float()
            v = model.model_dit(x, cur_ts, **extra_args).float()
            
            pred = x * alphas[ii] - v * sigmas[ii] ## ex) If given track idx=1, then C:2C is the dummy region.
            eps = x * sigmas[ii] + v * alphas[ii]
            
            if ii < steps - 1:
                if eta > 0:
                    ddim_sigma = eta * (sigmas[ii + 1]**2 / sigmas[ii]**2).sqrt() * \
                        (1 - alphas[ii]**2 / alphas[ii + 1]**2).sqrt()
                    adjusted_sigma = (sigmas[ii + 1]**2 - ddim_sigma**2).sqrt()
                    x = pred * alphas[ii + 1] + eps * adjusted_sigma
                    x += torch.randn_like(x) * ddim_sigma
                else:
                    x = pred * alphas[ii + 1] + eps * sigmas[ii + 1]
                
                if overlap_z is not None:
                    overlap_noised = overlap_z * alphas[ii+1] + torch.randn_like(overlap_z) * sigmas[ii+1]
                    x[:, :, :overlap_z.shape[-1]] = overlap_noised
            
            ## RePaint for time-axis outpainting        
            if jj < repaint_n - 1 and ii < steps - 1:
                ## one-step forward: re-corrupt
                noise_step = torch.randn_like(x)
                alpha_t_sqrt = alphas[ii] / alphas[ii+1]
                alpha_t = alpha_t_sqrt ** 2
                beta_t = 1 - alpha_t
                beta_t_sqrt = beta_t.sqrt()
                x = x * alpha_t_sqrt + noise_step * beta_t_sqrt
                
        if verbose and (ii + 1) % itv == 0:
            t_e.record()
            torch.cuda.synchronize()
            proc_time = t_s.elapsed_time(t_e) / 1000.
            print_once(f"{ii + 1}\t / {steps}  [{itv / proc_time:.2f} iter/sec]")
            t_s.record()
    
    # return pred[:, :, overlap_z.shape[-1]:]  # Return the generated part only (T_2)
    if overlap_z is None:
        pred[:, tidx1:tidx2, :] = given_z
    else:
        pred[:, tidx1:tidx2, :t_over] = overlap_z[:, tidx1:tidx2, :]
        pred[:, tidx1:tidx2, t_over:] = given_z
    return pred
            