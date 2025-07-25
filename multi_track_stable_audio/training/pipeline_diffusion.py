import os
import gc
import typing as tp
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from einops import rearrange
import torchaudio
import wandb
from safetensors.torch import save_model
from ema_pytorch import EMA
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from safetensors.torch import save_file

from ..utils import reduce_batch, float_to_int16_audio, print_once
from ..inference.sampling import get_alphas_sigmas
from ..inference.sampling import sample_simple as sample
from ..inference.sampling import sample_extraction_simple as sample_extraction
from ..models.autoencoders import AudioAutoencoder
from ..models.diffusion import MGELDM
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss
from ..models.bottleneck import VAEBottleneck

from .losses import MultiLoss, AuralossLoss, ValueLoss, L1Loss, MSELoss
from .losses.auraloss import SumAndDifferenceSTFTLoss, MultiResolutionSTFTLoss
from .scheduler import create_optimizer_from_config, create_scheduler_from_config
from .logging import MetricsLogger
from .viz import audio_spectrogram_image, tokens_spectrogram_image, pca_point_cloud
from time import time
from copy import deepcopy

class MGELDM_TrainingWrapper(pl.LightningModule):
    """
    MGE-LDM training wrapper for diffusion models.
    """
    def __init__(
        self,
        model: MGELDM,
        use_ema: bool = True,
        log_loss_info: bool = False,
        optimizer_configs: dict = None,
        pre_encoded: bool = True, 
        cfg_dropout_prob: float = 0.1,
        timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform", ## Unifrom
        timestep_dropout_prob: float=0.5, 
        timestep_eps: float=0.02, 
        # mask_loss_track: int=None, # 0: mix, 1: submix, 2: src
        mixture_only_pretraining: bool = False,
        logging_config: dict = {} ## logging={"log_every": 20} from config
    ):
        """
        timestep_dropout_prob
            - If the probability is hit, the following happens:
            - Randomly select one of the 3 tracks (uniformly)
            - Set the timestep of the selected track to 0.
            => The role of the timestep 0 is to provide the given audio.
            => Loss is only calculated for the regions with noise.
        
        mask_loss_track
            don't calculate the loss for the given track.
            => used when pretraining with mixture only dataset
            => pretrain mix/submix track.
            => thus, it is set to 2.
        """
        super().__init__()
        
        self.use_ema = use_ema
        self.log_loss_info = log_loss_info
        self.optimizer_configs = optimizer_configs
        self.pre_encoded = pre_encoded
        self.cfg_dropout_prob = cfg_dropout_prob
        self.timestep_sampler = timestep_sampler
        self.timestep_dropout_prob = timestep_dropout_prob
        self.log_every = logging_config.get("log_every", 1)
        self.metrics_logger = MetricsLogger()
        
        self.timestep_dropout_prob = timestep_dropout_prob
        self.timestep_eps = timestep_eps
        
        self.mixture_only_pretraining = mixture_only_pretraining
        if mixture_only_pretraining:
            self.mask_loss_track = 2
            assert timestep_dropout_prob == 0.0
        else:
            self.mask_loss_track = None
            
            
            
        self.diffusion = model
        # self.diffusion_ema = model
        self.diffusion_ema = EMA(
            self.diffusion.model_dit,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        ) if use_ema else None
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        
        self.loss_modules = [
            MSELoss(
                key_a="output",
                key_b="targets",
                weight=1.0,
                mask_key=None,
                loss_mask_key="timestep_mask", ## we don't calculate the loss for the given track
                name="mse_loss"
            )
        ]
        
        self.losses = MultiLoss(self.loss_modules)
        
        if not self.pre_encoded:
            raise NotImplementedError("On-the-fly pretransform training is not implemented yet.")
        
        self.num_tracks = self.diffusion.num_tracks
        self.diffusion_objective = self.diffusion.diffusion_objective
        
    
    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config["optimizer"], self.diffusion.parameters())
        
        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]
            
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the pre-extracted latents.
        TODO: on-the-fly pretransform training
        
        batch is like:
        {
            "src_latent": (B, 64, 80)
            "src_clap": (B, 512) => Audio Clap condition
            "src_prompt": ["The sound of a Bass", "The sound of a Drums", ...]
            "submix_latent": (B, 64, 80)
            "submix_clap": (B, 512) 
            "mix_latent": (B, 64, 80)
            "mix_clap": (B, 512)
        }
        """
        z_mix, z_submix, z_src = batch["mix_latent"], batch["submix_latent"], batch["src_latent"]
        batch_size, latent_dim, _ = z_mix.shape
        
        loss_info = {}
        
        # with torch.cuda.amp.autocast():
        self.diffusion.conditioner.set_device(self.device)
        conditioning = self.diffusion.conditioner(batch)
        """
        conditioning:
        {
            "audio_cond": [(B, 1, 512), (B, 1, 512), (B, 1, 512)],
            "prompt_cond": [None, (B, 512), (B, 512)],
        }
            """
        
        diffusion_input = torch.cat([z_mix, z_submix, z_src], dim=1) # (B, 3C, T)
        if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
            diffusion_input = diffusion_input / self.diffusion.pretransform.scale
        
        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(batch_size)[:, 0].to(self.device) ## t: (B, )
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(batch_size, device=self.device)) # t: (B, )
        else:
            raise ValueError(f"Unknown timestep sampler: {self.timestep_sampler}")

        t = torch.clamp(t, min=self.timestep_eps)
        
        
        ## Random sampling within batch
        timestep_mask = torch.ones(batch_size, self.num_tracks, device=self.device)
        dropout_decision = torch.rand(batch_size, device=self.device) < self.timestep_dropout_prob
        if dropout_decision.any():
            dropout_indices = dropout_decision.nonzero(as_tuple=True)[0]
            ## From each sample, randomly select one track to drop
            track_to_drop = torch.randint(0, self.num_tracks, (dropout_indices.shape[0],), device=self.device)
            timestep_mask[dropout_indices, track_to_drop] = 0.0
        
        if self.mask_loss_track is not None:
            timestep_mask[:, self.mask_loss_track] = 0.0 # (B, 3)
        
        t_ = t.unsqueeze(1).expand(-1, self.num_tracks) # (B, 3)
        t_ = t_ * timestep_mask # (B, 3)
        
        ## Calculate the noise schedule schedule parameter for those timesteps
        ## We use different timesteps for each track
        t_expand = t_.repeat_interleave(latent_dim, dim=1) # (B, 3C)
        t_expand = t_expand.unsqueeze(-1) # (B, 3C, 1)
        
        tmask_expand = timestep_mask.repeat_interleave(latent_dim, dim=1) # (B, 3C)
        tmask_expand = tmask_expand.unsqueeze(-1) # (B, 3C, 1)

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t_expand) ## alphas, sigmas: (B, 3C, 1)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1 - t_expand, t_expand
        else:
            raise ValueError(f"Unknown diffusion objective: {self.diffusion_objective}")

        noise = torch.randn_like(diffusion_input) # (B, 3C, T)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        
        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input
        
        # with torch.cuda.amp.autocast():
        output = self.diffusion(
            noised_inputs,
            t_,
            cond=conditioning,
            cfg_dropout_prob=self.cfg_dropout_prob,
        )
        
        loss_info.update({
            "output": output, ## (B, 3C, T)
            "targets": targets,
            "timestep_mask": tmask_expand, ## if 0, the loss is not calculated for the track
        })
        
        loss, losses = self.losses(loss_info)
        
        if self.log_loss_info:
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none") # loss_all: (B, 3C, T)
            
            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()
            # w: world size, b: batch size, c: channels, n: seq_len
            
            # gather loss_all across all gpus
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")
            
            ## Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean()
                                    for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i / num_loss_buckets:.1f}":
                loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)
                
        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.metrics_logger.add(log_dict)
        if (self.global_step - 1) % self.log_every == 0:
            log_dict = self.metrics_logger.pop()
            self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema:
            self.diffusion.model_dit = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)
            
            
class MGELDM_DemoCallback(pl.Callback):
    def __init__(
        self,
        demo_every: int,
        num_demos: int,
        demo_steps: int,
        t_min: float,
        # log_duration: float = 60. # in seconds,
        sample_size: int, 
        sample_rate: int,
        demo_conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]],
        demo_cond_from_batch: bool,
        demo_cfg_scales: tp.List[float],
    ):
        """
        demo_conditioning:
            - generate demo with this condition
        demo_cond_from_batch:
            - If true, the callback will use the metadata from the batch to generate the demo conditioning
        """
        super().__init__()
        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        # print(f"Demo samples: {self.demo_samples} (in samples)")
        # assert False
        self.demo_steps = demo_steps
        self.t_min = t_min
        # self.log_duration = log_duration
        self.sample_rate = sample_rate
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales
        self.demo_cond_from_batch = demo_cond_from_batch
        
        self.last_demo_step = -1
        
        if not demo_cond_from_batch:
            assert demo_conditioning
            self.num_demos = len(demo_conditioning)
            self.demo_conditioning = demo_conditioning[:self.num_demos]
        
        
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(
        self, 
        trainer, 
        module: MGELDM_TrainingWrapper, 
        outputs,
        batch, 
        batch_idx):

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        """
        batch:
        {
            "src_latent": (B, 64, 80)
            "src_clap": (B, 512) => Audio Clap condition
            "src_prompt": ["The sound of a Bass", "The sound of a Drums", ...]
            "submix_latent": (B, 64, 80)
            "submix_clap": (B, 512) 
            "mix_latent": (B, 64, 80)
            "mix_clap": (B, 512)
        }
        """

        module.eval()

        print("->->-> Generating samples...")
        self.last_demo_step = trainer.global_step

        demo_samples = self.demo_samples ## Audio length (in samples)
        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            # Get metadata from the batch // real, metadata = batch
            # demo_cond = batch[1][:self.num_demos]
            demo_cond = batch
            try:
                demo_cond = reduce_batch(demo_cond, self.num_demos)
            except:
                print("Error in reducing batch")
                print(f"Batch size should be larger than the number of demos {self.num_demos}")
                return
        

        if module.diffusion.pretransform:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        ## Diffusion input: (B, 3C, T)
        dim_in = module.diffusion.num_tracks * module.diffusion.io_channels
        noise = torch.randn([self.num_demos, dim_in, demo_samples]).to(module.device)

        specify_global_key = "prompt_cond"
        
        try:
            # Generation demo
            # with torch.cuda.amp.autocast():
            conditioning = module.diffusion.conditioner(demo_cond) ## Forward of multiconditioner_batch

            # cond_inputs = module.diffusion.get_conditioning_inputs(conditioning, 
            #                                                        specify_global_key=specify_global_key)
            # => cond_inputs: {"global_embed": (B, 3, 512)}
            

            log_dict = {}
            ##################################### Total Generation (src conditioned) ##################################################################################
            """
            Total Generation Demo
            """
            for cfg_scale in self.demo_cfg_scales:

                print(f"[ CFG scale: {cfg_scale} ] Mixture Generation")

                # with torch.cuda.amp.autocast():
                model = module.diffusion_ema.model if module.diffusion_ema else module.diffusion

                # Sample
                ## nosie: (B, C3, T)
                # global_embed = torch.zeros_like(cond_inputs["global_embed"]) # (B, 3, 512)
                cond_inputs = module.diffusion.get_conditioning_inputs(
                    conditioning,
                    specify_global_key=specify_global_key,
                    mask_tracks=[0, 1] ## For unconditional generation
                )
                if module.diffusion_objective == "v":
                    fakes = sample(model, noise, self.demo_steps, 0,
                                    verbose=True, t_min=self.t_min,
                                    **cond_inputs, cfg_scale=cfg_scale, 
                                    # batch_cfg=True
                                    )
                    # fakes: (B, C3, T)
                elif module.diffusion_objective == "rectified_flow":
                    raise NotImplementedError("t_min should be implemented for rectified_flow")
                    fakes = sample_discrete_euler(model, noise, self.demo_steps, verbose=True, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)

                # Decode
                if module.diffusion.pretransform:
                    # fake_mix, fake_submix, fake_src = fakes.chunk(3, dim=1)
                    fake_mix_z, fake_submix_z, fake_src_z = fakes.chunk(3, dim=1) # latents
                    fake_tracks_z = torch.cat([fake_mix_z, fake_submix_z, fake_src_z], dim=0) # (3B, C, T)
                    fakes_tracks = module.diffusion.pretransform.decode(fake_tracks_z)
                    # fakes_tracks: (3B, 1, T)
                    bs = fake_mix_z.shape[0]
                    fake_mix, fake_submix, fake_src = torch.chunk(fakes_tracks, 3, dim=0) 
                    # each, (B, 1, T)
                else:
                    raise NotImplementedError

                sample_dir = os.path.join(trainer.default_root_dir, 'samples', f'{trainer.global_step:08}', 'total_gen', f'cfg_{cfg_scale}')
                os.makedirs(sample_dir, exist_ok=True)

                # fakes : (bs, ch, length)
                # fakes_tracks : (3bs, ch, length)

                table_audio = wandb.Table(columns=['id', 'mix', 'mix_spec', 'submix', 'submix_spec', 'src', 'src_spec'])
                for idx in range(bs):
                    mix_audio = float_to_int16_audio(fake_mix[idx])
                    submix_audio = float_to_int16_audio(fake_submix[idx])
                    src_audio = float_to_int16_audio(fake_src[idx])
                    
                    # sample_audio = float_to_int16_audio(fakes[idx])

                    # add audio row
                    # desc = demo_cond[idx].get('prompt', f'sample_{idx}')
                    
                    if specify_global_key=="prompt_cond":
                        # pass
                        src_inst_txt = demo_cond["src_prompt"][idx]
                        # submix_inst_txt = demo_cond["submix_prompt"][idx]
                    else:
                        raise NotImplementedError

                    # desc = f"sample_{idx}"
                    # desc = f"src: {src_inst_txt},\n submix: {submix_inst_txt}"
                    desc = f"src: {src_inst_txt}"
                    
                    table_audio.add_data(
                        desc,
                        wandb.Audio(mix_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(mix_audio)),
                        wandb.Audio(submix_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(submix_audio)),
                        wandb.Audio(src_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(src_audio))
                    )

                    # save audio
                    torchaudio.save(f'{sample_dir}/mix_{idx + 1}.wav', mix_audio, self.sample_rate) # mix_audio: (1, T)
                    torchaudio.save(f'{sample_dir}/submix_{idx + 1}.wav', submix_audio, self.sample_rate)
                    torchaudio.save(f'{sample_dir}/src_{idx + 1}.wav', src_audio, self.sample_rate)

                log_dict = {f'Generation samples cfg: {cfg_scale}': table_audio}
                trainer.logger.experiment.log(log_dict)

            del fakes

            ################################ Source Extraction #######################################################################################
            cond_inputs = module.diffusion.get_conditioning_inputs(
                        conditioning,
                        specify_global_key=specify_global_key,
                        mask_tracks=[1] ## submix is null cond. (mix cond is None, so also null cond.)
                    )

            ## xxx0: denotes that xxx is given.
            noise_mix0 = torch.randn([self.num_demos, dim_in, demo_samples]).to(module.device)
            ## noise: (n_demos, 3C, T)
            for cfg_scale in self.demo_cfg_scales:
                print(f"[ CFG scale: {cfg_scale} ] Separation (submix cond not used)")
                mix_z = demo_cond["mix_latent"] # (B, 64, 80)

                # mix = batch["mix_signal"].audio_data.to(module.device) # (B, 1, T)
                # mix_z = module.diffusion.pretransform.encode(mix) # (B, C, T) ## TODO: on-the-fly training
                latent_dim = mix_z.shape[1]
                # noise[:, :latent_dim].shape : (B, C, T)
                noise_mix0[:, :latent_dim] = mix_z[:self.num_demos]


                # with torch.cuda.amp.autocast():
                model = module.diffusion_ema.model if module.diffusion_ema else module.diffusion

                if module.diffusion_objective == "v":
                    fakes = sample_extraction(
                        model, noise_mix0, self.demo_steps, eta=0.0, 
                        verbose=True, t_min=self.t_min, 
                        **cond_inputs, cfg_scale=cfg_scale, 
                        # batch_cfg=True
                    )
                    ## fakes: (B, 3C, T)
                elif module.diffusion_objective == "rectified_flow":
                    raise NotImplementedError
                    fakes = sample_discrete_euler(model, noise_mix0, self.demo_steps, verbose=True, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)

                # Decode
                if module.diffusion.pretransform:
                    fake_mix_z_dummy, fake_submix_z, fake_src_z = fakes.chunk(3, dim=1)
                    fake_tracks_z = torch.cat([fake_submix_z, fake_src_z], dim=0) # (2B, C, T)
                    fakes_tracks = module.diffusion.pretransform.decode(fake_tracks_z)
                    # fakes_tracks: (2B, 1, T)
                    bs = fake_submix_z.shape[0]
                    fake_submix, fake_src = torch.chunk(fakes_tracks, 2, dim=0)
                    
                    ## gt mix, src wav
                    gt_src_z = demo_cond["src_latent"] # (B, 64, 80)
                    
                    mix_wav = module.diffusion.pretransform.decode(mix_z)
                    gt_src_wav = module.diffusion.pretransform.decode(gt_src_z)
                else:
                    raise NotImplementedError
                        
                sample_dir = os.path.join(trainer.default_root_dir, 'samples', f'{trainer.global_step:08}', 'extraction', f'cfg_{cfg_scale}')
                os.makedirs(sample_dir, exist_ok=True)

                table_audio = wandb.Table(columns=['id', 'mix_0', 'mix0_spec', 
                                                   'submix', 'submix_spec', 
                                                   'src', 'src_spec',
                                                   'gt_src', 'gt_src_spec'])
                for idx in range(bs):
                    mix_audio = float_to_int16_audio(mix_wav[idx])
                    submix_audio = float_to_int16_audio(fake_submix[idx])
                    src_audio = float_to_int16_audio(fake_src[idx])
                    gt_src_audio = float_to_int16_audio(gt_src_wav[idx])
                    
                    if specify_global_key=="prompt_cond":
                        src_inst_txt = demo_cond["src_prompt"][idx]
                        # submix_inst_txt = "||".join(demo_cond["src_inst"][idx])
                        # submix_inst_txt = demo_cond["rest_prompt"][idx]
                    else:
                        raise NotImplementedError

                    # desc = f"sample_{idx}"
                    # desc = f"src: {src_inst_txt},\n submix: {submix_inst_txt}"
                    desc = f"src: {src_inst_txt}"

                    table_audio.add_data(
                        desc,
                        wandb.Audio(mix_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(mix_audio)),
                        wandb.Audio(submix_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(submix_audio)),
                        wandb.Audio(src_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(src_audio)),
                        wandb.Audio(gt_src_audio.numpy().T, sample_rate=self.sample_rate),
                        wandb.Image(audio_spectrogram_image(gt_src_audio))
                    )
                
                    torchaudio.save(f'{sample_dir}/mix_given_{idx + 1}.wav', mix_audio, self.sample_rate)
                    torchaudio.save(f'{sample_dir}/submix_{idx + 1}.wav', submix_audio, self.sample_rate)
                    torchaudio.save(f'{sample_dir}/src_{idx + 1}.wav', src_audio, self.sample_rate)
                    torchaudio.save(f'{sample_dir}/gt_src_{idx + 1}.wav', gt_src_audio, self.sample_rate)
                    
                
                log_dict = {f'Separation samples cfg: {cfg_scale}': table_audio}
                trainer.logger.experiment.log(log_dict)
            
            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()

