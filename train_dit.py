import os; opj = os.path.join
import re
from omegaconf import OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
from multi_track_stable_audio.models.factory import create_mgeldm_from_config
from multi_track_stable_audio.training.factory import create_training_mgeldm_from_config, create_demo_callback_mgeldm_from_config, create_tqdm_callback_from_config
from multi_track_stable_audio.utils import set_seed, ExceptionCallback, ModelConfigEmbedderCallback
from multi_track_stable_audio.utils import load_ckpt_state_dict
# from data.dataset_multi import create_multi_dataloader_from_config
from data.dataset_multi_latent import create_multi_latent_dataloader_from_config

import warnings
warnings.filterwarnings("ignore")
# os.environ["HYDRA_FULL_ERROR"] = "1"  # Show full Hydra error messages

@hydra.main(version_base=None, config_path="configs", config_name="default_dit")
def main(config):
    """
    Main function to train the diffusion model.
    """
    seed = config.seed
    save_dir = config.save_dir
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))
    
    set_seed(seed)
    
    if config.ckpt_path:
        print(f"Resuming from checkpoint: {config.ckpt_path}")
        # assert False
    # import pdb; pdb.set_trace()
    
    # Create the diffusion model
    model = create_mgeldm_from_config(config.model)
    
    ## Load pretransform weights
    if config.autoencoder_ckpt_path:
        print(f"Loading AutoEncoder weights from {config.autoencoder_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(config.autoencoder_ckpt_path,),
            strict=True
        )
    else:
        raise NotImplementedError("Autoencoder checkpoint path is not provided. Please set 'autoencoder_ckpt_path' in the config.")
    
    if config.pretrained_ckpt_path:
        assert config.ckpt_path is None, "Cannot set both 'pretrained_ckpt_path' and 'ckpt_path'. Please choose one."
        print(f'Mixture-only pretrained model')
        print(f"Loading pretrained weights from {config.pretrained_ckpt_path}")
        # copy_state_dict(model, load_ckpt_state_dict(config.pretrained_ckpt_path, strict=False))
        model.load_state_dict(
            load_ckpt_state_dict(config.pretrained_ckpt_path,),
            strict=True
        )
    else:
        print(f"Training from scratch")
        print(f"Mixture only pretraining: {config.mixture_only_pretraining}")
    
    print("Model created successfully.")
    
    train_loader = create_multi_latent_dataloader_from_config(
        dataset_configs=config.data,
        split='train',
        batch_size=config.batch_size,
        sample_rate=config.model.sample_rate,
        segment_length=config.model.segment_length,
        mixture_only_pretraining=config.mixture_only_pretraining,
        ae_comp_ratio=model.pretransform.downsampling_ratio,
        num_workers=config.num_workers,
        shuffle=True,
    )
    print(f"Train dataloader created")
    
    training_wrapper = create_training_mgeldm_from_config(model=model,
                                                          training_config=config.trainer,)
    print("Training wrapper created successfully.")
    
    hc = HydraConfig.get()
    exp_name = hc.job.config_name
    save_dir = opj(save_dir, exp_name)
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    wandb_logger = pl.loggers.WandbLogger(project=config.wandb_project,
                                          name=exp_name,
                                          config=cfg_dict,
                                          )
    wandb_logger.watch(training_wrapper)
    
    exc_callback = ExceptionCallback()
    
    ckpt_config = config.trainer.get("checkpoint", {"every_n_train_steps": 10000, "save_top_k": 1, "save_last": True, "monitor": "train/loss", "mode": "min"})
    checkpoint_dir = opj(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # ckpt_callback = pl.callbacks.ModelCheckpoint(**ckpt_config, dirpath=checkpoint_dir)
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch}-step-{step}-{train/loss:.4f}",     
        **ckpt_config,  # Unpack the checkpoint config
        # monitor=config.trainer["checkpoint"].get("monitor", "train/loss"),                    
        # mode="min",                               
        # every_n_train_steps=config.trainer["checkpoint"].get("every_n_train_steps", 10000),               
        # save_top_k=config.trainer["checkpoint"].get("save_top_k", 1),
        # save_last=True,
    )
    save_model_config_callback = ModelConfigEmbedderCallback(config.model)
    
    demo_callback = create_demo_callback_mgeldm_from_config(
        training_config=config.trainer,
        model_config=config.model,
    )
    tqdm_callback = create_tqdm_callback_from_config(config.trainer)
    
    trainer_config = config.trainer
    
    if trainer_config.strategy:
        if trainer_config.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True
            )      
        else:
            strategy = trainer_config.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if trainer_config.num_gpus > 1 else "auto"
        # strategy = 'ddp' if trainer_config.num_gpus > 1 else "auto"
            
    trainer = pl.Trainer(
        devices=trainer_config.num_gpus,
        accelerator="gpu",
        num_nodes=trainer_config.num_nodes,
        strategy=strategy,
        precision=trainer_config.precision,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        callbacks=[ckpt_callback, demo_callback, tqdm_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=trainer_config.max_epochs,
        default_root_dir=save_dir,
        gradient_clip_val=trainer_config.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=2,
        # detect_anomaly=True,
    )
    
    # import pdb; pdb.set_trace()
    
    trainer.fit(training_wrapper,
                train_dataloaders=train_loader,
                ckpt_path=config.ckpt_path if config.ckpt_path else None ## Resuming
                )

if __name__=="__main__":
    main()