import os; opj = os.path.join
import re
from omegaconf import OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))
# import pdb; pdb.set_trace()

from multi_track_stable_audio.models.factory import create_autoencoder_from_config
from multi_track_stable_audio.training.factory import (
    create_training_wrapper_autoencoder_from_config,
    create_demo_callback_autoencoder_from_config,
    create_tqdm_callback_from_config)

from data.dataset_single import create_single_dataloader_from_config
# from multi_track_stable_audio.utils.torch_common import set_seed, copy_state_dict
# from multi_track_stable_audio.utils.model_utils import load_ckpt_state_dict

from multi_track_stable_audio.utils import (
    set_seed, copy_state_dict, load_ckpt_state_dict, get_exp_id,
    ExceptionCallback, ModelConfigEmbedderCallback
)


import warnings
warnings.filterwarnings("ignore")

import wandb
    

@hydra.main(version_base=None, config_path="configs", config_name="default_ae")
def main(config):
    """
    Main function to train the autoencoder.
    """
    
    seed = config.seed
    save_dir = config.save_dir
    
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))
    
    set_seed(seed)
    
    
    # Create the autoencoder model
    model = create_autoencoder_from_config(config.model)
    
    train_loader = create_single_dataloader_from_config(
        config.data,
        split='train',
        batch_size=config.batch_size,
        sample_rate=config.model.sample_rate,
        segment_length=config.model.sample_size,
        audio_channels=config.model.audio_channels,
        num_workers=config.num_workers,
        shuffle=True,
    )
    valid_loader = create_single_dataloader_from_config(
        config.data,
        split='valid',
        batch_size=config.batch_size,
        sample_rate=config.model.sample_rate,
        segment_length=config.model.sample_size,
        audio_channels=config.model.audio_channels,
        num_workers=config.num_workers,
        shuffle=False,
    )
    
    # Load pre-trained weights if specified
    # if config.pretrained_ckpt_path:
    #     print(f"### Loading pre-trained weights from {config.pretrained_ckpt_path}")
    #     state_dict = load_ckpt_state_dict(config.pretrained_ckpt_path)
    #     copy_state_dict(model, state_dict, strict=True)

    # training_wrapper = create_training_wrapper_from_config(model_config, model)
    training_wrapper = \
        create_training_wrapper_autoencoder_from_config(
            training_config=config.trainer,
            model=model,
            model_config=config.model
        )
    
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
    
    # ckpt_config = model_config["training"].get("checkpoint", {"every_n_train_steps": 10000, "save_top_k": 1, "save_last": True})
    ckpt_config = config.trainer.get("checkpoint", {"every_n_train_steps": 10000, "save_top_k": 1, "save_last": True})
    checkpoint_dir = opj(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_callback = pl.callbacks.ModelCheckpoint(**ckpt_config, dirpath=checkpoint_dir)
    save_model_config_callback = ModelConfigEmbedderCallback(config.model)

    demo_callback = create_demo_callback_autoencoder_from_config(
        training_config=config.trainer,
        model_config=config.model,
        demo_dl=train_loader
        )
    tqdm_callback = create_tqdm_callback_from_config(config.trainer)

    trainer_config = config.trainer
    
    # import pdb; pdb.set_trace()

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
        gradient_clip_val=None, ## Clipped in the pipeline
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=2
    )
    
    trainer.fit(training_wrapper,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader if valid_loader is not None else None,
                ckpt_path=config.ckpt_path if config.ckpt_path else None ## Resuming
                )

if __name__=="__main__":
    main()