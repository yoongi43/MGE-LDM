import torch
from torch.nn import Parameter
from ..models.factory import create_autoencoder_from_config


def create_training_wrapper_autoencoder_from_config(training_config,
                                                    model,
                                                    model_config=None
                                                    ):
    from .pipeline_ae import AutoencoderTrainingWrapper
    
    ema_copy = None
    
    if training_config.get("use_ema", False):
        ema_copy = create_autoencoder_from_config(model_config)
        ema_copy = create_autoencoder_from_config(model_config)  # I don't know why this needs to be called twice but it broke when I called it once
        # Copy each weight to the ema copy
        for name, param in model.state_dict().items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameter
                param = param.data
            ema_copy.state_dict()[name].copy_(param)
    
    use_ema = training_config.get("use_ema", False)
    
    latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)
    
    return AutoencoderTrainingWrapper(
        autoencoder=model,
        loss_config=training_config["loss_configs"],
        optimizer_configs=training_config["optimizer_configs"],
        lr=training_config["learning_rate"],
        warmup_steps=training_config.get("warmup_steps", 0),
        encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
        sample_rate=model_config["sample_rate"],
        use_ema=use_ema,
        ema_copy=ema_copy if use_ema else None,
        force_input_mono=training_config.get("force_input_mono", False),
        latent_mask_ratio=latent_mask_ratio,
        logging_config=training_config.get("logging", {})
    )    

def create_demo_callback_autoencoder_from_config(
    training_config,
    model_config, 
    **kwargs):
    from .pipeline_ae import AutoencoderDemoCallback
    demo_config = training_config.get("demo", {})
    return AutoencoderDemoCallback(
        demo_every=demo_config.get("demo_every", 2000),
        max_num_sample=demo_config.get("max_num_sample", 4),
        sample_size=model_config["sample_size"], # seg len?
        sample_rate=model_config["sample_rate"],
        **kwargs
    )
    
def create_training_mgeldm_from_config(
    training_config,
    model,
    mixture_only_pretraining=False,
):
    from .pipeline_diffusion import MGELDM_TrainingWrapper
    return MGELDM_TrainingWrapper(
        model=model,
        use_ema=training_config["use_ema"],
        log_loss_info=training_config["log_loss_info"],
        optimizer_configs=training_config["optimizer_configs"],
        pre_encoded=training_config["pre_encoded"],
        cfg_dropout_prob=training_config["cfg_dropout_prob"],
        timestep_sampler=training_config["timestep_sampler"],
        timestep_dropout_prob=training_config["timestep_dropout_prob"],
        timestep_eps=training_config["timestep_eps"],
        
        mixture_only_pretraining=mixture_only_pretraining,
        logging_config=training_config["logging_config"],
        # **training_config,
    )
    
def create_demo_callback_mgeldm_from_config(
    training_config,
    model_config,
):
    from .pipeline_diffusion import MGELDM_DemoCallback
    # demo_config = training_config.get("demo_config", {})
    demo_config = training_config["demo"]
    return MGELDM_DemoCallback(
        demo_every=demo_config["demo_every"],
        num_demos=demo_config["num_demos"],
        demo_steps=demo_config["demo_steps"],
        t_min=demo_config["t_min"],
        sample_size=model_config["sample_size"], # seg len
        sample_rate=model_config["sample_rate"],
        demo_conditioning=demo_config["demo_conditioning"], # None
        demo_cond_from_batch=demo_config["demo_cond_from_batch"], # True
        demo_cfg_scales=demo_config["demo_cfg_scales"], # [3.0]
    )

    

def create_tqdm_callback_from_config(training_config, **kwargs):
    from pytorch_lightning.callbacks import TQDMProgressBar
    tqdm_config = training_config.get("tqdm", {})

    return TQDMProgressBar(**tqdm_config)
