import torch
from torch.nn.parameter import Parameter
# from stable_audio_tools.models import create_model_from_config
from multi_track_stable_audio.models.factory import create_autoencoder_from_config, create_mgeldm_from_config
import hydra
from hydra.core.hydra_config import HydraConfig
import os

"""
In config, set new args
    - type: Type of model to unwrap (autoencoder or mgeldm)
    - ckpt_path: Path to the checkpoint file
    - use_safetensors: Whether to export the model as a safetensors file
        => default: False
    - output_name: Name of the exported model file (without extension)
        => = output path without extension
"""

@hydra.main(version_base=None, config_path="configs", config_name="default")
def unwrap_model(config):
    assert config.type in ["autoencoder", "mgeldm"]
    device = torch.device(f"cuda:0")
    # device = torch.device('cpu')
    model_config = config.model
    trainer_config = config.trainer
    if config.type == "autoencoder":
        from multi_track_stable_audio.training.pipeline_ae import AutoencoderTrainingWrapper
        model = create_autoencoder_from_config(model_config).to(device)
        ema_copy = None
        use_ema = trainer_config.get("use_ema", False)
        
        if use_ema:
            ema_copy = create_autoencoder_from_config(model_config).to(device)
            ema_copy = create_autoencoder_from_config(model_config)  # I don't know why this needs to be called twice but it broke when I called it once
            
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)
        training_wrapper = AutoencoderTrainingWrapper.load_from_checkpoint(
            config.ckpt_path,
            autoencoder=model,
            strict=True,
            loss_config=trainer_config["loss_configs"],
            optimizer_configs=trainer_config["optimizer_configs"],
            use_ema=trainer_config["use_ema"],
            ema_copy=ema_copy
        ).to(device)
        
    elif config.type == "mgeldm":
        from multi_track_stable_audio.training.pipeline_diffusion import MGELDM_TrainingWrapper
        model = create_mgeldm_from_config(config.model).to(device)
        use_ema = trainer_config.get("use_ema", True)
        training_wrapper = MGELDM_TrainingWrapper.load_from_checkpoint(
            config.ckpt_path,
            model=model,
            use_ema=use_ema,
            optimizer_configs=trainer_config.get("optimizer_configs", None),
            strict=True
        ).to(device)
    else:
        raise ValueError(f"Unknown model type {config.type}")

    print(f"Loaded model from {config.ckpt_path}")
    ext = "safetensors" if config.use_safetensors else "ckpt"
    ckpt_path = f"{config.output_name}.{ext}"
    training_wrapper.export_model(
        ckpt_path, use_safetensors=config.use_safetensors
    )
    
    print(f"Exported model to {ckpt_path}")
    
    
if __name__ == "__main__":
    unwrap_model()