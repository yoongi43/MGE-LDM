import os
import re
import copy
import pytorch_lightning as pl

def get_exp_id(base_dir, config_filename_without_ext, project=None):
    """base_dir 폴더 안에서 config 이름 기반으로 version 붙이기"""
    # base_name = os.path.splitext(config_filename_without_ext)[0]  # 확장자 제거
    base_name = config_filename_without_ext
    candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    matching = [d for d in candidates if d == base_name or re.match(rf"{re.escape(base_name)}_v\d+$", d)]

    if base_name not in matching:
        return base_name
    else:
        versions = [int(re.search(r'_v(\d+)$', d).group(1)) for d in matching if '_v' in d]
        next_version = max(versions, default=0) + 1
        return f"{base_name}_v{next_version}"
    
    
class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config
        
def push_wandb_config(wandb_logger, args, omit=[]): 
    """
    save config to wandb (for possible retrieval later)
    Omit: list of args you don't want pushed to wandb; will push an empty string for these
    """
    if hasattr(wandb_logger.experiment.config, 'update'): #On multi-GPU runs, only process rank 0 has this attribute!
        copy_args = copy.deepcopy(args)
        for var_str in omit:  # don't push certain reserved settings to wandb
            if hasattr(copy_args, var_str):
                setattr(copy_args, var_str, 'OMITTED')
        wandb_logger.experiment.config.update(copy_args)
        
        
def reduce_batch(batch, num_samples):
    """
    Reduces the batch size to num_samples by taking the first num_samples samples
    from each tensor in the batch.
    """
    return {key: value[:num_samples] for key, value in batch.items()}