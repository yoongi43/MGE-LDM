import torch
from torch.nn.utils import remove_weight_norm
from safetensors.torch import load_file


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
    # if adjust_key_names:
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith("model."):
    #             print(f"Adjusting key name: {key} -> {key.replace('model.', '')}")
    #             new_key = key.replace("model.", "")
    #         else:
    #             new_key = key
    #             print('Not  adjusting key name:', key)
    #             assert False
    #         new_state_dict[new_key] = value
    #     state_dict = new_state_dict

    return state_dict

def exists(x: torch.Tensor):
    return x is not None