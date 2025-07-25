# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import logging
import warnings
import string
import typing as tp
import gc

import torch
from torch import nn

from einops import rearrange

# from .adp import NumberEmbedder
# from ..utils import set_audio_channels, load_ckpt_state_dict
# from .factory import create_pretransform_from_config
# from .pretransforms import Pretransform
# from ..utils import load_ckpt_state_dict


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
    ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim

        ##! We do not use proj_out
        assert not project_out
        assert dim == output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def set_device(self, device: tp.Any) -> None:
        raise NotImplementedError()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    

class CLAPTextConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        clap_ckpt_path: str,
        use_text_features=False,
        feature_layer_ix: int = -1,
        audio_model_type="HTSAT-base",
        enable_fusion=True,
        project_out: bool = False,
        finetune: bool = False
    ):
        super().__init__(
            dim=512,
            output_dim=output_dim,
            project_out=project_out
        )
        assert output_dim==512
        assert use_text_features is False
        assert feature_layer_ix is None
        assert finetune is False
        assert project_out is False

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune
        self.device = 'cpu'

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict

                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model 

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()
            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def set_device(self, device):
        self.to(device)
        self.model.to(device)
        self.device = device

    def get_clap_features(self, prompts, layer_ix=-2):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=self.device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=self.device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    @torch.no_grad() # finetune is False.
    def forward(self, texts: tp.List[str]) -> tp.Any:
        if self.use_text_features: # False
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix)

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True) # (B, 512) 

        text_embedding = text_embedding.unsqueeze(1)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(self.device)]


class CLAPAudioConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        clap_ckpt_path: str,
        audio_model_type="HTSAT-base",
        enable_fusion=True,
        project_out: bool = False,
        finetune: bool = False
    ):
        """
        We actually don't use this conditioner while training the diffusion model.
        Becuase we pre-extract the audio features and save them in the dataset.
        """
        super().__init__(
            dim=512,
            output_dim=output_dim,
            project_out=project_out
        )
        assert output_dim==512
        assert project_out is False
        assert finetune is False
        self.finetune = finetune
        self.device = 'cpu'
        
        print("WARNING!! sample rate must be 48kHz for CLAP. \
            Please check the sample rate of the input audio. \
            In the CLAPAudioConditioner, we don't check the adequacy of the sample rate.")

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict

                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def set_device(self, device):
        self.to(device)
        self.model.to(device)
        self.device = device

    @torch.no_grad() # finetune is False.
    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]]) -> tp.Any:
        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1) # (B, L)

        # with torch.cuda.amp.autocast(enabled=False):
        audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)
        # -> (B, 512)

        audio_embedding = audio_embedding.unsqueeze(1) # (B, 1, 512)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(self.device)]
    
    
    
class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """

    def __init__(self, conditioners: tp.Dict[str, Conditioner]):
        super().__init__()
        ## HARD-CODED CONDITIONERS: audio_cond and prompt_cond
        self.conditioners = nn.ModuleDict(conditioners)
        self.key_list = ["audio_cond", "prompt_cond"]

    def set_device(self, device):
        for mod in self.conditioners.values():
            mod.set_device(device)
            
    def forward(self, batch: tp.Dict)->tp.Dict:
        """
        Batch is a dictionary such as:
        {
            "src_latent": (B, C, T) = (B, 64, 80)
            "src_clap" : (B, 512) => Audio CLAP condition. We pre-extract this feature.
            "src_prompt" : ["The sound of a Bass", "The sound of a Drums", ...]
            "submix_latent": (B, 64, 80)
            "submix_clap" : (B, 512) => Audio CLAP condition. We pre-extract this feature.
            "mix_latent": (B, 64, 80)
            "mix_clap": (B, 512) => Audio CLAP condition. We pre-extract this feature.
        }
        """
        output = {}
        for key in self.key_list:
            if key == "audio_cond":
                ## There is no 'audio_cond' key in self.conditioners
                mix_emb, submix_emb, src_emb = batch["mix_clap"], batch["submix_clap"], batch["src_clap"]
                mix_emb, submix_emb, src_emb = mix_emb.unsqueeze(1), submix_emb.unsqueeze(1), src_emb.unsqueeze(1)
            
            elif key == "prompt_cond":
                conditioner = self.conditioners[key]
                src_txt = batch.get("src_prompt", None)
                if src_txt:
                    src_emb, mask_txt = conditioner(src_txt) # (B, 1, 512), ones(B, 1)
                else:
                    src_emb = None
                
                submix_txt = batch.get("submix_prompt", None)
                if submix_txt:
                    submix_emb, mask_submix_txt = conditioner(submix_txt)
                else:   
                    submix_emb = None
                mix_emb = None
            
            output[key] = [mix_emb, submix_emb, src_emb]
        return output