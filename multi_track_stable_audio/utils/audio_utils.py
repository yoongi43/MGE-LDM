
import torch
import torch.nn as nn
from torchaudio import transforms as T

EPS = 1e-8


def is_silence(audio: torch.Tensor, thresh: float = -60.):
    """checks if entire clip is 'silence' below some dB threshold

    Args:
        audio (Tensor): torch tensor of (multichannel) audio
        thresh (float): threshold in dB below which we declare to be silence
    """

    def get_dbmax(audio: torch.Tensor):
        """finds the loudest value in the entire clip and puts that into dB (full scale)"""
        return 20 * torch.log10(torch.flatten(audio.abs()).max() + EPS).item()

    return get_dbmax(audio) < thresh


def float_to_int16_audio(x: torch.Tensor, maximize: bool = False):
    div = x.abs().max().item()
    if not maximize:
        div = max(div, 1.0)

    return x.div(div).mul(32767).to(torch.int16).cpu()

# from ..data.modification import PadCrop

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output
    
    
def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    assert target_channels in [1, 2]

    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio

def to_numpy(tensor: torch.Tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()