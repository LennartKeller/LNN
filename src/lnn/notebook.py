import numpy as np
import torch
from IPython.display import Audio, display


def display_audio(waveform: torch.Tensor | np.ndarray, rate: int):
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().numpy()
    display(Audio(data=waveform, rate=rate))
