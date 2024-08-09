from pathlib import Path
from typing import Optional

import numpy as np
import torch
from IPython.display import Audio, display

from .utils import load_audio


def display_audio(
    waveform: Optional[torch.Tensor | np.ndarray] = None,
    rate: Optional[int] = None,
    path: Optional[str | Path] = None,
):
    if path is None:
        if waveform is None:
            raise ValueError(
                "Audio has to either provided as waveform or via a path to an audio-file."
            )
        if rate is None:
            raise ValueError(
                "If a waveform is provided, the rate must also be specified."
            )
        if torch.is_tensor(waveform):
            waveform = waveform.detach().cpu().numpy()
    else:
        waveform, rate = load_audio(path)
    display(Audio(data=waveform, rate=rate))
