from pathlib import Path
from typing import Optional

import numpy as np
import torch
from IPython.display import Audio, display

from .utils import load_audio


def display_audio(
    waveform: torch.Tensor | np.ndarray,
    rate: Optional[int] = None,
    path: Optional[str | Path] = None,
):
    if path is None:
        if rate is None:
            raise ValueError(
                "If a waveform is provided, the rate must also be specified."
            )
        if torch.is_tensor(waveform):
            waveform = waveform.detach().cpu().numpy()
    else:
        waveform, rate = load_audio(path)
    display(Audio(data=waveform, rate=rate))
