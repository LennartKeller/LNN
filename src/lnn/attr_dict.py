from __future__ import annotations

import numpy as np
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TensorAttrDict(AttrDict):
    def to(self, device) -> TensorAttrDict:
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self

    def cpu(self) -> TensorAttrDict:
        return self.to("cpu")

    def detach(self) -> TensorAttrDict:
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.detach()
        return self

    def numpy(self) -> TensorAttrDict:
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.numpy()
        return self

    def torch(self) -> TensorAttrDict:
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                self[k] = torch.from_numpy(v)
        return self
