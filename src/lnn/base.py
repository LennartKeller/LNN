from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from dataclasses_json import DataClassJsonMixin
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .attr_dict import AttrDict

# from trident import TridentModule


@dataclass
class BaseConfig(DataClassJsonMixin):

    def to_attr_dict(self) -> AttrDict[str, Any]:
        return asdict(self, dict_factory=AttrDict)

    def to_omega(self) -> DictConfig:
        return OmegaConf.structured(self)

    @classmethod
    def from_omega(cls, conf: DictConfig) -> BaseConfig:
        return cls.from_dict(conf.to_object())

    @classmethod
    def from_cli(cls):
        conf = OmegaConf.from_cli()
        field_names = set(cls.__annotations__)
        data = {k: v for k, v in conf.items() if k in field_names}
        return cls.from_dict(data)

    def get(self, field: str, default: Any = None) -> Any:
        if hasattr(self, field):
            return getattr(self, field)
        return None


class ConfigurableLightningModule(L.LightningModule):
    CONFIG_CLASS = BaseConfig

    def __init__(self, config: CONFIG_CLASS) -> None:
        super().__init__()
        self.save_config(config)

    def save_config(self, config: BaseConfig):
        self.config = config
        self.save_hyperparameters(config.to_omega())

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path).expanduser().absolute()
        path.mkdir(exist_ok=True, parents=True)
        weights_path = path / "model.pt"
        torch.save(self.state_dict(), weights_path)
        config_path = path / "config.json"
        Path(config_path).write_text(self.config.to_json(indent=4))

    @classmethod
    def from_pretrained(cls, path: str | Path) -> None:
        path = Path(path)
        assert all((path.exists(), path.is_dir()))
        weighs_path = path / "model.pt"
        weights = torch.load(weighs_path, map_location="cpu")
        config_path = path / "config.json"
        config = cls.CONFIG_CLASS.from_json(config_path.read_text())
        self = cls(config)
        self.load_state_dict(weights)
        return self

    @property
    def device(self):
        return next(self.parameters()).device


# class ConfigurableTridentModule(ConfigurableLightningModule, TridentModule): ...
