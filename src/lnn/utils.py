from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from iso639 import languages
from torch import nn

# map macro lang to its individual lang with the largest population or simply the first child entry on Wikipedia
_macro_to_individual = {
    "zho": "cmn",
    "ara": "arb",
    "aze": "azb",
    "fas": "pes",
    "msa": "zlm",
    "ori": "ory",
    "kok": "gom",
    "srd": "sro",
    "est": "ekk",
    "grn": "gug",
}


def normalize_lang_id(lang_id):
    if len(lang_id) == 3:
        # macro to individual language normalization
        if lang_id in _macro_to_individual:
            return _macro_to_individual[lang_id]

        return lang_id

    assert len(lang_id) == 2

    language = languages.get(part1=lang_id)
    iso3 = language.part3
    if iso3 in _macro_to_individual:
        return _macro_to_individual[iso3]
    return iso3


def gets(d: dict, keys: list[str]) -> list:
    return [d[k] for k in keys]


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    n_params = 0
    for param in model.parameters():
        if torch.is_floating_point(param):
            if only_trainable:
                if param.requires_grad:
                    n_params += param.numel()
            else:
                n_params += param.numel()
    return n_params


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def flatten_dataset_dict(ds_dict: DatasetDict, key_col_name: str = "split") -> Dataset:
    datasets = []
    for key, dataset in ds_dict.items():
        key_column = [key] * len(dataset)
        dataset = dataset.add_column(key_col_name, key_column)
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def load_dataset_split_from_disk(dataset_path: str | Path, split: str):
    dataset = load_from_disk(dataset_path)[split]
    return dataset
