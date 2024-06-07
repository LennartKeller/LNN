import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
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


def get_lang_code(language: str) -> str:
    language = languages.get(name=language)
    if language is not None:
        return language.part3
    return None


def gets(d: dict, *keys: list[str]) -> list:
    return [d[k] for k in keys]


def rename_keys(d: dict, **rename_map: dict) -> dict:
    return {rename_map.get(k, k): v for k, v in d.items()}


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


def check_chars(string: str) -> list[tuple[str, int, str]]:
    return [(c, ord(c), hex(ord(c))) for c in string]


def dataset_dict_to_tsv(dataset: DatasetDict, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    for split, ds in dataset.items():
        path = save_dir / f"{split}.tsv"
        df = ds.to_pandas()
        df.to_csv(path, sep="\t", index=False)


def load_tsv_dataset(
    root_dir: str | Path, split: Optional[str] = None
) -> Dataset | DatasetDict:
    root_dir = Path(root_dir)
    if split is None:
        split_files = list(root_dir.glob("*.tsv"))
    else:
        split_files = [root_dir / f"{split}.tsv"]
    dataset = DatasetDict(
        {
            split_file.stem: Dataset.from_pandas(pd.read_table(split_file))
            for split_file in split_files
        }
    )
    if split is not None:
        dataset = dataset[split]
    return dataset


def ds_to_ndjson(dataset: Dataset, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.write_text("\n".join(json.dumps(line) for line in dataset))


def dataset_dict_to_ndjson(dataset: DatasetDict, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    for split, ds in dataset.items():
        split_file = save_dir / f"{split}.ndjson"
        ds_to_ndjson(ds, split_file)


def load_ndjson_dataset(
    root_dir: str | Path, split: Optional[str] = None, suffix: str = ".ndjson"
) -> Dataset | DatasetDict:
    root_dir = Path(root_dir)
    if split is None:
        split_files = {f.stem: str(f.absolute()) for f in root_dir.glob(f"*{suffix}")}
    else:
        split_files = {split: str((root_dir / f"{split}{suffix}").absolute())}
    dataset = load_dataset("json", data_files=split_files, split=split)
    return dataset


def rename_keys(d: dict, rename_map: dict) -> dict:
    return {rename_map.get(k, k): v for k, v in d.items()}


def conv1d_output_length(
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    output_length = (
        input_length + 2 * padding - dilation * (kernel_size - 1)
    ) // stride + 1
    return output_length


def get_conv1d_mod_output_length(conv1d: nn.Conv1d, input_length: int) -> int:
    def maybe_extract(x: tuple | int) -> int:
        if isinstance(x, tuple):
            return x[0]
        return x

    kwargs = {
        k: maybe_extract(conv1d.__dict__[k])
        for k in ("kernel_size", "stride", "padding", "dilation")
    }
    kwargs["input_length"] = input_length
    return conv1d_output_length(**kwargs)


def pad_waveform(
    waveform: np.array, length: int = 30, sampling_rate: int = 16_000
) -> np.array:
    """
    Pads an audio signal with silence.
    Args:
        waveform (np.array): Audio signal
        length (int): Length of padded signal in seconds
        sampling_rate (int): Sampling rate of provided signal
    """
    n_samples = waveform.shape[-1]
    padding_length = (length * sampling_rate) - n_samples
    if padding_length < 0:
        raise ValueError(f"Provided signal is longer than desired length.")
    padding_shape = waveform.shape[:-1] + (padding_length,)
    padding_values = np.zeros(shape=padding_shape, dtype=waveform.dtype)
    padded_waveform = np.concatenate((waveform, padding_values), axis=-1)
    return padded_waveform
