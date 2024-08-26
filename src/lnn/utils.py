import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
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


def extracts(d: dict, keys: list) -> dict:
    return {k: d[k] for k in keys}


def gets(d: dict, keys: list) -> list:
    return [d[k] for k in keys]


def rename_keys(d: dict, rename_map: dict) -> dict:
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
        raise ValueError("Provided signal is longer than desired length.")
    padding_shape = waveform.shape[:-1] + (padding_length,)
    padding_values = np.zeros(shape=padding_shape, dtype=waveform.dtype)
    padded_waveform = np.concatenate((waveform, padding_values), axis=-1)
    return padded_waveform


def load_audio(
    path: str | Path,
    rate: Optional[int] = None,
    mono: bool = True,
    return_tensor: str = Literal["pt", "np", "py"],
) -> tuple[torch.Tensor | np.ndarray | list, int]:
    wv, orig_rate = torchaudio.load(path)
    if mono and wv.size(0) == 2:
        wv = wv.mean(axis=0).reshape(1, -1)
    if rate is not None and rate != orig_rate:
        wv = torchaudio.functional.resample(wv, orig_freq=orig_rate, new_freq=rate)
    elif rate is None:
        rate = orig_rate
    if return_tensor == "np":
        wv = wv.numpy()
    elif return_tensor == "py":
        wv = wv.tolist()
    return wv, rate


def dataframe_map(
    df: pd.DataFrame,
    apply_column: str,
    func: callable,
    result_column: str,
    num_proc: int = os.cpu_count(),
    batched: bool = True,
    batch_size: int = 1000,
    func_is_batched: bool = False,
):
    ds = Dataset.from_pandas(df)
    if func_is_batched:

        def map_fn(examples):
            targets = examples[apply_column]
            results = func(targets)
            examples[result_column] = results
            return examples

    else:

        def map_fn(examples):
            targets = examples[apply_column]
            results = [func(entry) for entry in targets]
            examples[result_column] = results
            return examples

    ds = ds.map(map_fn, num_proc=num_proc, batched=batched, batch_size=batch_size)
    df = ds.to_pandas()
    return df


def aggregate_dfs(dfs: list[pd.DataFrame], op: str = "mean") -> pd.DataFrame:
    df = dfs[0]
    columns = df.columns
    index = df.index
    data = np.stack([df.to_numpy() for df in dfs])
    op_fn = getattr(np, op)
    data_aggregated = op_fn(data, axis=0)
    averaged_df = pd.DataFrame(index=index, columns=columns, data=data_aggregated)
    return averaged_df
