from lnn.notebook import display_audio

from .attr_dict import AttrDict, TensorAttrDict
from .base import BaseConfig, ConfigurableLightningModule
from .pathlib import Path
from .slurm import scancel, squeue
from .utils import (
    aggregate_dfs,
    conv1d_output_length,
    count_params,
    dataframe_map,
    dataset_dict_to_ndjson,
    dataset_dict_to_tsv,
    extracts,
    flatten_dataset_dict,
    freeze_module,
    get_conv1d_mod_output_length,
    get_lang_code,
    gets,
    load_audio,
    load_dataset_split_from_disk,
    load_ndjson_dataset,
    load_tsv_dataset,
    normalize_lang_id,
    pad_waveform,
    rename_keys,
)
