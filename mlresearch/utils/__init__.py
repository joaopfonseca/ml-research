"""
This module contains a variety of general utility functions and tools used to format and
prepare tables to incorporate into LaTeX code.
"""
from ._utils import (
    generate_mean_std_tbl,
    generate_pvalues_tbl,
    sort_tbl,
    generate_paths,
    make_bold,
    generate_mean_std_tbl_bold,
)
from ._image import img_array_to_pandas
from ._data import load_datasets
from ._check_pipelines import check_pipelines, check_pipelines_wrapper
from ._visualization import load_plt_sns_configs, val_to_color

__all__ = [
    "generate_mean_std_tbl",
    "generate_pvalues_tbl",
    "sort_tbl",
    "generate_paths",
    "make_bold",
    "generate_mean_std_tbl_bold",
    "img_array_to_pandas",
    "load_datasets",
    "check_pipelines",
    "check_pipelines_wrapper",
    "load_plt_sns_configs",
    "val_to_color",
]
