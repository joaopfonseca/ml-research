"""
This module contains a variety of general utility functions and tools used to format and
prepare tables to incorporate into LaTeX code.
"""
from ._image import image_to_dataframe, dataframe_to_image
from ._data import load_datasets
from ._check_pipelines import check_pipelines, check_pipelines_wrapper, check_random_states
from ._visualization import set_matplotlib_style, feature_to_color
from ._parallelize import parallel_loop
from ._utils import generate_paths

__all__ = [
    "image_to_dataframe",
    "dataframe_to_image",
    "load_datasets",
    "check_pipelines",
    "check_pipelines_wrapper",
    "check_random_states",
    "set_matplotlib_style",
    "feature_to_color",
    "parallel_loop",
    "generate_paths",
]
