"""Toolbox to develop research in Machine Learning.

``research`` is a library containing the implementation of various algorithms developed
in Machine Learning research, as well as utilities to facilitate the formatting of pandas
dataframes into LaTeX tables.

Subpackages
-----------
active_learning
    Module which contains the code developed for experiments related to Active Learning.
data_augmentation
    Module which contains the implementation of variations of oversampling/data
    augmentation algorithms, as well as helper classes to use oversampling algorithms as
    data augmentation techniques.
datasets
    Module which contains code to download, transform and simulate various datasets.
metrics
    Module which contains performance metrics/scorers that are not
    included in scikit-learn's scorers' dictionary.
utils
    contains a variety of general utility functions and tools used to format and prepare
    tables to incorporate into LaTeX code.
"""
from . import active_learning
from . import data_augmentation
from . import datasets
from . import metrics
from . import utils

from ._version import __version__

__all__ = [
    "active_learning",
    "data_augmentation",
    "datasets",
    "metrics",
    "utils",
    "__version__",
]
