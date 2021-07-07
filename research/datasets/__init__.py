"""
Download, transform and simulate various datasets.

These classes were extracted from the `utils.py` script from AlgoWit's
publications repo, to which I have also contributed.

Link to related repo: https://github.com/AlgoWit/publications
"""

from ._base import (
    Datasets,
    ImbalancedBinaryDatasets,
    BinaryDatasets,
    ContinuousCategoricalDatasets,
    MulticlassDatasets,
    RemoteSensingDatasets
)

__all__ = [
    'Datasets',
    'ImbalancedBinaryDatasets',
    'BinaryDatasets',
    'ContinuousCategoricalDatasets',
    'MulticlassDatasets',
    'RemoteSensingDatasets'
]
