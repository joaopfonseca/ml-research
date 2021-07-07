"""
This submodule contains the code developed for experiments related to Active
Learning.
"""

from ._active_learning import ALWrapper
from ._selection_methods import (
    entropy,
    breaking_ties,
    random,
    SELECTION_CRITERIA
)
from ._generator import OverSamplingAugmentation

__all__ = [
    'ALWrapper',
    'OverSamplingAugmentation',
    'entropy',
    'breaking_ties',
    'random',
    'SELECTION_CRITERIA'
]
