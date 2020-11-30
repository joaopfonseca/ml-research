"""
This submodule contains the code developed for experiments related to Active
Learning.
"""

from ._al_wrapper import ALWrapper
from ._selection_methods import (
    entropy,
    breaking_ties,
    random,
    SELECTION_CRITERIA
)

__all__ = [
    'ALWrapper',
    'entropy',
    'breaking_ties',
    'random',
    'SELECTION_CRITERIA'
]
