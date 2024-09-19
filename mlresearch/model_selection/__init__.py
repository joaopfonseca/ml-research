"""
The :mod:`mlresearch.model_selection` module includes the
model and parameter search methods.
"""

from ._search import ModelSearchCV
from ._search_successive_halving import HalvingModelSearchCV

__all__ = ["ModelSearchCV", "HalvingModelSearchCV"]
