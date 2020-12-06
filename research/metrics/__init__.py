"""
This submodule contains various performance metrics/scorers that are not
included in scikit-learn's scorers' dictionary. Additionally, an expanded
dictionary of scorers (as compared with scikit-learn's) is also provided.

Parts of this code was taken from the `utils.py` script from AlgoWit's
publications repo, to which I have also contributed.

Link to related repo: https://github.com/AlgoWit/publications
"""
from ._metrics import (
    SCORERS,
    geometric_mean_score_macro,
    ALScorer,
    area_under_learning_curve,
    data_utilization_rate
)

__all__ = [
    'SCORERS',
    'geometric_mean_score_macro',
    'area_under_learning_curve',
    'data_utilization_rate',
    'ALScorer'
]
