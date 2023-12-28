"""
This module contains various performance metrics/scorers that are not
included in scikit-learn's scorers' dictionary. Additionally, an expanded
dictionary of scorers (as compared with scikit-learn's) is also provided.
"""
from ._metrics import (
    geometric_mean_score_macro,
    ALScorer,
    area_under_learning_curve,
    data_utilization_rate,
)
from ._rankings import (
    RankingScorer,
    precision_at_k,
)
from ._synth_data_quality import AlphaPrecision, BetaRecall, Authenticity

# Update the list of scorers
from sklearn.metrics import get_scorer, get_scorer_names

__all__ = [
    "get_scorer",
    "get_scorer_names",
    "geometric_mean_score_macro",
    "area_under_learning_curve",
    "data_utilization_rate",
    "precision_at_k",
    "ALScorer",
    "RankingScorer",
    "AlphaPrecision",
    "BetaRecall",
    "Authenticity",
]
