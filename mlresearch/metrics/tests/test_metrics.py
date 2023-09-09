"""Test performance metrics"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>

import pytest
import numpy as np
from sklearn.utils._testing import ignore_warnings
from sklearn.linear_model import LogisticRegression

from mlresearch.active_learning import StandardAL
from mlresearch.metrics._metrics import (
    ALScorer,
    geometric_mean_score_macro,
    area_under_learning_curve,
    data_utilization_rate,
)
from mlresearch.metrics._rankings import RankingScorer, precision_at_k

RANDOM_STATE = 42

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]


def test_geometric_mean_score_macro_binary():
    # Dense label indicator matrix format
    y1 = np.array([0, 1, 1])
    y2 = np.array([0, 0, 1])

    assert geometric_mean_score_macro(y1, y2) == 0.75
    assert geometric_mean_score_macro(y1, y1) == 1
    assert geometric_mean_score_macro(y2, y2) == 1
    assert geometric_mean_score_macro(y2, np.logical_not(y2)) == 0
    assert geometric_mean_score_macro(y1, np.logical_not(y1)) == 0
    assert geometric_mean_score_macro(y1, np.zeros(y1.shape)) == 0.5
    assert geometric_mean_score_macro(y2, np.zeros(y1.shape)) == 0.5


@ignore_warnings
def test_geometric_mean_score_macro_single_class():
    # Test G-mean behavior with a single positive or negative class
    # Such a case may occur with non-stratified cross-validation
    assert 0.0 == geometric_mean_score_macro([1, 1], [1, 1])
    assert 0.0 == geometric_mean_score_macro([-1, -1], [-1, -1])


def test_geometric_mean_score_macro_multiclass():
    # Dense label indicator matrix format
    y1 = np.array([0, 1, 2, 0, 1, 2])
    y2 = np.array([0, 2, 1, 0, 0, 1])

    assert geometric_mean_score_macro(y1, y2) == pytest.approx(0.471, rel=1e-2)


def test_al_metrics():
    al_model = StandardAL(random_state=RANDOM_STATE)
    al_model.fit(X, y, X_test=T, y_test=true_result)

    assert ALScorer(data_utilization_rate)(al_model, None, None) == 1 / 3
    assert ALScorer(area_under_learning_curve)(al_model, None, None) == 1


def test_rank_metrics():
    y_score = [0.7, 0.1, 0.5]
    assert precision_at_k(true_result, y_score, k=1) == 0.0
    assert precision_at_k(true_result, y_score, k=2) == 0.5
    assert precision_at_k(true_result, y_score, k=3) == 2 / 3

    # Check if scorer is being created properly
    clf = LogisticRegression().fit(X, y)
    assert RankingScorer(precision_at_k, k=3)(clf, T, true_result) == 2 / 3
    assert RankingScorer(precision_at_k, k=2)(clf, T, true_result) == 1
