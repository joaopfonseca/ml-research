import pytest
import numpy as np
from sklearn.utils._testing import ignore_warnings

from .._metrics import geometric_mean_score_macro


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
    # Test G-mean behave with a single positive or negative class
    # Such a case may occur with non-stratified cross-validation
    assert 0.0 == geometric_mean_score_macro([1, 1], [1, 1])
    assert 0.0 == geometric_mean_score_macro([-1, -1], [-1, -1])


def test_geometric_mean_score_macro_multiclass():
    # Dense label indicator matrix format
    y1 = np.array([0, 1, 2, 0, 1, 2])
    y2 = np.array([0, 2, 1, 0, 0, 1])

    assert geometric_mean_score_macro(y1, y2) == pytest.approx(0.471, rel=1e-2)
