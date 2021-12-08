"""
Test the check_pipelines module.
"""
from itertools import product

import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from rlearn.utils import check_random_states

from research.utils import check_pipelines


def test_check_pipelines():
    """Test the check of oversamplers and classifiers."""

    # Initialization
    n_runs = 5
    rnd_seed = 0
    oversamplers = [
        ("ovs", BorderlineSMOTE(), [{"k_neighbors": [2, 4]}, {"m_neighbors": [6, 8]}])
    ]
    classifiers = [("clf", DecisionTreeClassifier(), {"max_depth": [3, 5]})]

    # Estimators and parameters grids
    estimators, param_grids = check_pipelines(
        [oversamplers, classifiers], rnd_seed, n_runs
    )
    names, pips = zip(*estimators)
    steps = [
        [(step[0], step[1].__class__.__name__) for step in pip.steps] for pip in pips
    ]

    # Expected estimators and parameters grids
    exp_name = "ovs|clf"
    exp_steps = [("ovs", "BorderlineSMOTE"), ("clf", "DecisionTreeClassifier")]
    exp_random_states = check_random_states(rnd_seed, n_runs)
    partial_param_grids = []
    for k_neighbors, max_depth in product([2, 4], [3, 5]):
        partial_param_grids.append(
            {
                "ovs|clf__ovs__k_neighbors": [k_neighbors],
                "ovs|clf__clf__max_depth": [max_depth],
            }
        )
    for m_neighbors, max_depth in product([6, 8], [3, 5]):
        partial_param_grids.append(
            {
                "ovs|clf__ovs__m_neighbors": [m_neighbors],
                "ovs|clf__clf__max_depth": [max_depth],
            }
        )
    exp_param_grids = []
    for rnd_seed, partial_param_grid in product(exp_random_states, partial_param_grids):
        partial_param_grid = partial_param_grid.copy()
        partial_param_grid.update(
            {
                "est_name": ["ovs|clf"],
                "ovs|clf__ovs__random_state": [rnd_seed],
                "ovs|clf__clf__random_state": [rnd_seed],
            }
        )
        exp_param_grids.append(partial_param_grid)

    # Assertions
    assert names[0] == exp_name
    assert steps[0] == exp_steps
    assert len(param_grids) == len(exp_param_grids)
    assert all([param_grid in exp_param_grids for param_grid in param_grids])


def test_check_oversamplers_classifiers_none():
    """Test the check of oversamplers and classifiers for no oversampler."""

    # Initialization
    n_runs = 2
    rnd_seed = 12
    oversamplers = [("ovs", None, {})]
    classifiers = [("clf", DecisionTreeClassifier(), {"max_depth": [3, 5, 8]})]

    # Estimators and parameters grids
    estimators, param_grids = check_pipelines(
        [oversamplers, classifiers], rnd_seed, n_runs
    )
    names, pips = zip(*estimators)
    steps = [
        [(step[0], step[1].__class__.__name__) for step in pip.steps] for pip in pips
    ]

    # Expected names, steps and parameter grids
    exp_name = "ovs|clf"
    exp_steps = [("ovs", "FunctionTransformer"), ("clf", "DecisionTreeClassifier")]
    exp_random_states = check_random_states(rnd_seed, n_runs)
    partial_param_grids = []
    for max_depth in [3, 5, 8]:
        partial_param_grids.append({"ovs|clf__clf__max_depth": [max_depth]})
    exp_param_grids = []
    for rnd_seed, partial_param_grid in product(exp_random_states, partial_param_grids):
        partial_param_grid = partial_param_grid.copy()
        partial_param_grid.update(
            {"est_name": ["ovs|clf"], "ovs|clf__clf__random_state": [rnd_seed]}
        )
        exp_param_grids.append(partial_param_grid)

    # Assertions
    assert names[0] == exp_name
    assert steps[0] == exp_steps
    assert len(param_grids) == len(exp_param_grids)
    assert all([param_grid in exp_param_grids for param_grid in param_grids])


def test_check_oversamplers_classifiers_pipeline():
    """Test the check of oversampler and classifiers pipelines."""

    # Initialization
    n_runs = 2
    rnd_seed = 3
    scalers = [
        ("scaler", MinMaxScaler(), {"scaler__feature_range": [(0, 1), (0, 0.5)]})
    ]
    oversamplers = [("ovs", SMOTE(), {"smote__k_neighbors": [3, 5]})]
    classifiers = [
        (
            "clf",
            Pipeline([("pca", PCA()), ("dtc", DecisionTreeClassifier())]),
            {"pca__n_components": [4, 8], "dtc__max_depth": [3, 5]},
        )
    ]

    # Estimators and parameters grids
    estimators, param_grids = check_pipelines(
        [scalers, oversamplers, classifiers], rnd_seed, n_runs
    )
    names, pips = zip(*estimators)
    steps = [
        [(step[0], step[1].__class__.__name__) for step in pip.steps] for pip in pips
    ]

    # Expected names, steps and parameter grids
    exp_name = "scaler|ovs|clf"
    exp_steps = [
        ("scaler", "MinMaxScaler"),
        ("smote", "SMOTE"),
        ("pca", "PCA"),
        ("dtc", "DecisionTreeClassifier"),
    ]
    exp_random_states = check_random_states(rnd_seed, n_runs)
    partial_param_grids = []
    for feature_range, k_neighbors, n_components, max_depth in product(
        [(0, 1), (0, 0.5)], [3, 5], [4, 8], [3, 5]
    ):
        partial_param_grids.append(
            {
                "scaler|ovs|clf__scaler__feature_range": [feature_range],
                "scaler|ovs|clf__smote__k_neighbors": [k_neighbors],
                "scaler|ovs|clf__pca__n_components": [n_components],
                "scaler|ovs|clf__dtc__max_depth": [max_depth],
            }
        )
    exp_param_grids = []
    for rnd_seed, partial_param_grid in product(exp_random_states, partial_param_grids):
        partial_param_grid = partial_param_grid.copy()
        partial_param_grid.update(
            {
                "est_name": ["scaler|ovs|clf"],
                "scaler|ovs|clf__smote__random_state": [rnd_seed],
                "scaler|ovs|clf__dtc__random_state": [rnd_seed],
                "scaler|ovs|clf__pca__random_state": [rnd_seed],
            }
        )
        exp_param_grids.append(partial_param_grid)

    # Assertions
    assert names[0] == exp_name
    assert steps[0] == exp_steps
    assert len(param_grids) == len(exp_param_grids)
    assert all([param_grid in exp_param_grids for param_grid in param_grids])
