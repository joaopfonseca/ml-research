"""
Test the check_pipelines module.
"""
from itertools import product

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from rlearn.model_selection import ModelSearchCV

from .._check_pipelines import (
    check_pipelines,
    check_pipelines_wrapper,
    check_random_states,
)
from ...active_learning import StandardAL
from ...synthetic_data import OverSamplingAugmentation


def test_check_pipeline_single():
    """Test the check of pipelines with a single element."""

    # Initialization
    n_runs = 5
    rnd_seed = 0
    classifiers = [("clf", DecisionTreeClassifier(), {"max_depth": [3, 5]})]

    # Estimators and parameters grids
    estimators, param_grids = check_pipelines(
        classifiers, random_state=rnd_seed, n_runs=n_runs
    )
    names, pips = zip(*estimators)
    steps = [
        [(step[0], step[1].__class__.__name__) for step in pip.steps] for pip in pips
    ]

    # Expected estimators and parameters grids
    exp_name = "clf"
    exp_steps = [("clf", "DecisionTreeClassifier")]
    exp_random_states = check_random_states(rnd_seed, n_runs)
    partial_param_grids = []
    for max_depth in [3, 5]:
        partial_param_grids.append(
            {
                "clf__clf__max_depth": [max_depth],
            }
        )

    exp_param_grids = []
    for rnd_seed, partial_param_grid in product(exp_random_states, partial_param_grids):
        partial_param_grid = partial_param_grid.copy()
        partial_param_grid.update(
            {
                "est_name": ["clf"],
                "clf__clf__random_state": [rnd_seed],
            }
        )
        exp_param_grids.append(partial_param_grid)

    # Assertions
    assert names[0] == exp_name
    assert steps[0] == exp_steps
    assert len(param_grids) == len(exp_param_grids)
    assert all([param_grid in exp_param_grids for param_grid in param_grids])


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
        oversamplers, classifiers, random_state=rnd_seed, n_runs=n_runs
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
        oversamplers, classifiers, random_state=rnd_seed, n_runs=n_runs
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
    """Test the check of pipelines with higher lengths."""

    # Initialization
    n_runs = 2
    rnd_seed = 3
    scalers = [("scaler", MinMaxScaler(), {"feature_range": [(0, 1), (0, 0.5)]})]
    oversamplers = [("ovs", SMOTE(), {"k_neighbors": [3, 5]})]
    classifiers = [
        (
            "clf",
            Pipeline([("pca", PCA()), ("dtc", DecisionTreeClassifier())]),
            {"pca__n_components": [4, 8], "dtc__max_depth": [3, 5]},
        )
    ]

    # Estimators and parameters grids
    estimators, param_grids = check_pipelines(
        scalers, oversamplers, classifiers, random_state=rnd_seed, n_runs=n_runs
    )
    names, pips = zip(*estimators)
    steps = [
        [(step[0], step[1].__class__.__name__) for step in pip.steps] for pip in pips
    ]

    # Expected names, steps and parameter grids
    exp_name = "scaler|ovs|clf"
    exp_steps = [
        ("scaler", "MinMaxScaler"),
        ("ovs", "SMOTE"),
        ("clf", "Pipeline"),
    ]
    exp_random_states = check_random_states(rnd_seed, n_runs)
    partial_param_grids = []
    for feature_range, k_neighbors, n_components, max_depth in product(
        [(0, 1), (0, 0.5)], [3, 5], [4, 8], [3, 5]
    ):
        partial_param_grids.append(
            {
                "scaler|ovs|clf__scaler__feature_range": [feature_range],
                "scaler|ovs|clf__ovs__k_neighbors": [k_neighbors],
                "scaler|ovs|clf__clf__pca__n_components": [n_components],
                "scaler|ovs|clf__clf__dtc__max_depth": [max_depth],
            }
        )
    exp_param_grids = []
    for rnd_seed, partial_param_grid in product(exp_random_states, partial_param_grids):
        partial_param_grid = partial_param_grid.copy()
        partial_param_grid.update(
            {
                "est_name": ["scaler|ovs|clf"],
                "scaler|ovs|clf__ovs__random_state": [rnd_seed],
                "scaler|ovs|clf__clf__dtc__random_state": [rnd_seed],
                "scaler|ovs|clf__clf__pca__random_state": [rnd_seed],
            }
        )
        exp_param_grids.append(partial_param_grid)

    # Assertions
    assert names[0] == exp_name
    assert steps[0] == exp_steps
    assert len(param_grids) == len(exp_param_grids)
    assert all([param_grid in exp_param_grids for param_grid in param_grids])


def test_check_pipelines_wrapper():
    """Based on the parameter keys error found in the experiment of a working paper."""

    # Initialization
    X, y = load_iris(return_X_y=True)
    n_runs = 1
    rnd_seed = 0
    oversamplers = [
        (
            "ovs",
            OverSamplingAugmentation(BorderlineSMOTE()),
            [
                {"oversampler__k_neighbors": [2, 4]},
                {"oversampler__m_neighbors": [6, 8]},
            ],
        )
    ]
    classifiers = [("clf", DecisionTreeClassifier(), {"max_depth": [3, 5]})]
    al_model = (
        "AL-TEST",
        StandardAL(max_iter=2),
        {"acquisition_func": ["random", "entropy", "breaking_ties"]},
    )

    we_wpg = check_pipelines_wrapper(
        classifiers,
        wrapper=al_model,
        random_state=rnd_seed,
        n_runs=n_runs,
        wrapped_only=True,
    )

    we_wpg2 = check_pipelines_wrapper(
        oversamplers,
        classifiers,
        wrapper=al_model,
        random_state=rnd_seed,
        n_runs=n_runs,
        wrapped_only=True,
    )
    for we, wpg in [we_wpg, we_wpg2]:
        ModelSearchCV(estimators=we, cv=2, param_grids=wpg, n_jobs=-1).fit(X, y)
