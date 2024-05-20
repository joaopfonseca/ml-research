"""
Test the search module.
"""

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification

from ...model_selection._search import (
    MultiEstimatorMixin,
    MultiRegressor,
    MultiClassifier,
    ModelSearchCV,
)


RND_SEED = 0
X_reg, y_reg = make_regression(random_state=RND_SEED)
X_clf, y_clf = make_classification(random_state=RND_SEED)
REGRESSORS = [
    ("lr", LinearRegression()),
    ("dtr", DecisionTreeRegressor()),
    ("pip", make_pipeline(StandardScaler(with_mean=False), LinearRegression())),
]
CLASSIFIERS = [
    ("lr", LogisticRegression()),
    ("dtc", DecisionTreeClassifier()),
    (
        "pip",
        make_pipeline(StandardScaler(with_mean=False), LogisticRegression()),
    ),
]
REGRESSORS_PARAM_GRIDS = [
    {"dtr__max_depth": [3, 5], "dtr__random_state": [RND_SEED, RND_SEED + 3]},
    {
        "pip__standardscaler__with_std": [True, False],
        "pip__linearregression__fit_intercept": [True, False],
    },
]
CLASSIFIERS_PARAM_GRIDS = {
    "dtc__max_depth": [3, 5],
    "dtc__criterion": ["entropy", "gini"],
    "dtc__random_state": [RND_SEED, RND_SEED + 5],
}


@pytest.mark.parametrize("estimators", [None, [], DecisionTreeClassifier()])
def test_multi_estimator_wrong_type(estimators):
    """Test the initialization of multi-estimator class with wrong inputs."""
    with pytest.raises(TypeError):
        MultiEstimatorMixin(estimators, "est").fit(X_clf, y_clf)


def test_multi_estimator_unique_names():
    """Test the initialization of multi-estimator class with duplicate names."""
    estimators = [("est", LinearRegression()), ("est", DecisionTreeRegressor())]
    with pytest.raises(ValueError):
        MultiEstimatorMixin(estimators, "estimator").fit(X_clf, y_clf)


def test_multi_estimator_wrong_name():
    """Test the initialization of multi-estimator class with wrong estimator name."""
    estimators = [("lr", LinearRegression()), ("dtr", DecisionTreeRegressor())]
    with pytest.raises(ValueError):
        MultiEstimatorMixin(estimators, "est").fit(X_clf, y_clf)


def test_multi_estimator_params_methods():
    """Test the set and get parameters methods."""

    # Get parameters
    est_name = "dtr"
    multi_estimator = MultiEstimatorMixin(REGRESSORS, est_name)
    params = multi_estimator.get_params(deep=False)
    assert params["est_name"] == "dtr"

    # Set parameters
    est_name = "reg"
    multi_estimator.set_params(est_name="reg")
    params = multi_estimator.get_params(deep=False)
    assert params["est_name"] == est_name


@pytest.mark.parametrize(
    "estimators,est_name,X,y",
    [
        (REGRESSORS, "lr", X_reg, y_reg),
        (REGRESSORS, "dtr", X_reg, y_reg),
        (CLASSIFIERS, "lr", X_clf, y_clf),
        (CLASSIFIERS, "dtc", X_clf, y_clf),
    ],
)
def test_multi_estimator_fitting(estimators, est_name, X, y):
    """Test multi-estimator fitting process."""
    multi_estimator = MultiEstimatorMixin(estimators, est_name)
    multi_estimator.fit(X, y)
    fitted_estimator = dict(estimators)[est_name]
    assert isinstance(fitted_estimator, multi_estimator.estimator_.__class__)
    assert fitted_estimator.get_params() == multi_estimator.estimator_.get_params()


@pytest.mark.parametrize(
    "estimators,X,y,est_name",
    [(REGRESSORS, X_reg, y_reg, "reg"), (CLASSIFIERS, X_clf, y_clf, None)],
)
def test_multi_estimator_fitting_error(estimators, X, y, est_name):
    """Test parametrized estimators fitting error."""
    with pytest.raises(ValueError):
        MultiEstimatorMixin(estimators, est_name).fit(X, y)


def test_multi_classifier_type():
    """Test multi-classifier type of estimator attribute."""
    multi_clf = MultiClassifier(CLASSIFIERS)
    assert multi_clf._estimator_type == "classifier"


def test_multi_regressor_type():
    """Test multi-regressor type of estimator attribute."""
    multi_reg = MultiRegressor(REGRESSORS)
    assert multi_reg._estimator_type == "regressor"


@pytest.mark.parametrize(
    "estimators,param_grids,estimator_type",
    [
        (REGRESSORS, REGRESSORS_PARAM_GRIDS, "regressor"),
        (CLASSIFIERS, CLASSIFIERS_PARAM_GRIDS, "classifier"),
    ],
)
def test_model_search_cv(estimators, param_grids, estimator_type):
    """Test model search cv."""
    est_names, *_ = zip(*estimators)
    mscv = ModelSearchCV(estimators, param_grids)
    if estimator_type == "regressor":
        mscv.fit(X_reg, y_reg)
    elif estimator_type == "classifier":
        mscv.fit(X_clf, y_clf)
    assert set(est_names) == set(mscv.cv_results_["param_est_name"])
