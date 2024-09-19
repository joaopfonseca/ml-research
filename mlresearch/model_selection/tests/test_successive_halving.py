import pytest
import numpy as np
from .test_search import (
    REGRESSORS,
    CLASSIFIERS,
    REGRESSORS_PARAM_GRIDS,
    CLASSIFIERS_PARAM_GRIDS,
    X_reg,
    y_reg,
    X_clf,
    y_clf,
)
from .._search_successive_halving import HalvingModelSearchCV


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
    mscv = HalvingModelSearchCV(estimators, param_grids, factor=2, min_resources=100)
    if estimator_type == "regressor":
        mscv.fit(X_reg, y_reg)
    elif estimator_type == "classifier":
        mscv.fit(X_clf, y_clf)

    iters = mscv.cv_results_["iter"]
    iters_idx = np.flatnonzero(iters == iters.max())
    est_names_res_ = mscv.cv_results_["param_est_name"][iters_idx]
    assert set(est_names) == set(est_names_res_)
