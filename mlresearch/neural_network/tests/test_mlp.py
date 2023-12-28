"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

import re
import sys
import warnings
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.datasets import (
    load_digits,
    load_iris,
    make_regression,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.neural_network._multilayer_perceptron import _STOCHASTIC_SOLVERS

from mlresearch.neural_network import OneClassMLP

ACTIVATION_TYPES = ["identity", "logistic", "tanh", "relu"]

X_digits, y_digits = load_digits(n_class=3, return_X_y=True)

X_digits_multi = MinMaxScaler().fit_transform(X_digits[:200])
y_digits_multi = y_digits[:200]

X_digits, y_digits = load_digits(n_class=2, return_X_y=True)

X_digits_binary = MinMaxScaler().fit_transform(X_digits[:200])
y_digits_binary = y_digits[:200]

classification_datasets = [
    (X_digits_multi, y_digits_multi),
    (X_digits_binary, y_digits_binary),
]

X_reg, y_reg = make_regression(
    n_samples=200, n_features=10, bias=20.0, noise=100.0, random_state=7
)
y_reg = scale(y_reg)
regression_datasets = [(X_reg, y_reg)]

iris = load_iris()

X_iris = iris.data
y_iris = iris.target


def test_alpha():
    # Test that larger alpha yields weights closer to zero
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]

    alpha_vectors = []
    alpha_values = np.arange(2)
    absolute_sum = lambda x: np.sum(np.abs(x))  # noqa

    for alpha in alpha_values:
        mlp = OneClassMLP(hidden_layer_sizes=(10, 10), alpha=alpha, random_state=1)
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
        alpha_vectors.append(
            np.array([absolute_sum(mlp.coefs_[0]), absolute_sum(mlp.coefs_[1])])
        )

    for i in range(len(alpha_values) - 1):
        assert (alpha_vectors[i] > alpha_vectors[i + 1]).all()


@pytest.mark.parametrize("X,y", classification_datasets)
def test_lbfgs_classification_maxfun(X, y):
    # Test lbfgs parameter max_fun.
    # It should independently limit the number of iterations for lbfgs.
    max_fun = 10
    # classification tests
    for activation in ACTIVATION_TYPES:
        mlp = OneClassMLP(
            solver="lbfgs",
            hidden_layer_sizes=(50, 50),
            max_iter=150,
            max_fun=max_fun,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            assert max_fun >= mlp.n_iter_


def test_learning_rate_warmstart():
    # Tests that warm_start reuse past solutions.
    X = [[3, 2], [1, 6], [5, 6], [-2, -4]]
    y = [1, 1, 1, 0]
    for learning_rate in ["invscaling", "constant"]:
        mlp = OneClassMLP(
            solver="sgd",
            hidden_layer_sizes=4,
            learning_rate=learning_rate,
            batch_size=4,
            max_iter=1,
            power_t=0.25,
            warm_start=True,
        )
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
            prev_eta = mlp._optimizer.learning_rate
            mlp.fit(X, y)
            post_eta = mlp._optimizer.learning_rate

        if learning_rate == "constant":
            assert prev_eta == post_eta
        elif learning_rate == "invscaling":
            assert mlp.learning_rate_init / pow(8 + 1, mlp.power_t) == post_eta


def test_partial_fit():
    # Test partial_fit
    # `partial_fit` should yield the same results as 'fit'
    X = X_reg
    y = y_reg

    for momentum in [0, 0.9]:
        mlp = OneClassMLP(
            solver="sgd",
            max_iter=100,
            n_iter_no_change=100,
            activation="identity",
            random_state=1,
            learning_rate_init=1e-05,
            batch_size=X.shape[0],
            momentum=momentum,
            verbose=1,
        )
        with warnings.catch_warnings(record=True):
            # catch convergence warning
            mlp.fit(X, y)
        pred1 = mlp.predict(X)

        mlp = OneClassMLP(
            solver="sgd",
            activation="identity",
            learning_rate_init=1e-05,
            random_state=1,
            batch_size=X.shape[0],
            momentum=momentum,
        )
        for i in range(100):
            mlp.partial_fit(X, y)

        pred2 = mlp.predict(X)
        assert_allclose(pred1, pred2)


def test_partial_fit_errors():
    # lbfgs doesn't support partial_fit
    assert not hasattr(OneClassMLP(solver="lbfgs"), "partial_fit")


@pytest.mark.filterwarnings("ignore")
def test_nonfinite_params():
    # Check that OneClassMLP throws ValueError when dealing with non-finite
    # parameter values
    rng = np.random.RandomState(0)
    n_samples = 10
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))

    mlp = OneClassMLP()
    msg = (
        "Solver produced non-finite parameter weights. The input data may contain large"
        " values and need to be preprocessed."
    )
    with pytest.raises(ValueError, match=msg):
        mlp.fit(X)


@pytest.mark.filterwarnings("ignore")
def test_shuffle():
    # Test that the shuffle parameter affects the training process (it should)
    X, y = make_regression(n_samples=50, n_features=5, n_targets=1, random_state=0)

    # The coefficients will be identical if both do or do not shuffle
    for shuffle in [True, False]:
        mlp1 = OneClassMLP(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        mlp2 = OneClassMLP(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        mlp1.fit(X, y)
        mlp2.fit(X, y)

        assert np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])

    # The coefficients will be slightly different if shuffle=True
    mlp1 = OneClassMLP(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=True
    )
    mlp2 = OneClassMLP(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=False
    )
    mlp1.fit(X, y)
    mlp2.fit(X, y)

    assert not np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])


def test_tolerance():
    # Test tolerance.
    # It should force the solver to exit the loop when it converges.
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = OneClassMLP(tol=0.5, batch_size=2, max_iter=3000, solver="sgd")
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_


def test_verbose_sgd():
    # Test verbose.
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = OneClassMLP(
        solver="sgd", batch_size=2, max_iter=2, verbose=10, hidden_layer_sizes=2
    )
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()

    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    clf.partial_fit(X, y)

    sys.stdout = old_stdout
    assert "Iteration" in output.getvalue()


def test_adaptive_learning_rate():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = OneClassMLP(
        hidden_layer_sizes=1,
        max_iter=3000,
        solver="sgd",
        learning_rate="adaptive",
        batch_size=2,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_
    assert 1e-6 > clf._optimizer.learning_rate


@ignore_warnings(category=ConvergenceWarning)
def test_warm_start_full_iteration():
    # Check that the MLP estimator accomplish `max_iter` with a
    # warm started estimator.
    X, y = X_iris, y_iris
    max_iter = 3
    clf = OneClassMLP(
        hidden_layer_sizes=2, solver="sgd", warm_start=True, max_iter=max_iter
    )
    clf.fit(X, y)
    assert max_iter == clf.n_iter_
    clf.fit(X, y)
    assert max_iter == clf.n_iter_


def test_n_iter_no_change():
    # test n_iter_no_change using binary data set
    # the classifying fitting process is not prone to loss curve fluctuations
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.01
    max_iter = 3000

    # test multiple n_iter_no_change
    for n_iter_no_change in [2, 5, 10, 50, 100]:
        clf = OneClassMLP(
            tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
        )
        clf.fit(X, y)

        # validate n_iter_no_change
        assert clf._no_improvement_count == n_iter_no_change + 1
        assert max_iter > clf.n_iter_


@ignore_warnings(category=ConvergenceWarning)
def test_n_iter_no_change_inf():
    # test n_iter_no_change using binary data set
    # the fitting process should go to max_iter iterations
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]

    # set a ridiculous tolerance
    # this should always trigger _update_no_improvement_count()
    tol = 1e9

    # fit
    n_iter_no_change = np.inf
    max_iter = 300
    clf = OneClassMLP(
        tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
    )
    clf.fit(X, y)

    # validate n_iter_no_change doesn't cause early stopping
    assert clf.n_iter_ == max_iter

    # validate _update_no_improvement_count() was always triggered
    assert clf._no_improvement_count == clf.n_iter_ - 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mlp_param_dtypes(dtype):
    # Checks if input dtype is used for network parameters
    # and predictions
    X, y = X_digits.astype(dtype), y_digits
    mlp = OneClassMLP(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    mlp.fit(X[:300], y[:300])
    pred = mlp.predict(X[300:])

    assert all([intercept.dtype == dtype for intercept in mlp.intercepts_])

    assert all([coef.dtype == dtype for coef in mlp.coefs_])

    assert pred.dtype == dtype


def test_preserve_feature_names():
    """Check that feature names are preserved when early stopping is enabled.

    Feature names are required for consistency checks during scoring.

    Non-regression test for gh-24846
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)

    X = pd.DataFrame(data=rng.randn(10, 2), columns=["colname_a", "colname_b"])
    y = pd.Series(data=np.full(10, 1), name="colname_y")

    model = OneClassMLP(batch_size=10, random_state=rng)

    # If a warning is raised, raise an error instead
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)

    # Raise an error if a warning is not raised
    with pytest.warns(UserWarning):
        model.predict(X.values)


@pytest.mark.parametrize("solver", _STOCHASTIC_SOLVERS)
def test_mlp_warm_start_no_convergence(solver):
    """Check that we stop the number of iteration at `max_iter` when warm starting.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24764
    """
    model = OneClassMLP(
        solver=solver,
        warm_start=True,
        max_iter=10,
        n_iter_no_change=np.inf,
        random_state=0,
    )

    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    assert model.n_iter_ == 10

    model.set_params(max_iter=20)
    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    assert model.n_iter_ == 20


def test_hidden_layer_sizes():
    layers = [10, 0]
    model = OneClassMLP(hidden_layer_sizes=layers)
    msg = re.escape(f"hidden_layer_sizes must be > 0, got {layers}.")
    with pytest.raises(ValueError, match=msg):
        model.fit(X_iris)
