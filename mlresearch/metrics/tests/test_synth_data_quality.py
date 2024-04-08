import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from mlresearch.neural_network import OneClassMLP
from mlresearch.metrics import AlphaPrecision, BetaRecall, Authenticity


rng = check_random_state(42)
X = np.concatenate([rng.normal(size=(500, 1)), rng.normal(size=(500, 1))], axis=1)
X = MinMaxScaler().fit_transform(X)
X_synth = X.copy() + 0.1
alpha = beta = 0.05


def test_alpha_precision():
    mlp_real = OneClassMLP(max_iter=5, random_state=42)
    with ignore_warnings(category=ConvergenceWarning):
        mlp_real.fit(X)

    precision = AlphaPrecision(mlp_real.predict, alpha=alpha).fit(X)
    scores = precision.score(X_synth)

    # Test __repr__ bug due to undefined attributes (#67)
    assert precision

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (500,)
    assert (np.unique(scores) == np.array([0, 1])).all()


def test_beta_recall_with_scorer():
    mlp_synth = OneClassMLP(max_iter=5, random_state=42)
    with ignore_warnings(category=ConvergenceWarning):
        mlp_synth.fit(X)

    recall = BetaRecall(mlp_synth.predict, beta=beta).fit(X_synth)
    scores = recall.score(X)

    # Test __repr__ bug due to undefined attributes (#67)
    assert recall

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (500,)
    assert (np.unique(scores) == np.array([0, 1])).all()


def test_beta_recall_without_scorer():
    recall = BetaRecall(beta=beta).fit(X_synth)
    scores = recall.score(X)

    # Test __repr__ bug due to undefined attributes (#67)
    assert recall

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (500,)
    assert (np.unique(scores) == np.array([0, 1])).all()


def test_authenticity():
    authenticity = Authenticity().fit(X)
    scores = authenticity.score(X_synth)

    # Test __repr__ bug due to undefined attributes (#67)
    assert authenticity

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (500,)
    assert (np.unique(scores) == np.array([0, 1])).all()
