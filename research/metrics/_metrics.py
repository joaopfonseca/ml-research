import numpy as np
from sklearn.metrics import SCORERS, make_scorer
from sklearn.metrics._scorer import _PredictScorer
from imblearn.metrics import geometric_mean_score
from rlearn.model_selection.search import MultiClassifier


class ALScorer(_PredictScorer):
    """
    Make an Active Learning scorer from a AL-specific metric or loss function.

    This factory class wraps scoring functions for use in
    :class:`~rlearn.model_selection.ModelSearchCV` and
    :class:`~sklearn.model_selection.GridSearchCV`. It takes a score function, such as
    :func:`~research.metrics.area_under_learning_curve` or
    :func:`~research.metrics.data_utilization_rate` and is used to score an AL
    simulation. The signature of the call is `(estimator, X, y)` where `estimator` is the
    model to be evaluated, `X` is the data and `y` is the ground truth labeling (or
    `None` in the case of unsupervised models).

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    sign : int, default=1
        Use 1 to keep the original variable's scale, use -1 to reverse the scale.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score.
    """
    def __init__(self, score_func, sign=1):
        super().__init__(score_func, sign, {})

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if type(estimator) == MultiClassifier:
            (
                data_utilization,
                test_scores,
            ) = estimator.estimator_._get_performance_scores()
        else:
            data_utilization, test_scores = estimator._get_performance_scores()

        return self._sign * self._score_func(test_scores, data_utilization)


def geometric_mean_score_macro(y_true, y_pred):
    """Geometric mean score with macro average."""
    return geometric_mean_score(y_true, y_pred, average="macro")


def area_under_learning_curve(test_scores, *args):
    """Area under the learning curve. Used in Active Learning experiments."""
    auc = np.sum(test_scores) / len(test_scores)
    return auc


def data_utilization_rate(test_scores, data_utilization, threshold=0.8):
    """Data Utilization Rate. Used in Active Learning Experiments."""
    indices = np.where(np.array(test_scores) >= threshold)[0]
    arg = indices[0] if len(indices) != 0 else -1
    dur = data_utilization[arg] if arg != -1 else 1
    return dur


SCORERS["geometric_mean_score_macro"] = make_scorer(geometric_mean_score_macro)
SCORERS["area_under_learning_curve"] = ALScorer(area_under_learning_curve)
SCORERS["data_utilization_rate"] = ALScorer(data_utilization_rate)
