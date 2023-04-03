import numpy as np
from sklearn.metrics._scorer import _BaseScorer, _SCORERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import column_or_1d


class RankingScorer(_BaseScorer):
    """
    Make an Rank-based scorer from a probability-based metric or loss function.

    This factory class wraps scoring functions to be used in
    :class:`~rlearn.model_selection.ModelSearchCV` and
    :class:`~sklearn.model_selection.GridSearchCV`. It takes a score function, such as
    :func:`~mlresearch.metrics.precision_at_k` and is used to score a
    classifier. The signature of the call is `(estimator, X, y)` where `estimator` is the
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

    def __init__(self, score_func, sign=1, **kwargs):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs

    def _score(self, method_caller, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.
        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        sample_weight : array-like, default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_type = type_of_target(y)
        y_pred = method_caller(clf, "predict_proba", X)
        if y_type == "binary" and y_pred.shape[1] <= 2:
            # `y_type` could be equal to "binary" even in a multi-class
            # problem: (when only 2 class are given to `y_true` during scoring)
            # Thus, we need to check for the shape of `y_pred`.
            target_idx = (
                self._kwargs["target_label"]
                if "target_label" in self._kwargs.keys()
                else -1
            )
            y_pred = y_pred[:, target_idx]

        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


def precision_at_k(y_true, y_score, k=10, target_label=1):
    """
    Calculate precision at ``k``, where ``k`` is the number of relevant items to consider
    (sorted in descending order by its score). This metric consists of the ration between
    the number of items with label ``target_label``, out of the top ``k`` items with
    highest scores.

    .. warning::
        This metric is not the same as ``sklearn.metrics.top_k_accuracy_score``, which
        calculates the amount of times ``y_true`` is within the top ``k`` predicted
        classes for each item.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or non-thresholded
        decision values (as returned by :term:`decision_function` on some classifiers).
        Expects scores with shape (n_samples,).

    k : int, default=10
        Number of most likely predictions considered to compute the number of correct
        labels.

    target_label : int, default=1
        Value of the label with relevant items.
    """

    # Makes this compatible with various array types
    y_true_arr = column_or_1d(y_true)
    y_score_arr = column_or_1d(y_score)

    y_true_arr = y_true_arr == target_label
    top_idx = np.argsort(y_score_arr)[-k:]
    return y_true_arr[top_idx].sum() / k


_SCORERS["precision_at_10"] = RankingScorer(precision_at_k, k=10)
