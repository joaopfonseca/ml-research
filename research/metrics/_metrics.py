import numpy as np
from sklearn.metrics import SCORERS, make_scorer
from sklearn.metrics._scorer import _PredictScorer
from imblearn.metrics import geometric_mean_score


class ALScorer(_PredictScorer):
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

        return self._sign * self._score_func(estimator)


def geometric_mean_score_macro(y_true, y_pred):
    """Geometric mean score with macro average."""
    return geometric_mean_score(y_true, y_pred, average='macro')


def area_under_learning_curve(estimator):
    """Area under the learning curve. Used in Active Learning experiments."""
    test_scores = estimator.test_scores_
    auc = np.sum(test_scores) / len(test_scores)
    return auc


def data_utilization_rate(estimator, threshold=.8):
    """Data Utilization Rate. Used in Active Learning Experiments."""
    test_scores = estimator.test_scores_
    data_utilization = [
        i[1] for i in estimator.data_utilization_
    ]
    indices = np.where(np.array(test_scores) >= threshold)[0]
    arg = (
        indices[0]
        if len(indices) != 0
        else -1
    )
    dur = (
        data_utilization[arg]
        if arg != -1
        else 1
    )
    return dur


SCORERS['geometric_mean_score_macro'] = make_scorer(geometric_mean_score_macro)
SCORERS['area_under_learning_curve'] = ALScorer(area_under_learning_curve)
SCORERS['data_utilization_rate'] = ALScorer(data_utilization_rate)
