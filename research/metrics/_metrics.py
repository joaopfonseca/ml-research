import numpy as np
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score


def geometric_mean_score_macro(y_true, y_pred):
    """Geometric mean score with macro average."""
    return geometric_mean_score(y_true, y_pred, average='macro')


def area_under_learning_curve(clf):
    """Area under the learning curve. Used in Active Learning experiments."""
    test_scores = clf.test_scores_
    auc = np.sum(test_scores) / len(test_scores)
    return auc


def data_utilization_rate(clf, threshold=.8):
    """Data Utilization Rate. Used in Active Learning Experiments."""
    test_scores = clf.test_scores_
    data_utilization = [
        i[1] for i in clf.data_utilization_
    ]
    indices = np.where(test_scores >= threshold)[0]
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
SCORERS['area_under_learning_curve'] = make_scorer(area_under_learning_curve)
SCORERS['data_utilization_rate'] = make_scorer(data_utilization_rate)
