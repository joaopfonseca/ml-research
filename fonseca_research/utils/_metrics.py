from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score


def geometric_mean_score_macro(y_true, y_pred):
    """Geometric mean score with macro average."""
    return geometric_mean_score(y_true, y_pred, average='macro')


SCORERS['geometric_mean_score_macro'] = make_scorer(geometric_mean_score_macro)
