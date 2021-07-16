"""
Uncertainty estimators to be used along with the ALWrapper object. All functions must be
set up so that a higher value means higher uncertainty (higher likelihood of selection)
and vice-versa.
"""
import numpy as np


def breaking_ties(probabilities):
    """Breaking Ties uncertainty measurement. The output is scaled and reversed."""
    probs_sorted = np.sort(probabilities, axis=1)[:, ::-1]
    # The extra minus is redundant but I kept as a highlight of the change in the
    # original formula.
    bt = -(probs_sorted[:, 0] - probs_sorted[:, 1])
    return (bt - bt.min(0)) / (bt.max(0) - bt.min(0))


UNCERTAINTY_FUNCTIONS = {
    'entropy': (
        lambda probabilities: (-probabilities * np.log2(probabilities)).sum(axis=1)
    ),
    'breaking_ties': breaking_ties,
    'random': (
        lambda probabilities: np.ones(probabilities.shape[0]) / probabilities.shape[0]
    )
}
