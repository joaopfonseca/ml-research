"""
Acquisition functions that may be used along with Active Learning objects. All functions
must be set up so that a higher value means higher uncertainty (higher likelihood of
selection) and vice-versa.
"""
import numpy as np


def breaking_ties(probabilities):
    """Breaking Ties uncertainty measurement. The output is scaled and reversed."""
    probs_sorted = np.sort(probabilities, axis=1)[:, ::-1]
    # The extra minus is redundant but I kept as a highlight of the change in the
    # original formula.
    bt = -(probs_sorted[:, 0] - probs_sorted[:, 1])
    return bt


def entropy(probabilities):
    """Shannon's Entropy-based uncertainty measurement."""
    return (-probabilities * np.log2(probabilities)).sum(axis=1)


def random(probabilities):
    """Random data selection. The uncertainty is the same for all observations."""
    return np.ones(probabilities.shape[0]) / probabilities.shape[0]


ACQUISITION_FUNCTIONS = {
    "entropy": entropy,
    "breaking_ties": breaking_ties,
    "random": random,
}
