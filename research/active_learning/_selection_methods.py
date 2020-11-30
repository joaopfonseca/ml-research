"""
Selection criteria to be used along with the ALWrapper object.
"""
import numpy as np


def entropy(unlabeled_ids, increment, probabilities, **kwargs):
    e = (-probabilities * np.log2(probabilities)).sum(axis=1)
    new_ids = unlabeled_ids[np.argsort(e)[::-1][:increment]]
    return new_ids


def breaking_ties(unlabeled_ids, increment, probabilities, **kwargs):
    """
    Selecting samples as a smallest difference of probability values
    between the first and second most likely classes
    """
    probs_sorted = np.sort(probabilities, axis=1)[:, ::-1]
    values = probs_sorted[:, 0] - probs_sorted[:, 1]
    new_ids = unlabeled_ids[np.argsort(values)[:increment]]
    return new_ids


def random(unlabeled_ids, increment, random_state=None, **kwargs):
    """
    Random sample selection.
    """
    rng = np.random.RandomState(random_state)
    new_ids = rng.choice(unlabeled_ids, increment, replace=False)
    return new_ids


SELECTION_CRITERIA = dict(
    entropy=entropy,
    breaking_ties=breaking_ties,
    random=random
)
