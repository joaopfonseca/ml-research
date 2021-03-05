"""
Selection criteria to be used along with the ALWrapper object.
"""
import numpy as np


def entropy(unlabeled_ids, increment, probabilities, **kwargs):
    """
    Sample selection based on Entropy selection criterion.

    Parameters
    ----------
    unlabeled_ids : array-like of shape (n_samples,)
        Indices of the unlabeled samples in the original (unlabeled training)
        dataset.

    increment : int
        Number of observations to select.

    probabilities : array-like of shape (n_samples, n_classes)
        Class probabilities of the input samples belonging to the unlabeled
        dataset.

    Returns
    -------
    new_ids : array of shape (increment,)
        Indices of unlabeles samples to be added to the labeled training
        dataset.
    """
    e = (-probabilities * np.log2(probabilities)).sum(axis=1)
    new_ids = unlabeled_ids[np.argsort(e)[::-1][:increment]]
    return new_ids


def breaking_ties(unlabeled_ids, increment, probabilities, **kwargs):
    """
    Sample selection based on breaking ties selection criterion.

    Selects samples as a smallest difference of probability values
    between the first and second most likely classes

    Parameters
    ----------
    unlabeled_ids : array-like of shape (n_samples,)
        Indices of the unlabeled samples in the original (unlabeled training)
        dataset.

    increment : int
        Number of observations to select.

    probabilities : array-like of shape (n_samples, n_classes)
        Class probabilities of the input samples belonging to the unlabeled
        dataset.

    Returns
    -------
    new_ids : array of shape (increment,)
        Indices of unlabeles samples to be added to the labeled training
        dataset.
    """
    probs_sorted = np.sort(probabilities, axis=1)[:, ::-1]
    values = probs_sorted[:, 0] - probs_sorted[:, 1]
    new_ids = unlabeled_ids[np.argsort(values)[:increment]]
    return new_ids


def random(unlabeled_ids, increment, random_state=None, **kwargs):
    """
    Random sample selection.

    Parameters
    ----------
    unlabeled_ids : array-like of shape (n_samples,)
        Indices of the unlabeled samples in the original (unlabeled training)
        dataset.

    increment : int
        Number of observations to select.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    Returns
    -------
    new_ids : array of shape (increment,)
        Indices of unlabeles samples to be added to the labeled training
        dataset.
    """
    rng = np.random.RandomState(random_state)
    new_ids = rng.choice(unlabeled_ids, increment, replace=False)
    return new_ids


SELECTION_CRITERIA = dict(
    entropy=entropy,
    breaking_ties=breaking_ties,
    random=random
)
