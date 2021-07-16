"""
Wrapper for cluster-based initialization methods and random initialization.
"""
import numpy as np
from scipy.special import softmax
from ._selection_methods import UNCERTAINTY_FUNCTIONS


def init_strategy(
    X,
    n_initial,
    clusterer=None,
    selection_method=None,
    random_state=None
):
    """
    Defaults to random.

    Selection method is only relevant if a clusterer object is passed.
    Possible selection methods:
    - None (default): random selection
    - centroid: Gets observations close to the centroids of
        the clusters.
    -
    """

    unlabeled_ids = np.indices(X.shape[:1]).squeeze()

    # Random selection
    if clusterer is None or selection_method in ['random', None]:
        rng = np.random.RandomState(random_state)
        ids = rng.choice(unlabeled_ids, n_initial, replace=False)
        return None, ids

    # Cluster-based selection
    clusterer.fit(X)

    if hasattr(clusterer, 'predict_proba'):
        # Use probabilities to compute uncertainty
        probs = clusterer.predict_proba(X)
    else:
        # Use cluster distances to compute probabilities
        dist = clusterer.transform(X)
        # The first one is another possible alternative
        # dist_inv = 1 - (dist / np.expand_dims(dist.max(1), 1))
        dist_inv = (np.expand_dims(dist.max(1), 1) / dist) - 1
        probs = softmax(dist_inv, axis=1)

    uncertainty = UNCERTAINTY_FUNCTIONS[selection_method](probs)
    ids = unlabeled_ids[np.argsort(uncertainty)[::-1][:n_initial]]

    return clusterer, ids
