"""
Wrapper for cluster-based initialization methods and random initialization.
"""
import numpy as np
from scipy.special import softmax
from ._selection_methods import random, SELECTION_CRITERIA


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
    if clusterer is None:
        ids = random(
            unlabeled_ids=unlabeled_ids,
            increment=n_initial,
            random_state=random_state
        )
        return None, ids

    # Cluster-based selection
    if selection_method is None:
        # TODO
        # selection_method = 'centroid'
        pass

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

    ids = SELECTION_CRITERIA[selection_method](
        unlabeled_ids=unlabeled_ids,
        increment=n_initial,
        probabilities=probs,
        random_state=random_state
    )

    return clusterer, ids
