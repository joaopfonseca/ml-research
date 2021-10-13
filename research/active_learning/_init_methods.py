"""
Wrapper for cluster-based initialization methods and random initialization.
"""
import numpy as np
from somlearn import SOM
from imblearn.pipeline import Pipeline as imblearn_pipeline
from sklearn.pipeline import Pipeline as sklearn_pipeline


def init_strategy(
    X,
    y,
    n_initial,
    clusterer=None,
    init_strategy=None,
    selection_strategy=None,
    random_state=None
):
    """
    Defaults to random.

    Selection method is only relevant if a clusterer object is passed.
    Possible initialization strategies:
    - None (default): defaults to edge selection
    - centroid: Gets observations close to the centroids of
      the clusters.
    - edge: Gets observations close to the clusters' decision borders
    - hybrid: Some close to the centroid, others far
    """

    unlabeled_ids = np.indices(X.shape[:1]).squeeze()
    rng = np.random.RandomState(random_state)

    # Random selection
    if clusterer is None or init_strategy in ['random', None]:
        ids = rng.choice(unlabeled_ids, n_initial, replace=False)
        # There must be at least 2 different initial classes
        if len(np.unique(y[ids])) == 1:
            ids[-1] = rng.choice(unlabeled_ids[y != y[ids][0]], 1, replace=False)

        return None, ids

    # Cluster-based selection
    clusterer.fit(X)

    # SOM-based selection
    if type(clusterer) == SOM or (
        type(clusterer) in [imblearn_pipeline, sklearn_pipeline]
        and type(clusterer.steps[-2][-1]) == SOM
    ):
        if type(clusterer) == SOM:
            labels = clusterer.labels_
        else:
            labels = clusterer.steps[-1][-1].labels_

        ids = []
        for clust_id in np.unique(labels):
            ids_ = np.where(labels == clust_id)[0]
            ids.append(rng.choice(ids_))

        ids = np.array(ids)

        if len(ids) < n_initial:
            amount = n_initial - len(ids)
            new_ids = rng.choice(unlabeled_ids[~np.array(ids)], amount)
            ids = np.append(ids, new_ids)
        if len(ids) > n_initial:
            amount = len(ids) - n_initial
            ids = rng.choice(ids, n_initial)
        return clusterer, ids

    # Remaining clustering methods
    if hasattr(clusterer, 'predict_proba'):
        # Use probabilities to compute uncertainty
        probs = clusterer.predict_proba(X)
    else:
        # Use cluster distances to compute probabilities
        dist = clusterer.transform(X)

        # The first one is another possible alternative
        dist_inv = 1 - (dist / np.expand_dims(dist.max(1), 1))
        # dist_inv = (np.expand_dims(dist.max(1), 1) / dist) - 1
        # probs = softmax(dist_inv, axis=1)
        probs = dist_inv / np.expand_dims(dist_inv.sum(1), 1)

    # Some strategies don't deal well with zero values
    probs = np.where(probs == 0., 1e-10, probs)
    uncertainty = selection_strategy(probs)

    if init_strategy == 'edge' or init_strategy is None:
        ids = unlabeled_ids[np.argsort(uncertainty)[::-1][:n_initial]]

    elif init_strategy == 'centroid':  # This will have to be refactored later
        ids = unlabeled_ids[np.argsort(-uncertainty)[::-1][:n_initial]]

    elif init_strategy == 'hybrid':
        ids_edge = unlabeled_ids[np.argsort(uncertainty)[::-1][:n_initial]]
        ids_centroid = unlabeled_ids[np.argsort(-uncertainty)[::-1][:n_initial]]
        ids = rng.choice(np.concatenate([ids_edge, ids_centroid]), n_initial)

    # There must be at least 2 different initial classes
    if len(np.unique(y[ids])) == 1:
        ids[-1] = rng.choice(unlabeled_ids[y != y[ids][0]], 1, replace=False)

    return clusterer, ids
