from collections import Counter
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
)

from .._oversampling_augmentation import _modify_nn, _clone_modify

RANDOM_STATE = 42
OVERSAMPLERS = [
    RandomOverSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
    BorderlineSMOTE(random_state=RANDOM_STATE),
    SVMSMOTE(random_state=RANDOM_STATE),
]


def test_modify_nn_object():
    """Test the modification of nn object."""
    assert _modify_nn(NearestNeighbors(n_neighbors=5), 3).n_neighbors == 2
    assert _modify_nn(NearestNeighbors(n_neighbors=3), 3).n_neighbors == 2
    assert _modify_nn(NearestNeighbors(n_neighbors=2), 5).n_neighbors == 2


def test_modify_nn_int():
    """Test the modification of integer nn."""
    assert _modify_nn(5, 3) == 2
    assert _modify_nn(3, 3) == 2
    assert _modify_nn(2, 5) == 2


def test_clone_modify_ros():
    """Test the cloning and modification of random oversampler."""
    cloned_oversampler = _clone_modify(OVERSAMPLERS[0], None)
    assert isinstance(cloned_oversampler, RandomOverSampler)


@pytest.mark.parametrize(
    'oversampler',
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_single_min_sample(oversampler):
    """Test the cloning and modification for one minority class sample."""
    y = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    cloned_oversampler = _clone_modify(oversampler, y)
    assert isinstance(cloned_oversampler, RandomOverSampler)


@pytest.mark.parametrize(
    'oversampler',
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_neighbors(oversampler):
    """Test the cloning and modification of neighbors based oversamplers."""
    y = np.array([0, 0, 0, 0, 2, 2, 2])
    n_minority_samples = Counter(y).most_common()[-1][1]
    cloned_oversampler = _clone_modify(oversampler, y)
    assert isinstance(cloned_oversampler, oversampler.__class__)
    if hasattr(cloned_oversampler, 'k_neighbors'):
        assert cloned_oversampler.k_neighbors == n_minority_samples - 1
    if hasattr(cloned_oversampler, 'm_neighbors'):
        assert (cloned_oversampler.m_neighbors == y.size - 1) or (
            cloned_oversampler.m_neighbors == 'deprecated'
        )
    if hasattr(cloned_oversampler, 'n_neighbors'):
        assert (cloned_oversampler.n_neighbors == n_minority_samples - 1) or (
            cloned_oversampler.n_neighbors == 'deprecated'
        )
