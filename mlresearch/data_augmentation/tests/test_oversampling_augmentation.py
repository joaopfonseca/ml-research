import pytest
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import ignore_warnings
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
)

from .._oversampling_augmentation import (
    _modify_nn,
    _clone_modify,
    OverSamplingAugmentation,
)

RANDOM_STATE = 42
OVERSAMPLERS = [
    RandomOverSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
    BorderlineSMOTE(random_state=RANDOM_STATE),
    SVMSMOTE(random_state=RANDOM_STATE),
]

X = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
y = np.array([0, 0, 1, 1, 1])


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
    "oversampler",
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_single_min_sample(oversampler):
    """Test the cloning and modification for one minority class sample."""
    y = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    cloned_oversampler = _clone_modify(oversampler, y)
    assert isinstance(cloned_oversampler, RandomOverSampler)


@pytest.mark.parametrize(
    "oversampler",
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_neighbors(oversampler):
    """Test the cloning and modification of neighbors based oversamplers."""
    y = np.array([0, 0, 0, 0, 2, 2, 2])
    n_minority_samples = Counter(y).most_common()[-1][1]
    cloned_oversampler = _clone_modify(oversampler, y)
    assert isinstance(cloned_oversampler, oversampler.__class__)
    if hasattr(cloned_oversampler, "k_neighbors"):
        assert cloned_oversampler.k_neighbors == n_minority_samples - 1
    if hasattr(cloned_oversampler, "m_neighbors"):
        assert (cloned_oversampler.m_neighbors == y.size - 1) or (
            cloned_oversampler.m_neighbors == "deprecated"
        )
    if hasattr(cloned_oversampler, "n_neighbors"):
        assert (cloned_oversampler.n_neighbors == n_minority_samples - 1) or (
            cloned_oversampler.n_neighbors == "deprecated"
        )


@pytest.mark.parametrize(
    "generator",
    [
        OverSamplingAugmentation(
            oversampler=SMOTE(k_neighbors=5),
            random_state=RANDOM_STATE,
        ),
        OverSamplingAugmentation(
            oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            augmentation_strategy="constant",
            value=10,
        ),
        OverSamplingAugmentation(
            oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            augmentation_strategy="proportional",
            value=10,
        ),
        OverSamplingAugmentation(
            oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            augmentation_strategy=2,
        ),
        OverSamplingAugmentation(
            oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            augmentation_strategy={0: 6, 1: 10},
        ),
    ],
)
@ignore_warnings
def test_fit_resample(generator):
    """Test the fit_resample method for various
    cases and data generator."""
    n_exp_obs = {
        "oversampling": {0: 3, 1: 3},
        "constant": {0: 10, 1: 10},
        "proportional": {0: 4, 1: 6},
        "2": {0: 4, 1: 6},
        "{0: 6, 1: 10}": {0: 6, 1: 10},
    }

    X_res, y_res = generator.fit_resample(X, y)
    y_count = dict(Counter(y_res))
    assert y_count == n_exp_obs[str(generator.augmentation_strategy)]


def test_errors():
    oversampler = OverSamplingAugmentation(
        oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        augmentation_strategy="proportional",
        value=2,
    )
    err_msg = (
        "The new size of the augmented dataset must be larger than the original "
        + "dataset. Originally, there are 5 samples and 2 samples are asked."
    )
    with pytest.raises(ValueError, match=err_msg):
        oversampler.fit_resample(X, y)
