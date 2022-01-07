"""Testing active learning models."""
import pytest
import numpy as np
from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.utils._testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier

from ....data_augmentation import OverSamplingAugmentation, GeometricSMOTE
from .._acquisition_functions import ACQUISITION_FUNCTIONS
from .._active_learning import StandardAL, AugmentationAL

ACTIVE_LEARNERS = {"StandardAL": StandardAL, "AugmentationAL": AugmentationAL}

RANDOM_STATE = 42

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# Larger classification sample used for testing feature importances
X_large, y_large = datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=RANDOM_STATE,
)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(RANDOM_STATE)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also make a hastie_10_2 dataset
hastie_X, hastie_y = datasets.make_hastie_10_2(n_samples=20, random_state=RANDOM_STATE)
hastie_X = hastie_X.astype(np.float32)

DATASETS = [(X, y), (iris.data, iris.target), (hastie_X, hastie_y)]


@pytest.mark.parametrize("name", ACTIVE_LEARNERS.keys())
def test_classification_toy(name):
    """Test classification on a toy dataset."""
    al_model = ACTIVE_LEARNERS[name](random_state=RANDOM_STATE)
    al_model.fit(X, y)
    assert_array_equal(al_model.predict(T), true_result)
    assert al_model.random_state == RANDOM_STATE
    assert al_model.classifier_.random_state == RANDOM_STATE


@pytest.mark.parametrize("name", ACTIVE_LEARNERS.keys())
@pytest.mark.parametrize("X, y", DATASETS)
def test_default_parameters(name, X, y):
    """Test default parameters."""
    al_model = ACTIVE_LEARNERS[name]()
    al_model.fit(X, y)

    # Set up expected parameters
    exp_acquisition_func = ACQUISITION_FUNCTIONS["random"]
    exp_n_init = int(np.round(0.02 * np.array(X).shape[0]))
    exp_n_init = exp_n_init if exp_n_init >= 2 else 2
    exp_budget = int(np.round(0.02 * np.array(X).shape[0]))
    exp_budget = exp_budget if exp_budget >= 1 else 1
    exp_max_iter = int((np.array(X).shape[0] - exp_n_init) / exp_budget)

    assert type(al_model.classifier_) == RandomForestClassifier
    assert al_model.acquisition_func_ == exp_acquisition_func
    assert al_model.n_init_ == exp_n_init
    assert al_model.budget_ == exp_budget
    assert al_model.max_iter_ == exp_max_iter
    assert al_model.random_state is None
    assert al_model.continue_training is False


def test_augmentation_active_learning():
    """Test active learning with pipelined data augmentation and parameter tuning."""
    generator = OverSamplingAugmentation(GeometricSMOTE(n_jobs=-1))
    classifier = RandomForestClassifier(n_jobs=-1)
    al_model = AugmentationAL(
        classifier=classifier,
        generator=generator,
        param_grid={
            "generator__value": [1, 1.5, 2]
        },
        max_iter=2,
        random_state=RANDOM_STATE
    )
    al_model.fit(iris.data, iris.target)

    assert al_model.max_iter_ == 2
    assert al_model.random_state == RANDOM_STATE
    assert dict(
        al_model.classifier_.best_estimator_.steps
    )["generator"].random_state == RANDOM_STATE


@pytest.mark.parametrize("name", ACTIVE_LEARNERS.keys())
def test_classifier_metadata(name):
    al_model = ACTIVE_LEARNERS[name](random_state=RANDOM_STATE)
    al_model.fit(X, y, X_test=T, y_test=true_result)

    metadata = al_model.metadata_
    exp_metadata_keys = ['data', 'performance_metric', *range(5)]

    assert al_model._has_test
    assert list(metadata.keys()) == exp_metadata_keys