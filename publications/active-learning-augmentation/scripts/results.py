"""
Generate the main experimental results.
"""

# Author: Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import os
from os.path import join
from itertools import product
from zipfile import ZipFile
from rich.progress import track
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.base import SamplerMixin
from rlearn.model_selection import ModelSearchCV
from research.active_learning import ALSimulation
from research.utils import (
    generate_paths,
    load_datasets
)
from research.metrics import (
    data_utilization_rate,
    ALScorer,
    SCORERS
)
from research.data_augmentation import (
    GeometricSMOTE,
    OverSamplingAugmentation
)
from research.utils import (
    check_pipelines,
    check_pipelines_wrapper
)

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)
TEST_SIZE = 0.2
RANDOM_SEED = 42


# setup data utilization rate for each threshold
def make_dur(threshold):
    def dur(test_scores, data_utilization):
        return data_utilization_rate(
            test_scores,
            data_utilization,
            threshold=threshold
        )
    return dur


for i in range(60, 100, 5):
    SCORERS[f'dur_{i}'] = ALScorer(make_dur(i/100))


class remove_test(SamplerMixin):
    """
    Used to ensure the data used to train classifiers with and without AL
    is the same. This method replicates the split method in the ALWrapper
    object.
    """
    def __init__(self, test_size=.2):
        self.test_size = test_size

    def _fit_resample(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=RANDOM_SEED
        )
        return X_train, y_train

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)


# Experiment setup
CONFIG = {
    'generator': [
        ('NONE', None, {}),

        # Pure oversampling (same as last paper)
        ('G-SMOTE', OverSamplingAugmentation(
            GeometricSMOTE(k_neighbors=5, deformation_factor=.5, truncation_factor=.5)
        ), {}),

        # Oversampling augmentation
        ('G-SMOTE-AUGM', OverSamplingAugmentation(
            GeometricSMOTE(k_neighbors=5, deformation_factor=.5, truncation_factor=.5),
            augmentation_strategy='constant', value=1200
        ), {}),
    ],
    'remove_test': [
        ('remove_test', remove_test(TEST_SIZE), {})
    ],
    'classifiers': [
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {}),
        ('DT', DecisionTreeClassifier(), {}),
        ('RF', RandomForestClassifier(), {})
    ],
    'simulations': [
        ('AL-BASE', ALSimulation(
            n_initial=.016,
            increment=.016,
            max_iter=49,
            test_size=TEST_SIZE,
            random_state=42
        ), {
            'evaluation_metric': ['accuracy', 'f1_macro',
                                  'geometric_mean_score_macro'],
            'selection_strategy': ['random', 'entropy', 'breaking_ties'],
        }),
        ('AL-GEN', ALSimulation(
            n_initial=.016,
            increment=.016,
            max_iter=49,
            test_size=TEST_SIZE,
            random_state=42
        ), {
            'evaluation_metric': ['accuracy', 'f1_macro',
                                  'geometric_mean_score_macro'],
            'selection_strategy': ['random', 'entropy', 'breaking_ties'],
            'use_sample_weight': [True, False]
        })
    ],
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'scoring_al': [
        'accuracy',
        'f1_macro',
        'geometric_mean_score_macro',
        'area_under_learning_curve',
    ] + [f'dur_{i}' for i in range(60, 100, 5)],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 42,
    'n_jobs': -1,
    'verbose': 1
}


if __name__ == '__main__':

    # extract and load datasets
    ZipFile(join(DATA_PATH, 'active_learning_augmentation.db.zip'), 'r')\
        .extract('active_learning_augmentation.db', path=DATA_PATH)

    datasets = load_datasets(data_dir=DATA_PATH)

    # remove uncompressed database file
    os.remove(join(DATA_PATH, 'active_learning_augmentation.db'))

    # Extract pipelines and parameter grids
    parameter_grids = {
        'ceiling': check_pipelines(
            [CONFIG['remove_test'], CONFIG['classifiers']],
            CONFIG['rnd_seed'],
            CONFIG['n_runs']
        ),
        'al_base': check_pipelines_wrapper(
            [CONFIG['generator'][:2], CONFIG['classifiers']],
            CONFIG['simulations'][0],
            CONFIG['rnd_seed'],
            CONFIG['n_runs'],
            wrapped_only=True
        ),
        'al_proposed': check_pipelines_wrapper(
            [CONFIG['generator'][1:], CONFIG['classifiers']],
            CONFIG['simulations'][1],
            CONFIG['rnd_seed'],
            CONFIG['n_runs'],
            wrapped_only=True
        )
    }

    iterable = list(product(parameter_grids.items(), datasets))
    for (exp_name, (estimators, param_grids)), (name, (X, y)) in track(
            iterable, description='Running experiments across datasets'):
        if exp_name.startswith('al'):
            scoring = CONFIG['scoring_al']
        else:
            scoring = CONFIG['scoring']

        # Define and fit AL experiment
        experiment = ModelSearchCV(
            estimators,
            param_grids,
            scoring=scoring,
            n_jobs=CONFIG['n_jobs'],
            cv=StratifiedKFold(
                n_splits=CONFIG['n_splits'],
                shuffle=True,
                random_state=CONFIG['rnd_seed']
            ),
            verbose=CONFIG['verbose'],
            return_train_score=True,
            refit=False
        ).fit(X, y)

        # Save results
        file_name = f'{name.replace(" ", "_").lower()}_{exp_name}.pkl'
        pd.DataFrame(experiment.cv_results_)\
            .to_pickle(join(RESULTS_PATH, file_name))
