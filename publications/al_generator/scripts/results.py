"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from gsmote import GeometricSMOTE
from clover.over_sampling import ClusterOverSampler
from sklearn.model_selection import StratifiedKFold
from rlearn.model_selection import ModelSearchCV
from research.utils import (
    load_datasets,
    generate_paths,
    check_pipelines,
    check_pipelines_wrapper
)
from research.active_learning import ALWrapper

from imblearn.base import SamplerMixin
from sklearn.model_selection import train_test_split


class remove_test(SamplerMixin):
    """
    Used to ensure the data used to train classifiers with and without AL
    is the same.
    """
    def __init__(self):
        pass

    def _fit_resample(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=42
        )
        return X_train, y_train

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)


CONFIG = {
    # Remove .2 of the dataset from training, to replicate the training data
    # for AL methods
    'remove_test': [
        ('remove_test', remove_test(), {})
    ],
    'classifiers': [
        ('LR', LogisticRegression(
                multi_class='multinomial',
                solver='sag',
                penalty='none',
                max_iter=1e4
            ),
            {}
         ),
        ('KNN', KNeighborsClassifier(), {}),
        ('RF', RandomForestClassifier(), {})
    ],
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 42,
    'n_jobs': -1,
    'verbose': 1
}

CONFIG_AL = {
    'generator': [
        # ('NONE', None, {}),
        # ('SMOTE', ClusterOverSampler(SMOTE(k_neighbors=5), n_jobs=1), {}),
        ('G-SMOTE', ClusterOverSampler(GeometricSMOTE(
            k_neighbors=5, deformation_factor=.5, truncation_factor=.5
        ), n_jobs=-1), {})
    ],
    'wrapper': (
        'AL',
        ALWrapper(
            n_initial=20,
            increment=15,
            max_iter=400,
            test_size=.2,
            random_state=42
        ), {
            'evaluation_metric': ['accuracy', 'f1_macro',
                                  'geometric_mean_score_macro'],
            'selection_strategy': ['random', 'entropy', 'margin sampling']
        }
    ),
    'scoring': [
        'accuracy',
        'f1_macro',
        'geometric_mean_score_macro',
        'area_under_learning_curve',
        'data_utilization_rate'
    ]
}


if __name__ == '__main__':

    # Extract paths
    data_dir, results_dir, _ = generate_paths(__file__)

    # Load datasets
    datasets = load_datasets(data_dir=data_dir)

    # Extract pipelines and parameter grids
    estimators_al, param_grids_al = check_pipelines_wrapper(
        [CONFIG_AL['generator'], CONFIG['classifiers']],
        CONFIG_AL['wrapper'],
        CONFIG['rnd_seed'],
        CONFIG['n_runs'],
        wrapped_only=True
    )

    estimators_base, param_grids_base = check_pipelines(
        [CONFIG['remove_test'], CONFIG['classifiers']],
        CONFIG['rnd_seed'],
        CONFIG['n_runs']
    )

    for name, (X, y) in datasets:
        # Define and fit AL experiment
        experiment_al = ModelSearchCV(
            estimators_al,
            param_grids_al,
            scoring=CONFIG_AL['scoring'],
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
        file_name = f'{name.replace(" ", "_").lower()}_al.pkl'
        pd.DataFrame(experiment_al.cv_results_)\
            .to_pickle(join(results_dir, file_name))

        # Define and fit baseline experiment
        experiment_base = ModelSearchCV(
            estimators_base,
            param_grids_base,
            scoring=CONFIG['scoring'],
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
        file_name = f'{name.replace(" ", "_").lower()}_base.pkl'
        pd.DataFrame(experiment_base.cv_results_)\
            .to_pickle(join(results_dir, file_name))
