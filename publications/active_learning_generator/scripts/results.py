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
from imblearn.over_sampling import (
    SMOTE
)
from clover.over_sampling import ClusterOverSampler
from sklearn.model_selection import StratifiedKFold
from rlearn.model_selection import ModelSearchCV
from research.utils import (
    load_datasets,
    generate_paths,
    check_pipelines_wrapper
)
from research.active_learning import ALWrapper

CONFIG = {
    'oversamplers': [
        ('NONE', None, {}),
        ('SMOTE', ClusterOverSampler(SMOTE(), n_jobs=1), {
            'oversampler__k_neighbors': [3, 4, 5]
        })
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
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 8]}),
        ('RF', RandomForestClassifier(), {
            'max_depth': [None, 3, 6], 'n_estimators': [50, 100, 200]
        })
    ],
    'wrapper': (
        'AL',
        ALWrapper(
            n_initial=20,
            increment=15,
            max_iter=400,
            test_size=.1,
            random_state=0
        ), {
            'evaluation_metric': ['accuracy', 'f1_macro',
                                  'geometric_mean_score_macro'],
            'selection_strategy': ['random', 'entropy', 'margin sampling']
        }
    ),
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1,
    'verbose': 1
}


if __name__ == '__main__':

    # Extract paths
    data_dir, results_dir, _ = generate_paths(__file__)

    # Load datasets
    datasets = load_datasets(data_dir=data_dir)

    # Extract pipelines and parameter grids
    estimators, param_grids = check_pipelines_wrapper(
        [CONFIG['oversamplers'], CONFIG['classifiers']],
        CONFIG['wrapper'],
        CONFIG['rnd_seed'],
        CONFIG['n_runs'],
        wrapped_only=True
    )

    for name, (X, y) in datasets:
        # Define and fit experiment
        experiment = ModelSearchCV(
            estimators,
            param_grids,
            scoring=CONFIG['scoring'],
            n_jobs=CONFIG['n_jobs'],
            cv=StratifiedKFold(
                n_splits=CONFIG['n_splits'],
                shuffle=True,
                random_state=CONFIG['rnd_seed']
            ),
            verbose=CONFIG['verbose'],
            return_train_score=False,
            refit=False
        ).fit(X, y)

        # Save results
        file_name = f'{name.replace(" ", "_").lower()}.pkl'
        pd.DataFrame(experiment.cv_results_)\
            .to_pickle(join(results_dir, file_name))
