"""
Extract the dadtabase.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join, dirname
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from research.datasets import (
    MulticlassDatasets
)

RANDOM_STATE = 42
DATA_PATH = join(dirname(__file__), os.pardir, 'data')


def set_sample_size(n_instances):
    """
    Determine how many instances to keep in a dataset.

    Side note: Why not just set the number of observations to a fixed size?
    """
    thresholds = [
        (i, i*2000) for i in range(1, 31)
    ]
    for divisor, threshold in thresholds:
        if n_instances <= threshold:
            return int(n_instances/divisor)


if __name__ == '__main__':

    # Download datasets
    datasets = MulticlassDatasets().download()

    # Sample and standardize datasets
    content = []
    for name, data in datasets.content_:
        n_instances = data.shape[0]
        if n_instances > 2000:
            data, _ = train_test_split(
                data,
                train_size=set_sample_size(n_instances),
                stratify=data.target,
                random_state=RANDOM_STATE
            )

        data = pd.concat(
            [
                pd.DataFrame(
                    MinMaxScaler().fit_transform(data.drop(columns='target'))
                ),
                data.reset_index(drop=True).target
            ],
            axis=1,
        )
        content.append((name, data))

    datasets.content_ = content

    # Save database
    datasets.save(DATA_PATH, 'active_learning_augmentation')

    # Zip database (it was too large to store directly on GitHub)
    db_path = join(DATA_PATH, 'active_learning_augmentation.db')
    ZipFile(db_path + '.zip', 'w').write(
        db_path,
        arcname='active_learning_augmentation.db',
        compress_type=ZIP_DEFLATED
    )

    # Remove uncompressed database file
    os.remove(db_path)
