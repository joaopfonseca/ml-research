"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from collections import Counter
from os import pardir
from os.path import join, dirname

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from research.datasets import RemoteSensingDatasets

DATA_PATH = join(dirname(__file__), pardir, "data")


if __name__ == "__main__":

    # Download datasets
    datasets = RemoteSensingDatasets().download()

    # Sample datasets
    min_n_samples, dataset_size, rnd_seed = 150, 1500, 42
    content = []
    for name, data in datasets.content_:

        classes = [
            cl for cl, count in Counter(data.target).items() if count >= min_n_samples
        ]
        data = data[data.target.isin(classes)]

        data, _ = train_test_split(
            data, train_size=dataset_size, stratify=data.target, random_state=rnd_seed
        )

        data = pd.concat(
            [
                pd.DataFrame(MinMaxScaler().fit_transform(data.drop(columns="target"))),
                data.reset_index(drop=True).target,
            ],
            axis=1,
        )
        content.append((name, data))

    # Save database
    datasets.content_ = content
    datasets.save(DATA_PATH, "active_learning")
