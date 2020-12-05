"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         João Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rlearn.tools import (
    summarize_datasets
)
from research.datasets import RemoteSensingDatasets
from research.utils import (
    generate_paths,
    load_datasets
)

DATASETS_NAMES = [
    d.replace('fetch_', '')
    for d in dir(RemoteSensingDatasets())
    if d.startswith('fetch_')
]


def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={
            'Dataset name': 'Dataset',
            'Imbalance Ratio': 'IR',
            'Minority instances': 'Min. Instances',
            'Majority instances': 'Maj. Instances'
        })\
        .set_index('Dataset')\
        .join(pd.Series(dict(
            [(name, dat[-1].unique().size) for name, dat in datasets]
        ), name='Classes'))\
        .reset_index()
    summarized.loc[:, 'Dataset'] = summarized.loc[:, 'Dataset']\
        .apply(lambda x: x.title())
    return summarized


def plot_lulc_images():
    arrays_x = []
    arrays_y = []
    for dat_name in DATASETS_NAMES:
        X, y = RemoteSensingDatasets()._load_gic_dataset(dat_name)
        arrays_x.append(X[:, :, 100])
        arrays_y.append(np.squeeze(y))

    for X, y, figname in zip(arrays_x, arrays_y, DATASETS_NAMES):
        plt.figure(
            figsize=(20, 10),
            dpi=320
        )
        if figname == 'kennedy_space_center':
            X = np.clip(X, 0, 350)
        for i, (a, cmap) in enumerate(zip([X, y], ['gist_gray', 'terrain'])):
            plt.subplot(2, 1, i+1)
            plt.imshow(
                a, cmap=plt.get_cmap(cmap)
            )
            plt.axis('off')
        plt.savefig(
            join(analysis_path, figname),
            bbox_inches='tight',
            pad_inches=0
        )


if __name__ == '__main__':

    data_path, results_path, analysis_path = generate_paths(__file__)

    # load datasets
    datasets = load_datasets(data_dir=data_path)

    # datasets description
    summarize_multiclass_datasets(datasets).to_csv(
        join(analysis_path, 'datasets_description.csv'), index=False
    )

    # datasets visualization
    # plot_lulc_images()
