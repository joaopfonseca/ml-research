"""
Analyze the experimental results.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join
from zipfile import ZipFile
import pandas as pd
from rlearn.tools import summarize_datasets
from research.utils import generate_paths, load_datasets

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)


def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={
            'Dataset name': 'Dataset', 'Imbalance Ratio': 'IR',
        }).drop(columns=['Minority instances', 'Majority instances'])\
        .set_index('Dataset')\
        .join(pd.Series(dict(
            [(name, dat[-1].unique().size) for name, dat in datasets]
        ), name='Classes'))\
        .reset_index()
    return summarized


if __name__ == '__main__':

    # extract and load datasets
    ZipFile(join(DATA_PATH, 'active_learning_augmentation.db.zip'), 'r')\
        .extract('active_learning_augmentation.db', path=DATA_PATH)

    datasets = load_datasets(data_dir=DATA_PATH)

    # remove uncompressed database file
    os.remove(join(DATA_PATH, 'active_learning_augmentation.db'))

    # datasets_description
    summarize_multiclass_datasets(datasets).to_csv(
        join(ANALYSIS_PATH, 'datasets_description.csv'), index=False
    )
