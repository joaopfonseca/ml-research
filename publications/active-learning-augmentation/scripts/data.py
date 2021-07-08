"""
Extract the dadtabase.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from os import pardir
from os.path import join, dirname
# from sklearn.model_selection import train_test_split
from research.datasets import (
    MulticlassDatasets
)

DATA_PATH = join(dirname(__file__), pardir, 'data')


if __name__ == '__main__':

    # Download datasets
    datasets = MulticlassDatasets().download()

    # Save database
    datasets.save(DATA_PATH, 'active_learning_augmentation')
