"""
Analyze the experimental results.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join
from zipfile import ZipFile
from rlearn.tools import summarize_datasets
from research.utils import generate_paths, load_datasets

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

# extract and load datasets
ZipFile(join(DATA_PATH, 'active_learning_augmentation.db.zip'), 'r')\
    .extract('active_learning_augmentation.db', path=DATA_PATH)

datasets = load_datasets(data_dir=DATA_PATH)

# Remove uncompressed database file
os.remove(join(DATA_PATH, 'active_learning_augmentation.db'))
