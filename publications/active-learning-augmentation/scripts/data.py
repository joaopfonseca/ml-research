"""
Extract the dadtabase.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from os import pardir
from os.path import join, dirname
from research.datasets import MulticlassDatasets, ImbalancedBinaryDatasets

DATA_PATH = join(dirname(__file__), pardir, 'data')

datasets = MulticlassDatasets().download()
datasets2 = ImbalancedBinaryDatasets().download()
