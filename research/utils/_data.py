"""
Data I/O utils. Later on I might add other data handling utilities.
"""
from os import listdir
from os.path import isdir, join
import pandas as pd
from sqlite3 import connect


def load_datasets(data_dir):
    """Load datasets from sqlite database and/or csv files."""
    assert isdir(data_dir), '`data_dir` must be a directory.'
    datasets = []
    for dat_name in listdir(data_dir):
        data_path = join(data_dir, dat_name)
        if dat_name.endswith('.csv'):
            ds = pd.read_csv(data_path)
            name = dat_name.replace('.csv', '').replace('_', ' ').upper()
            X, y = ds.iloc[:, :-1], ds.iloc[:, -1]
            datasets.append((name, (X, y)))
        elif dat_name.endswith('.db'):
            with connect(data_path) as connection:
                datasets_names = [
                    name[0]
                    for name in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table';"
                    )
                ]
                for dataset_name in datasets_names:
                    ds = pd.read_sql(
                        f'select * from "{dataset_name}"', connection
                    )
                    X, y = ds.iloc[:, :-1], ds.iloc[:, -1]
                    datasets.append(
                        (dataset_name.replace('_', ' ').upper(), (X, y))
                    )
    return datasets
