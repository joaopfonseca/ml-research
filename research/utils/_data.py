"""
Data I/O utils. Later on I might add other data handling utilities.
"""
from os import listdir
from os.path import isdir, join
import pandas as pd
from sqlite3 import connect


def load_datasets(data_dir, suffix='', target_exists=True, **read_csv_kwargs):
    """Load datasets from sqlite database and/or csv files."""
    assert isdir(data_dir), '`data_dir` must be a directory.'

    # Filter data by suffix
    dat_names = [
        dat for dat in listdir(data_dir)
        if dat.endswith(suffix)
    ]

    # Read data
    datasets = []
    for dat_name in dat_names:
        data_path = join(data_dir, dat_name)

        # Handle csv data
        if dat_name.endswith('.csv'):
            ds = pd.read_csv(data_path, **read_csv_kwargs)
            name = dat_name.replace('.csv', '').replace('_', ' ').upper()
            if target_exists:
                ds = (ds.iloc[:, :-1], ds.iloc[:, -1])
            datasets.append((name, ds))

        # Handle sqlite database
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
                    if target_exists:
                        ds = (ds.iloc[:, :-1], ds.iloc[:, -1])
                    datasets.append(
                        (dataset_name.replace('_', ' ').upper(), ds)
                    )
    return datasets
