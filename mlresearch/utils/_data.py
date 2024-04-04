"""
Data I/O utils. Later on I might add other data handling utilities.
"""

from os import listdir
from os.path import isdir, join
import pandas as pd
from sqlite3 import connect


def load_datasets(
    data_dir, prefix="", suffix="", target_exists=True, **read_csv_kwargs
):
    """
    Load all datasets in a directory from sqlite databases and/or csv files.

    Parameters
    ----------
    data_dir : str
        Data directory to be crawled.

    prefix : str, default=''
        Load dataset if the file starts with the specified prefix.

    suffix : str, default=''
        Load dataset if the file starts with the specified suffix.

    target_exists : bool, default=True
        Specify wether there is a target feature. If True, it is assumed to be in the
        last position of the dataset.

    Returns
    -------
    datasets : list
        A list with nested tuples with structure (dataset_name, (X, y)).
    """
    assert isdir(data_dir), "`data_dir` must be a directory."

    # Filter data by suffix
    dat_names = [
        dat
        for dat in listdir(data_dir)
        if (dat.startswith(prefix) and dat.endswith(suffix))
    ]

    # Read data
    datasets = []
    for dat_name in dat_names:
        data_path = join(data_dir, dat_name)

        # Handle csv data
        if dat_name.endswith(".csv"):
            ds = pd.read_csv(data_path, **read_csv_kwargs)
            name = dat_name.replace(".csv", "").replace("_", " ").upper()
            if target_exists:
                ds = (ds.iloc[:, :-1], ds.iloc[:, -1])
            datasets.append((name, ds))

        # Handle sqlite database
        elif dat_name.endswith(".db"):
            with connect(data_path) as connection:
                datasets_names = [
                    name[0]
                    for name in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table';"
                    )
                ]
                for dataset_name in datasets_names:
                    ds = pd.read_sql(f'select * from "{dataset_name}"', connection)
                    if target_exists:
                        ds = (ds.iloc[:, :-1], ds.iloc[:, -1])
                    datasets.append((dataset_name.replace("_", " ").upper(), ds))
    return datasets
