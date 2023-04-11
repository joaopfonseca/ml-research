from os import remove
from pandas import DataFrame
from .._data import load_datasets
from ...datasets import ContinuousCategoricalDatasets


def test_load_datasets():
    DATASET_NAMES = ["echocardiogram", "acute"]

    # load and save datasets
    datasets = ContinuousCategoricalDatasets(names=DATASET_NAMES)
    datasets.download().save(".", "sample_data_base")
    for name, data in datasets.content_:
        data.to_csv(f"{name.lower()}_data_file.csv", index=False)

    # test csv passing suffix
    csv_datasets = load_datasets(".", suffix=".csv")

    # test .db passing prefix
    db_datasets = load_datasets(".", prefix="sample_")

    # load all (no filters) without target
    all_datasets = load_datasets(".", target_exists=False)

    # delete data files
    files = [f"{name}_data_file.csv" for name in DATASET_NAMES] + [
        "sample_data_base.db"
    ]
    for file in files:
        try:
            remove(file)
        except PermissionError:
            pass

    exp_csv_datasets = [f"{name.upper()} DATA FILE" for name in DATASET_NAMES]
    exp_db_datasets = [name.upper() for name in DATASET_NAMES]
    exp_all_datasets = [*exp_db_datasets, *exp_csv_datasets]

    assert len(csv_datasets) == len(DATASET_NAMES)
    assert len(db_datasets) == len(DATASET_NAMES)
    assert len(all_datasets) == len(DATASET_NAMES) * 2

    assert sorted(list(dict(csv_datasets).keys())) == sorted(exp_csv_datasets)
    assert sorted(list(dict(db_datasets).keys())) == sorted(exp_db_datasets)
    assert sorted(list(dict(all_datasets).keys())) == sorted(exp_all_datasets)

    assert [len(dat[-1]) for dat in csv_datasets] == [2, 2]
    assert [len(dat[-1]) for dat in db_datasets] == [2, 2]
    assert [type(dat[-1]) for dat in all_datasets] == [DataFrame for i in range(4)]
