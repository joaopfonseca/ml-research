from itertools import product
import numpy as np
import pandas as pd
import pytest
from .._utils import (
    _check_indices,
    format_table,
    make_bold,
    make_mean_sem_table,
    export_table,
)

classifiers = {"KNN": "K-NN", "LR": "LogReg", "DT": "DecTree"}
metrics = {"F1": "F-Score", "GM": "G-mean", "OA": "Acc"}
columns = ["Prop", "Bench2", "Bench1"]

rng = np.random.default_rng(42)

index = np.array(
    list(
        product(
            list(classifiers.keys()),
            list(metrics.keys()),
        )
    )
)
rng.shuffle(index)
rng.shuffle(columns)
X = rng.random((9, 3))

table = pd.DataFrame(
    data=X,
    index=pd.MultiIndex.from_arrays(index.T, names=["Clf", "Perf"]),
    columns=columns,
)

indices_list = [
    None,
    list(classifiers.keys()),
    [list(classifiers.keys()), list(metrics.keys())],
    [classifiers, metrics],
    classifiers,
    {"Perf": metrics},
    {"Perf": list(metrics.keys()), "Clf": list(classifiers.keys())},
    {"Perf": metrics, "Clf": classifiers},
]


@pytest.mark.parametrize("indices", indices_list)
def test_check_indices(indices):
    none_indices = {
        "Clf": {"LR": "LR", "KNN": "KNN", "DT": "DT"},
        "Perf": {"F1": "F1", "GM": "GM", "OA": "OA"},
    }
    indices_ = _check_indices(table.index, indices)

    assert type(indices_) is dict

    if indices is None:
        assert indices_ == none_indices

    if indices != {"Perf": metrics}:
        assert "Clf" in indices_.keys()
    else:
        assert "Perf" in indices_.keys()


@pytest.mark.parametrize(
    "columns", [{"Prop": "a", "Bench1": "b"}, ["Prop"], ["Prop", "Bench2", "Bench1"]]
)
def test_format_table(columns):
    indices = {"Perf": metrics}

    tab = format_table(table, indices=indices, columns=columns, drop_missing=False)
    assert len(tab.columns == 3)

    table_no_index = table.reset_index(drop=True)
    indices = list(range(table_no_index.shape[0]))
    np.random.shuffle(indices)

    tab = format_table(
        table_no_index, indices=indices[:-1], columns=columns, drop_missing=True
    )
    assert tab.index.tolist() == indices[:-1]
    assert tab.shape == (len(indices) - 1, len(columns))
    assert tab.index.tolist() == indices[:-1]


def test_make_bold():
    # Default values
    for i in [0, 1]:
        exp_max_indices = np.argmax(table.values, axis=i)
        table_bf = make_bold(table, decimals=5, axis=i)
        max_indices = np.argmax(
            table_bf.map(lambda x: x.startswith("\\textbf")).values, axis=i
        )
        assert (max_indices == exp_max_indices).all()

    # Threshold - higher than
    bold_test = [
        (table.values > 0.5, True),
        (table.values < 0.5, False),
    ]
    for exp_is_bold, maximum in bold_test:
        table_bf = make_bold(table, decimals=5, maximum=maximum, threshold=0.5)
        is_bold = table_bf.map(lambda x: x.startswith("\\textbf")).values
        assert (is_bold == exp_is_bold).all()


@pytest.mark.parametrize(
    "sem_vals, make_bold", product([None, rng.random((9, 3))], [True, False])
)
def test_make_mean_sem_table(sem_vals, make_bold):
    exp_max_indices = np.argmax(table.values, axis=1)
    mean_sem = make_mean_sem_table(table, sem_vals=sem_vals, make_bold=make_bold)

    if make_bold:
        max_indices = np.argmax(
            mean_sem.map(lambda x: x.startswith("\\textbf")).values, axis=1
        )
        assert (max_indices == exp_max_indices).all()

    if sem_vals is not None:
        assert mean_sem.map(lambda x: "$\\pm$" in x).values.all()

    if not make_bold and sem_vals is None:
        assert (mean_sem.values == table.round(2).astype(str).values).all()


def test_export_latex_longtable():
    mean_sem = make_mean_sem_table(table, rng.random((9, 3)), make_bold=True)
    longtable = export_table(mean_sem, index=True, longtable=True)
    assert longtable.startswith("\\begin{longtable}{ccc}")
    assert longtable.endswith("\\end{longtable}\n")
    assert longtable.count("$\\pm$") == mean_sem.size

    tabular = export_table(mean_sem, index=True, longtable=False)
    assert tabular.startswith("\\begin{tabular}{ccc}")
    assert tabular.endswith("\\end{tabular}\n")
    assert tabular.count("$\\pm$") == mean_sem.size
