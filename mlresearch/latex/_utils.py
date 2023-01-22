"""
Contains several functions to prepare and format tables for LaTeX documents.
"""

# Author: Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from itertools import product
import numpy as np
import pandas as pd


def _check_indices(table_index, indices):

    if indices is None:
        indices = table_index.to_frame().to_dict("list")

    # Check indices - list
    if type(indices) == list:
        if all([type(i) in [str, int] for i in indices]):
            indices_ = {table_index.names[0]: {i: i for i in indices}}
        elif all([type(i) == list for i in indices]):
            indices_ = {
                name: {i: i for i in values}
                for name, values in zip(table_index.names, indices)
            }
        elif all([type(i) == dict for i in indices]):
            indices_ = {
                name: values for name, values in zip(table_index.names, indices)
            }

    # Check indices - dict
    if type(indices) == dict:
        if all([type(i) == str for i in list(indices.values())]):
            index_name = table_index.names[0] if table_index.names[0] is not None else 0
            indices_ = {index_name: indices}
        elif all([type(i) == list for i in list(indices.values())]):
            indices_ = {
                name: {i: i for i in values} for name, values in indices.items()
            }
        elif all([type(i) == dict for i in list(indices.values())]):
            indices_ = indices

    # Replace index name if there's only a single index with no name
    if len(indices_) == 1 and None in indices_.keys():
        indices_[0] = indices_[None]
        indices_.pop(None)

    return indices_


def format_table(table, indices=None, columns=None, drop_missing=True):
    """
    Sort rows and columns. Mostly used to set results from
    experiments in the intended order.

    TODO: Rewrite docstring properly.

    Arguments
    ---------
    table : TODO

    indices : list, dict

        - If list of strings, orders index according to the passed values.
        - If list of lists, orders multi indices according to the passed values.
        - If list of dicts, orders and renames indices according to the passed values.

        - If dict, orders and renames index values.
        - If dict of lists orders index columns and values.
        - If dict of dicts orders both index columns and values and renames values.

    columns : list, dict
        - If list, orders and selects columns
        - If dict, orders, selects and renames columns

    drop_missing : bool, default=True
        If True, removes any values/columns/index not included in ``indices`` and
        ``columns``.

    """

    table = table.copy()
    index_dtype = table.index.dtype
    index_name = table.index.names

    # Check indices
    indices_ = _check_indices(table.index, indices)

    # Get index data
    id_data = table.index.to_frame().reset_index(drop=True)
    id_order = list(indices_.keys())

    # Update mappers with missing values if necessary
    if not drop_missing:
        id_order = id_order + id_data.columns[~id_data.columns.isin(id_order)].tolist()
        indices_ = {
            key: {
                **mapper,
                **{i: i for i in id_data[key].unique() if i not in mapper.keys()},
            }
            for key, mapper in indices_.items()
        }

    # Map index values
    for key, cat in indices_.items():
        id_data[key] = pd.Categorical(id_data[key].map(cat), categories=cat.values())

    # Sort index cols and replace index values
    id_data = id_data[id_order]
    id_ = pd.MultiIndex.from_arrays(id_data.values.T, names=id_data.columns)
    table.set_index(id_, inplace=True)

    # Sort index vals
    index_vals_unique = [list(mapper.values()) for mapper in indices_.values()]
    index_values_sorted = (
        list(product(*index_vals_unique))
        if len(index_vals_unique) > 1
        else index_vals_unique[0]
    )
    table = table.loc[index_values_sorted]
    table.index = table.index.astype(index_dtype)
    if len(index_name) == 1 and index_name[0] is None:
        table.index.name = None

    # Update column list/dictionary if necessary
    if type(columns) == list:
        columns_ = {i: i for i in columns}
    elif type(columns) == dict:
        columns_ = columns

    if not drop_missing:
        missing = {i: i for i in table.columns if i not in columns_.keys()}
        columns_ = {**columns_, **missing}

    # Sort columns
    table = table[list(columns_.keys())].rename(columns=columns_).copy()

    return table


def _make_bold(row, maximum=True, decimals=2, threshold=None, with_sem=False):
    """
    Make bold the lowest or highest value(s).
    with_sem returns an incomplete textbf latex function and a mask array.

    This function should be used in a pandas dataframe using the .apply method.
    """
    row = round(row, decimals)
    if threshold is None:
        val = row.max() if maximum else row.min()
        mask = row == val
    else:
        mask = (row > threshold) if maximum else (row < threshold)

    formatter = "{0:.%sf}" % decimals
    row = row.apply(lambda el: formatter.format(el))

    row[mask] = [
        "\\textbf{%s" % v if with_sem else "\\textbf{%s}" % v for v in row[mask]
    ]

    # Return mask only if function is being used to generate
    # a table with sem values
    if with_sem:
        return row, mask
    else:
        return row


def make_bold(df, maximum=True, decimals=2, threshold=None, axis=1):
    """TODO"""
    return df.apply(
        lambda row: _make_bold(
            row, maximum=maximum, decimals=decimals, threshold=threshold
        ),
        axis=axis,
    )


def make_mean_sem_table(
    mean_vals,
    sem_vals=None,
    make_bold=False,
    maximum=True,
    decimals=2,
    threshold=None,
    axis=1,
):
    """
    Generate table with rounded decimals, bold maximum/minimum values or values
    above/below a given threshold, and combine mean and sem values.
    """

    if sem_vals is not None:

        if type(sem_vals) == np.ndarray:
            sem_vals = pd.DataFrame(
                sem_vals, index=mean_vals.index, columns=mean_vals.columns
            )

        scores = (
            mean_vals.applymap(("{:,.%sf}" % decimals).format)
            + r" $\pm$ "
            + sem_vals.applymap(("{:,.%sf}" % decimals).format)
        )
    else:
        scores = mean_vals.applymap(("{:,.%sf}" % decimals).format)

    if make_bold:
        mask = mean_vals.apply(
            lambda row: _make_bold(row, maximum, decimals, threshold, with_sem=True)[1],
            axis=axis,
        ).values

        scores.iloc[:, :] = np.where(mask, "\\textbf{" + scores + "}", scores)

    return scores


def export_latex_longtable(df, path=None, caption=None, label=None, index=False):
    """
    Exports a pandas dataframe to longtable format.
    This function replaces ``df.to_latex`` when there are latex commands in
    the table.
    """

    wo_tex = (
        df.to_latex(
            longtable=True,
            caption=caption,
            label=label,
            index=index,
            column_format="c" * df.shape[1],
        )
        .replace(r"\textbackslash ", "\\")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
        .replace(r"\$", "$")
    )

    if path is not None:
        open(path, "w").write(wo_tex)
    else:
        return wo_tex
