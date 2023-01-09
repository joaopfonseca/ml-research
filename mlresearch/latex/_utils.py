"""
Contains several functions to prepare and format tables for LaTeX documents.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import numpy as np
import pandas as pd


def sort_table(table, index_order_dict, columns_order_list=None):
    """
    Sort tables rows and columns. Mostly used to format results from
    oversampling experiments.
    """
    cols = table.columns
    for key, cat in index_order_dict.items():
        if key in cols:
            table[key] = pd.Categorical(table[key], categories=cat)

    key_cols = [col for col in cols if col in index_order_dict.keys()]
    table.sort_values(key_cols, inplace=True)

    if columns_order_list is not None:
        table = table[key_cols + columns_order_list].set_index(key_cols)

    return table


def make_bold(row, maximum=True, decimals=2, threshold=None, with_sem=False):
    """
    Make bold the lowest or highest value(s).
    with_sem returns an incomplete textbf latex function and a mask array.
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
        "\\textbf{%s" % formatter.format(v)
        if with_sem
        else "\\textbf{%s}" % formatter.format(v)
        for v in row[mask].astype(float)
    ]

    # Return mask only if function is being used to generate
    # a table with sem values
    if with_sem:
        return row, mask
    else:
        return row


def make_mean_sem_table(
    mean_vals, sem_vals=None, make_bold=False, maximum=True, decimals=2, threshold=None
):
    """
    Generate table with rounded decimals, bold maximum/minimum values or values
    above/below a given threshold, and combine mean and sem values.
    """

    if sem_vals is not None:
        scores = (
            mean_vals.applymap(("{:,.%sf}" % decimals).format)
            + r" $\pm$ "
            + sem_vals.applymap(("{:,.%sf}" % decimals).format)
        )
    else:
        scores = mean_vals.applymap(("{:,.%sf}" % decimals).format)

    if make_bold:
        mask = mean_vals.apply(
            lambda row: make_bold(row, maximum, decimals, threshold, with_sem=True)[1],
            axis=1,
        ).values

        scores.iloc[:, :] = np.where(mask, "\\textbf{" + scores + "}", scores)

    return scores


def export_latex_longtable(df, path=None, caption=None, label=None):
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
            index=False,
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
