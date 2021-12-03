# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import numpy as np
import pandas as pd
from os.path import join, dirname, abspath


def generate_mean_std_tbl(mean_vals, std_vals):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    scores = (
        mean_vals.iloc[:, 2:].applymap("{:,.2f}".format)
        + r" $\pm$ "
        + std_vals.iloc[:, 2:].applymap("{:,.2f}".format)
    )
    tbl = pd.concat([index, scores], axis=1)
    return tbl


def generate_pvalues_tbl(tbl):
    """Format p-values."""
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(lambda pvalue: "%.1e" % pvalue)
    return tbl


def sort_tbl(tbl, ds_order=None, ovrs_order=None, clfs_order=None, metrics_order=None):
    """
    Sort tables rows and columns. Mostly used to format results from
    oversampling experiments.
    """
    cols = tbl.columns
    keys = ["Dataset", "Oversampler", "Classifier", "Metric"]
    for key, cat in zip(keys, (ds_order, ovrs_order, clfs_order, metrics_order)):
        if key in cols:
            tbl[key] = pd.Categorical(tbl[key], categories=cat)
    key_cols = [col for col in cols if col in keys]
    tbl.sort_values(key_cols, inplace=True)
    if ovrs_order is not None and set(ovrs_order).issubset(cols):
        tbl = tbl[key_cols + list(ovrs_order)]
    return tbl


def generate_paths(filepath):
    """
    Generate data, results and analysis paths.
    """
    prefix_path = join(dirname(abspath(filepath)), "..")
    paths = [join(prefix_path, name) for name in ("data", "results", "analysis")]
    return paths


def make_bold(row, maximum=True, num_decimals=2, threshold=None, with_sem=False):
    """
    Make bold the lowest or highest value(s).
    with_sem simply returns an incomplete textbf latex function.
    """
    row = round(row, num_decimals)
    if threshold is None:
        val = row.max() if maximum else row.min()
        mask = row == val
    else:
        mask = (row > threshold) if maximum else (row < threshold)
    formatter = "{0:.%sf}" % num_decimals
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


def generate_mean_std_tbl_bold(
    mean_vals, std_vals, maximum=True, decimals=2, threshold=None
):
    """
    Generate table that combines mean and sem values.
    """
    mean_bold = mean_vals.apply(
        lambda row: make_bold(row, maximum, decimals, threshold, with_sem=True)[0],
        axis=1,
    )
    mask = mean_vals.apply(
        lambda row: make_bold(row, maximum, decimals, threshold, with_sem=True)[1],
        axis=1,
    ).values

    formatter = "{0:.%sf}" % decimals
    std_bold = std_vals.applymap(lambda x: formatter.format(x))
    std_bold = np.where(mask, std_bold + "}", std_bold)
    scores = mean_bold + r" $\pm$ " + std_bold
    return scores
