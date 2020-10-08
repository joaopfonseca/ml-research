"""
This submodule contains a variety of general utility functions as well as
tools used to format and prepare tables to incorporate into LaTeX code.

Additionally, an expanded (as compared with scikit-learn's) dictionary of
scorers is also provided.

This code was taken from the `utils.py` script from AlgoWit's
publications repo, to which I have also contributed.

Link to related repo: https://github.com/AlgoWit/publications
"""
from ._metrics import SCORERS, geometric_mean_score_macro
from ._results import (
    generate_mean_std_tbl,
    generate_pvalues_tbl,
    sort_tbl,
    generate_paths,
    make_bold
)
from ._image import img_array_to_pandas

__all__ = [
    'SCORERS',
    'geometric_mean_score_macro',
    'generate_mean_std_tbl',
    'generate_pvalues_tbl',
    'sort_tbl',
    'generate_paths',
    'make_bold',
    'img_array_to_pandas'
]
