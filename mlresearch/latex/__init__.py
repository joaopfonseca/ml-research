"""
This module contains several functions to prepare and format tables for LaTeX
documents.
"""

from ._utils import (
    generate_mean_std_tbl,
    generate_pvalues_tbl,
    sort_tbl,
    generate_paths,
    make_bold,
    generate_mean_std_tbl_bold,
)


__all__ = [
    "generate_mean_std_tbl",
    "generate_pvalues_tbl",
    "sort_tbl",
    "generate_paths",
    "make_bold",
    "generate_mean_std_tbl_bold",
]
