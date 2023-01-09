"""
This module contains several functions to prepare and format tables for LaTeX
documents.
"""

from ._utils import (
    sort_table,
    make_bold,
    make_mean_sem_table,
    export_latex_longtable,
)


__all__ = [
    "sort_table",
    "make_bold",
    "make_mean_sem_table",
    "export_latex_longtable",
]
