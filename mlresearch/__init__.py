"""Toolbox to develop research in Machine Learning.

``ml-research`` is a library containing the implementation of various algorithms
developed in Machine Learning research, as well as utilities to facilitate the formatting
of pandas dataframes into LaTeX tables.
"""

import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of mlresearch when
    # the binaries are not built
    # mypy error: Cannot determine type of '__MLRESEARCH_SETUP__'
    __MLRESEARCH_SETUP__  # type: ignore
except NameError:
    __MLRESEARCH_SETUP__ = False

if __MLRESEARCH_SETUP__:
    sys.stderr.write("Partial import of imblearn during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import active_learning
    from . import synthetic_data
    from . import datasets
    from . import metrics
    from . import preprocessing
    from . import utils
    from ._version import __version__
    from .utils._show_versions import show_versions

    __all__ = [
        "active_learning",
        "synthetic_data",
        "datasets",
        "metrics",
        "preprocessing",
        "utils",
        # Non-modules:
        "show_versions",
        "__version__",
    ]
