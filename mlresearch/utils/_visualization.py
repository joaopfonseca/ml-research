"""
Functions for visualization formatting or producing pre-formatted
visualizations.
"""
from distutils.spawn import find_executable
import warnings
import types
import numpy as np


def _optional_import(module: str) -> types.ModuleType:
    """
    Import an optional dependency.

    Parameters
    ----------
    module : str
        The identifier for the backend. Either an entrypoint item registered
        with importlib.metadata, "matplotlib", or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    # This function was adapted from the _load_backend function from the pandas.plotting
    # source code.
    import importlib

    # Attempt an import of an optional dependency here and raise an ImportError if
    # needed.
    try:
        module_ = importlib.import_module(module)
    except ImportError:
        mod = module.split(".")[0]
        raise ImportError(f"{mod} is required to use this functionality.") from None

    return module_


def set_matplotlib_style(font_size=8, **rcparams):
    """
    Load LaTeX-style configurations for Matplotlib Visualizations.
    """
    plt = _optional_import("matplotlib.pyplot")

    # Replicates the rcParams of seaborn's "whitegrid" style and a few extra
    # configurations I like
    plt.style.use("seaborn-v0_8-whitegrid")
    base_style = {
        # "patch.edgecolor": "w",
        # "patch.force_edgecolor": True,
        # "xtick.bottom": False,
        # "ytick.left": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": (10 / 8) * font_size,
        "font.size": (10 / 8) * font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        # Subplots size/shape
        "figure.subplot.left": 0.098,
        "figure.subplot.right": 0.938,
        "figure.subplot.bottom": 0.12,
        "figure.subplot.top": 0.944,
        "figure.subplot.wspace": 0.071,
    }
    plt.rcParams.update(base_style)

    if find_executable("latex"):
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
        }
        plt.rcParams.update(tex_fonts)
    else:
        warn_msg = (
            "Could not find a LaTeX installation. ``text.usetex`` will be set to False."
        )
        warnings.warn(warn_msg)

    # Used to pass any additional custom configurations
    plt.rcParams.update(rcparams)


def feature_to_color(col, cmap="RdYlBu_r"):
    """
    Converts a column of values to hex-type colors.

    Parameters
    ----------
    col : {list, array-like} of shape (n_samples,)
        Values to convert to hex-type color code

    cmap : str or `~matplotlib.colors.Colormap`
        The colormap used to map normalized data values to RGBA colors

    Returns
    -------
    colors : array-like of shape (n_samples,)
        Array with hex values as string type.
    """
    colors = _optional_import("matplotlib.colors")
    cm = _optional_import("matplotlib.cm")

    if type(col) == list:
        col = np.array(col)

    norm = colors.Normalize(vmin=col.min(), vmax=col.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(col)

    return np.apply_along_axis(colors.rgb2hex, 1, rgba)
