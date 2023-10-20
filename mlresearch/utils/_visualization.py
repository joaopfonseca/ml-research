"""
Functions for visualization formatting or producing pre-formatted
visualizations.
"""
from setuptools import distutils
import warnings
import numpy as np
from ._utils import _optional_import


def set_matplotlib_style(font_size=8, use_latex=True, **rcparams):
    """
    Load LaTeX-style configurations for Matplotlib Visualizations. You may pass
    additional parameters to the rcParams as keyworded arguments.

    Parameters
    ----------
    font_size : int, default=8
        Desired default font size displayed in visualizations. ``axes.labelsize`` and
        ``font.size`` will take 1.25x the size passed in ``font_size``, whereas
        ``legend.fontsize``, ``xtick.labelsize`` and ``ytick.labelsize`` will take the
        value passed in this parameter.

    use_latex : bool, default=True
        Whether to use Latex to render visualizations. If ``True`` and a Latex
        installation is found in the system, the text will be rendered using Latex and
        math mode can be used. If ``True`` and no Latex installation is found,
        ``text.usetex`` will be set to ``False`` and an issue is raised. If ``False``,
        ``text.usetex`` will be set to ``False``.

    Returns
    -------
    None : NoneType
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
        "font.family": "Times",
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

    if distutils.spawn.find_executable("latex") and use_latex:
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
        }
        plt.rcParams.update(tex_fonts)
    elif use_latex:
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

    if type(col) is list:
        col = np.array(col)

    norm = colors.Normalize(vmin=col.min(), vmax=col.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(col)

    return np.apply_along_axis(colors.rgb2hex, 1, rgba)
