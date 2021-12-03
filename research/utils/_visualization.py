"""
Functions for visualization formatting or producing pre-formatted
visualizations.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable


def load_plt_sns_configs(font_size=8):
    """
    Load LaTeX style configurations for Matplotlib/Seaborn
    Visualizations.
    """
    sns.set_style("whitegrid")
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
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
        "figure.subplot.hspace": 0.2,
    }
    plt.rcParams.update(tex_fonts)


def val_to_color(col, cmap="RdYlBu_r"):
    """
    Converts a column of values to hex-type colors.

    Parameters
    ----------
    col : array-like of shape (n_samples,)
        Values to convert to hex-type color code

    cmap : str or `~matplotlib.colors.Colormap`
        The colormap used to map normalized data values to RGBA colors

    Returns
    -------
    colors : array-like of shape (n_samples,)
        Array with hex values as string type.
    """
    norm = Normalize(vmin=col.min(), vmax=col.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(col)

    return np.apply_along_axis(rgb2hex, 1, rgba)
