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

    # Find the font family that is available and similar to times new roman
    font_preferences = (["Times New Roman", "Times", "TeX Gyre Bonum", "Nimbus Roman"],)
    fonts = list_available_fonts()
    fonts = [font for font in font_preferences if font in fonts]
    if len(fonts) > 0:
        base_style["font.family"] = fonts[0]
    else:
        warn_msg = (
            "Could not find a font family similar to Times New Roman. Matplotlib's "
            "default font will be used instead."
        )
        warnings.warn(warn_msg)

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


def list_available_fonts():
    """
    Returns a list of available fonts in the current system.

    Returns
    -------
    list
        List of font names.
    """

    fonts = _optional_import("matplotlib.font_manager")
    return [font.name for font in fonts.fontManager.ttflist]


def _make_html(fontname):
    return (
        "<p>{font}: <span style='font-family:{font}; "
        "font-size: 24px;'>{font}</p>".format(font=fontname)
    )


def display_available_fonts(ipython_session=True):
    """
    Check and display the available fonts in matplotlib.

    Parameters
    ----------
    ipython_session : bool, optional
        Flag to determine whether to display the fonts in an IPython session or return
        the HTML output as a string. If True, the fonts will be displayed in the IPython
        session using the IPython.core.display.HTML function. If False, the HTML output
        will be returned as a string. Default is True.

    Returns
    -------
    str or None
        If `ipython_session` is True, the function displays the fonts in the IPython
        session and returns None. If `ipython_session` is False, the function returns the
        HTML output as a string.

    Examples
    --------
    >>> display_available_fonts()
    # Displays the available fonts in the IPython session

    >>> html_output = display_available_fonts(ipython_session=False)
    >>> print(html_output)
    # Prints the HTML output as a string
    """

    fonts = _optional_import("matplotlib.font_manager")

    code = "\n".join(
        [
            _make_html(font)
            for font in sorted(set([f.name for f in fonts.fontManager.ttflist]))
        ]
    )
    html_output = "<div style='column-count: 2;'>{}</div>".format(code)
    if ipython_session:
        HTML = _optional_import("IPython.core.display.HTML")
        HTML(html_output)
    else:
        return html_output
