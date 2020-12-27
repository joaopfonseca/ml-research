"""
Function for visualization formatting or producing pre-formatted
visualizations.
"""

import seaborn as sns
import matplotlib.pyplot as plt


def load_plt_sns_configs(font_size=8):
    """
    Load LaTeX style configurations for Matplotlib/Seaborn
    Visualizations.
    """
    sns.set_style('whitegrid')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": (10/8)*font_size,
        "font.size": (10/8)*font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        # Subplots size/shape
        "figure.subplot.left": .098,
        "figure.subplot.right": .938,
        "figure.subplot.bottom": .12,
        "figure.subplot.top": .944,
        "figure.subplot.wspace": .071,
        "figure.subplot.hspace": .2
    }
    plt.rcParams.update(tex_fonts)
