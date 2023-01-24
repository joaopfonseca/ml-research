import numpy as np
import matplotlib.pyplot as plt

from .._visualization import set_matplotlib_style, feature_to_color


def test_set_matplotlib_style():
    default_params = dict(plt.rcParams)
    set_matplotlib_style(12)
    new_params = dict(plt.rcParams)
    changed_params = [
        key for key in new_params.keys() if new_params[key] != default_params[key]
    ]
    assert len(changed_params) > 1


def test_feature_to_color():
    colors = feature_to_color(np.array([1, 2, 3, 4, 5]))
    colors2 = feature_to_color([1, 2, 3, 4, 5])
    assert (colors == colors2).all()
    assert colors.size == np.unique(colors).size
