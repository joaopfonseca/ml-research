import pytest
import numpy as np

try:
    import matplotlib.pyplot as plt

    matplotlib_installed = True
except ModuleNotFoundError:
    matplotlib_installed = False

from .._visualization import set_matplotlib_style, feature_to_color


@pytest.mark.skipif(not matplotlib_installed, reason="Matplotlib not installed.")
def test_set_matplotlib_style():
    default_params = dict(plt.rcParams)
    set_matplotlib_style(12)
    new_params = dict(plt.rcParams)
    changed_params = [
        key for key in new_params.keys() if new_params[key] != default_params[key]
    ]
    assert len(changed_params) > 1


@pytest.mark.skipif(not matplotlib_installed, reason="Matplotlib not installed.")
def test_feature_to_color():
    colors = feature_to_color(np.array([1, 2, 3, 4, 5]))
    colors2 = feature_to_color([1, 2, 3, 4, 5])
    assert (colors == colors2).all()
    assert colors.size == np.unique(colors).size
