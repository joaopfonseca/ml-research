import pytest
import numpy as np
from html.parser import HTMLParser

try:
    import matplotlib.pyplot as plt

    matplotlib_installed = True
except ModuleNotFoundError:
    matplotlib_installed = False

import shutil

from .._visualization import (
    set_matplotlib_style,
    feature_to_color,
    display_available_fonts,
    list_available_fonts,
)


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
def test_set_matplotlib_style_use_latex_reset():
    """Test that calling set_matplotlib_style with use_latex=False after
    use_latex=True correctly resets text.usetex to False.

    Regression test for GitHub issue #70.
    """
    with plt.rc_context():
        # Force text.usetex to True first (simulating a prior LaTeX-enabled call)
        plt.rcParams["text.usetex"] = True

        # Second call with use_latex=False — must reset text.usetex to False
        set_matplotlib_style(use_latex=False)
        assert not plt.rcParams["text.usetex"], (
            "text.usetex should be False after calling "
            "set_matplotlib_style(use_latex=False)"
        )


@pytest.mark.skipif(not matplotlib_installed, reason="Matplotlib not installed.")
def test_set_matplotlib_style_use_latex_fallback():
    """Test that when use_latex=True but LaTeX is not installed,
    text.usetex is reset to False and a warning is emitted.

    Regression test for GitHub issue #70 (fallback path).
    """
    original_which = shutil.which
    try:
        shutil.which = lambda cmd: None if cmd == "latex" else original_which(cmd)

        with plt.rc_context():
            with pytest.warns(UserWarning, match="Could not find a LaTeX installation"):
                set_matplotlib_style(use_latex=True)

            assert not plt.rcParams[
                "text.usetex"
            ], "text.usetex should be False when LaTeX is unavailable"
    finally:
        shutil.which = original_which


@pytest.mark.skipif(not matplotlib_installed, reason="Matplotlib not installed.")
def test_feature_to_color():
    colors = feature_to_color(np.array([1, 2, 3, 4, 5]))
    colors2 = feature_to_color([1, 2, 3, 4, 5])
    assert (colors == colors2).all()
    assert colors.size == np.unique(colors).size


def test_available_fonts():

    class MyHTMLParser(HTMLParser):
        tags = {}

        def handle_starttag(self, tag, attrs):
            self.tags["start"] = tag

        def handle_endtag(self, tag):
            self.tags["end"] = tag

    html_output = display_available_fonts(ipython_session=False, display=False)
    parser = MyHTMLParser()
    parser.feed(html_output)
    assert parser.tags["start"] == "span"
    assert parser.tags["end"] == "div"

    fonts = list_available_fonts()
    assert len(fonts) > 0
    assert all(font in html_output for font in fonts)
