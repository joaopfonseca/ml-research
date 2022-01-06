import pytest
import numpy as np
from .._acquisition_functions import ACQUISITION_FUNCTIONS


@pytest.mark.parametrize("name", ACQUISITION_FUNCTIONS.keys())
def test_acquisition_functions(name):
    func = ACQUISITION_FUNCTIONS[name]
    probabs = np.array([[0.5, 0.5], [.01, .99]])
    uncertainty = func(probabs)
    if name == 'random':
        assert (uncertainty == 0.5).all()
    else:
        assert uncertainty[0] > uncertainty[1]
