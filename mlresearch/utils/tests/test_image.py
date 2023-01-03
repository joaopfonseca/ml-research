"""
Test conversions between image and dataframe.
"""
from itertools import product
import pytest
import numpy as np

from .._image import image_to_dataframe

RND_SEED = 42


@pytest.mark.parametrize("number_of_bands", list(range(1, 20, 3)))
def test_image_to_dataframe(number_of_bands):
    """Test the conversion of image arrays to regular dataframes."""
    shape = (150, 200, number_of_bands)
    band_names = [f"a{i}" for i in range(number_of_bands)]

    X = np.random.default_rng(RND_SEED).random(shape)
    target = (
        np.random.default_rng(RND_SEED)
        .integers(0, 2, shape[0] * shape[1])
        .reshape(shape[0], shape[1])
    )

    for bands, y in product([band_names, None], [target, None]):
        df = image_to_dataframe(X=X, y=y, bands=bands)

        if y is None:
            assert df.shape == (shape[0] * shape[1], shape[2])
            for h, w in product([0, shape[0]-1], [0, shape[1]-1]):
                assert (df.loc[h, w] == X[h, w]).all()
        else:
            assert df.shape == (shape[0] * shape[1], shape[2] + 1)
            assert df.columns[-1] == "target"
            assert df["target"].dtype == int
            for h, w in product([0, shape[0]-1], [0, shape[1]-1]):
                assert (df.loc[h, w] == np.append(X[h, w], y[h, w])).all()

        if bands is None:
            assert df.columns[:number_of_bands].tolist() == list(range(number_of_bands))
        else:
            assert df.columns[:number_of_bands].tolist() == band_names
