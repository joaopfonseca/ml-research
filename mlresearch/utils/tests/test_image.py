"""
Test conversions between image and dataframe.
"""

from itertools import product
import pytest
import numpy as np

from .._image import image_to_dataframe, dataframe_to_image

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
            for h, w in product([0, shape[0] - 1], [0, shape[1] - 1]):
                assert (df.loc[h, w] == X[h, w]).all()
        else:
            assert df.shape == (shape[0] * shape[1], shape[2] + 1)
            assert df.columns[-1] == "target"
            assert df["target"].dtype == int
            for h, w in product([0, shape[0] - 1], [0, shape[1] - 1]):
                assert (df.loc[h, w] == np.append(X[h, w], y[h, w])).all()

        if bands is None:
            assert df.columns[:number_of_bands].tolist() == list(range(number_of_bands))
        else:
            assert df.columns[:number_of_bands].tolist() == band_names


@pytest.mark.parametrize("number_of_bands", list(range(1, 20, 3)))
def test_dataframe_to_image(number_of_bands):
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
        target_ = "target" if y is not None else None
        X_, y_, bands_ = dataframe_to_image(df=df, target_feature=target_)

        if y is None:
            assert y_ is None
        else:
            assert type(y_) is np.ndarray
            assert (y_ == y).all()

        assert (bands_ == df.columns.drop(target_, errors="ignore")).all()
        assert type(X_) is np.ndarray
        assert (X_ == X).all()

        # Without "h", "w" features at all
        with pytest.raises(IndexError):
            dataframe_to_image(df=df.reset_index(drop=True), target_feature=target_)

        # With "h", "w" in columns
        with pytest.raises(IndexError):
            dataframe_to_image(df=df.reset_index(drop=True), target_feature=target_)

    # Test passing a subset of bands
    if len(band_names) > 2:
        df = image_to_dataframe(X=X, y=target, bands=band_names)
        bands_filter = np.random.choice(band_names, 2, replace=False)

        X_, y_, bands_ = dataframe_to_image(
            df=df, bands=bands_filter, target_feature="target"
        )

        assert (bands_ == bands_filter).all()
        assert type(y_) is np.ndarray
        assert type(X_) is np.ndarray
        assert (y_ == target).all()

        band_indices = [
            np.where(df.columns.drop("target") == b)[0][0] for b in bands_filter
        ]
        assert (X_ == X[:, :, band_indices]).all()
