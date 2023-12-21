"""
Utility functions related to image processing
"""

import numpy as np
import pandas as pd


def image_to_dataframe(X, y=None, bands=None, target_feature="target"):
    """
    Converts an image array (height, width, bands) to a pandas dataframe
    (height * width, bands). If ``y`` is not ``None``, a feature with name
    ``target_feature`` will be added to the dataset.

    .. note:: Some workflows use image arrays of format (bands, width, height). In that
        case, you can simply transpose the image before calling this function.

    Parameters
    ----------
    X : array-like of shape (h, w, b)
        Matrix containing the image data.

    y : array-like of shape (h, w), default=None
        The target values (class labels) as integers or strings.

    bands : array-like of shape (b,), default=None
        The names of the bands in the image.

    target_feature : str, default="target"
        Target feature name.

    Returns
    -------
    df_image : pd.DataFrame, shape (h * w, b[+1])
        Dataframe with pixel coordinates (h, w) as index, counting from the top left
        corner.

    Examples
    --------

    >>> import numpy as np
    >>> X = np.random.default_rng(42).random((4,5,3))
    >>> y = np.random.default_rng(42).integers(0,2,4*5).reshape(4,5)
    >>> image_to_dataframe(X).head(5)
    ...             0         1         2
    ... h w
    ... 0 0  0.773956  0.438878  0.858598
    ...   1  0.697368  0.094177  0.975622
    ...   2  0.761140  0.786064  0.128114
    ...   3  0.450386  0.370798  0.926765
    ...   4  0.643865  0.822762  0.443414
    >>> image_to_dataframe(X, y, bands=["r","g", "b"], target_feature="classes").head(5)
    ...             r         g         b  classes
    ... h w
    ... 0 0  0.773956  0.438878  0.858598        0
    ...   1  0.697368  0.094177  0.975622        1
    ...   2  0.761140  0.786064  0.128114        1
    ...   3  0.450386  0.370798  0.926765        0
    ...   4  0.643865  0.822762  0.443414        0
    """
    # Check y's dimensionality
    if y is not None and len(y.shape) == 2:
        y = np.expand_dims(y, axis=-1)

    # Collect metadata
    shp = X.shape
    columns = [i for i in range(shp[-1])] if bands is None else bands
    indices = np.indices(shp[:-1]).reshape((2, shp[0] * shp[1]))
    indices = pd.MultiIndex.from_arrays(indices, names=["h", "w"])

    if y is None:
        dat = np.moveaxis(X, -1, 0).reshape((len(columns), shp[0] * shp[1]))
        df_image = pd.DataFrame(data=dat.T, columns=columns, index=indices)
    else:
        columns = columns + [target_feature]
        dat = np.moveaxis(np.append(X, y, axis=-1), -1, 0).reshape(
            (len(columns), shp[0] * shp[1])
        )
        df_image = pd.DataFrame(data=dat.T, columns=columns, index=indices)
        df_image[target_feature] = df_image[target_feature].astype(int)

    return df_image


def dataframe_to_image(df, bands=None, target_feature=None):
    """
    Converts a pandas dataframe to an image. The height ("h"), and width ("w")
    coordinates of the image must be in the index.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with pixel coordinates (h, w) as index, counting from the top left
        corner.

    bands : array-like of shape (b,), default=None
        The names of the bands in the dataframe to be passed to the image.

    target_feature : str, default=None
        Target feature name.

    Returns
    -------
    X : array-like of shape (h, w, b)
        Matrix containing the image data.

    y : array-like of shape (h, w), default=None
        The target values (class labels) as integers or strings.

    bands : array-like of shape (b,), default=None
        The names of the bands in the image.
    """

    if (np.sort(df.index.names) != ["h", "w"]).any():
        err_msg = (
            'Image coordinates with names ["h", "w"] must be in index,'
            f" got {list(df.index.names)} instead."
        )
        raise IndexError(err_msg)

    df_ = df.reset_index().pivot(index="h", columns="w")
    y = df_[target_feature].values if target_feature is not None else None

    if bands is None:
        bands = (
            df.columns if target_feature is None else df.columns.drop(target_feature)
        )

    X = np.array([df_[band] for band in bands])
    X = np.moveaxis(X, 0, -1)

    return X, y, bands
