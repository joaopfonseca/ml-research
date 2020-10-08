"""
Utility functions related to image processing
"""

import numpy as np
import pandas as pd


def img_array_to_pandas(X, y):
    """
    Converts an image as numpy array (with ground truth) to a pandas dataframe
    """
    shp = X.shape
    columns = [i for i in range(shp[-1])]+['target']
    dat = np.concatenate([
        np.moveaxis(X, -1, 0), np.moveaxis(y, -1, 0)
    ], axis=0).reshape((len(columns), shp[0]*shp[1]))
    return pd.DataFrame(data=dat.T, columns=columns)
