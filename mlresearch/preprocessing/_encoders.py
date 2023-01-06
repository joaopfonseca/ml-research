import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder


class PipelineEncoder(_BaseEncoder):
    """
    Pipeline-compatible adaptation of Scikit-learn's OneHotEncoder and OrdinalEncoder
    objects.

    The fitted encoder object from Scikit-learn is stored in ``self.encoder_``.

    Parameters
    ----------
    categorical_features : TODO

    encoder : TODO
    """

    _estimator_type = "encoder"

    def __init__(self, categorical_features=None, encoder=None, **kwargs):
        self.categorical_features = categorical_features
        self.encoder = encoder
        self._kwargs = kwargs

    def _check_X(self, X):
        if type(X) == pd.DataFrame:
            self.is_pandas_ = (
                True if not hasattr(self, "is_pandas_") else self.is_pandas_
            )
            self.columns_ = X.columns
            X_ = X.copy().values
        else:
            self.is_pandas_ = (
                False if not hasattr(self, "is_pandas_") else self.is_pandas_
            )
            X_ = X.copy()
        return X_

    def _check_categorical_features(self, X):

        if self.categorical_features is None or len(self.categorical_features) == 0:
            cat_features = np.zeros(X.shape[-1]).astype(bool)

        elif type(self.categorical_features) in [str, bool, int, float]:
            cat_features = np.array([self.categorical_features])

        elif hasattr(self.categorical_features, "__iter__"):
            if len(set([type(i) for i in self.categorical_features])) != 1:
                raise TypeError(
                    "``categorical_features`` cannot have more than one type of object."
                )
            cat_features = np.array(self.categorical_features)

        else:
            error_msg = (
                "``categorical_features`` must be an iterable or one of str,"
                + " bool, int, float or NoneType. Got "
                + f"{type(self.categorical_features).__name__} instead."
            )
            raise TypeError(error_msg)

        is_mask = np.array([type(i) == np.bool_ for i in cat_features]).all()
        is_indices = np.array(
            [type(i) in [np.int64, np.float64] for i in cat_features]
        ).all()
        is_col_names = np.array([type(i) == np.str_ for i in cat_features]).all()

        if is_mask:
            return cat_features

        elif is_indices:
            categorical_features_ = np.zeros(X.shape[-1])
            categorical_features_[cat_features] = 1
            return categorical_features_.astype(bool)

        elif is_col_names:
            if not self.is_pandas_:
                error_msg = (
                    "If ``categorical_features`` contains string values, "
                    + "``X`` must be a pandas dataframe."
                )
                raise TypeError(error_msg)
            in_columns = np.array([col in X.columns for col in cat_features])
            if any(~in_columns):
                not_in_columns = np.array(cat_features)[np.where(~in_columns)[0]]
                raise KeyError(", ".join(not_in_columns))
            elif self.is_pandas_ and any(X.columns.isin(cat_features)):
                return X.columns.isin(cat_features)

        else:
            raise TypeError(
                "Could not parse which features are categorical from "
                + f"``categorical_features``. Got {self.categorical_features}."
            )

    def fit(self, X, y=None):

        X_ = self._check_X(X)

        self.categorical_features_ = self._check_categorical_features(X)

        if not self.categorical_features_.any():
            # If there are no categorical features apply no change
            return self

        self.encoder_ = (
            clone(self.encoder)
            if self.encoder is not None
            else OneHotEncoder(**self._kwargs)
        )
        self.encoder_.fit(X_[:, self.categorical_features_], y)

        return self

    def transform(self, X):

        X_ = self._check_X(X)

        if not self.categorical_features_.any():
            # If there are no categorical features apply no change
            return X_

        if self.is_pandas_:
            metric_data = pd.DataFrame(
                X_[:, ~self.categorical_features_],
                columns=self.columns_[~self.categorical_features_],
            )
            enc_vals = self.encoder_.transform(X_[:, self.categorical_features_])
            encoded_data = pd.DataFrame(
                enc_vals if not issparse(enc_vals) else enc_vals.toarray(),
                columns=self.encoder_.get_feature_names_out(
                    self.columns_[self.categorical_features_]
                ),
            )
            data = pd.concat([metric_data, encoded_data], axis=1)

        else:
            metric_data = X_[:, ~self.categorical_features_]
            enc_vals = self.encoder_.transform(X_[:, self.categorical_features_])
            encoded_data = enc_vals if not issparse(enc_vals) else enc_vals.toarray()
            data = np.concatenate([metric_data, encoded_data], axis=1).astype(
                np.float64
            )

        return data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
