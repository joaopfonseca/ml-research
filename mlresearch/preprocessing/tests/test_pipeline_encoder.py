import pytest
import numpy as np
import pandas as pd
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .._encoders import PipelineEncoder


sklearn_encoders = [None, OrdinalEncoder(), OneHotEncoder()]


def generate_datasets():
    """
    Create datasets with different number of categorical features stored as both numpy
    and pandas formats.
    """

    MIN_CLASS = 20
    MAJ_CLASS = 100
    N_FEATURES = 4

    X = pd.DataFrame(
        np.random.random((30 + MIN_CLASS + MAJ_CLASS, N_FEATURES)),
        columns=[str(i) for i in range(N_FEATURES)],
    )
    y = pd.Series(
        [
            *[0 for i in range(MIN_CLASS)],
            *[1 for i in range(30)],
            *[2 for i in range(MAJ_CLASS)],
        ],
        name="target",
    )

    datasets = [(X, y, None), (X, y, [])]

    # Set up mixed type datasets with varying number of unique values
    for n_cat in range(1, N_FEATURES + 1):
        cat_columns = {
            col: f"cat_{i}"
            for i, col in enumerate(np.random.choice(X.columns, n_cat, replace=False))
        }
        X_ = X.rename(columns=cat_columns).copy()
        y_ = y.copy()

        X_ = X_.assign(
            **(X_[cat_columns.values()] * np.random.randint(1, 10)).round().astype(int)
        )

        cat_cols_1 = X_.columns.str.startswith("cat_")
        cat_cols_2 = list(cat_cols_1)
        cat_cols_3 = list(np.where(cat_cols_1)[0])
        cat_cols_4 = X_.columns[cat_cols_1].tolist()

        # Pandas dataframes
        for cat_cols in [cat_cols_1, cat_cols_2, cat_cols_3, cat_cols_4]:
            datasets.append((X_.copy(), y_.copy(), cat_cols))

        # Numpy arrays
        for cat_cols in [cat_cols_1, cat_cols_2, cat_cols_3]:
            datasets.append((X_.copy().values, y_.copy().values, cat_cols))

        # Pandas + Numpy
        for cat_cols in [cat_cols_1, cat_cols_2, cat_cols_3]:
            datasets.append((X_.copy().values, y_.copy(), cat_cols))

        # Numpy + Pandas
        for cat_cols in [cat_cols_1, cat_cols_2, cat_cols_3, cat_cols_4]:
            datasets.append((X_.copy(), y_.copy().values, cat_cols))

    return datasets


@pytest.mark.parametrize("X, y, cat_features", generate_datasets())
def test_default_encoder(X, y, cat_features):
    encoder = PipelineEncoder(
        cat_features,
        sparse_output=False,
        dtype=np.float64,
    )

    if (
        cat_features is not None
        and len(cat_features) > 0
        and type(X) is np.ndarray
        and type(cat_features[0]) is str
    ):
        with pytest.raises(TypeError):
            encoder.fit_transform(X, y)

    elif cat_features is None or len(cat_features) == 0:
        with pytest.warns(UserWarning):
            encoder.fit_transform(X, y)

    else:
        X_ = encoder.fit_transform(X, y)

        # Check if type of data structure is preserved
        assert type(X) is type(X_)

        if type(X) is pd.DataFrame:
            X = X.values
            X_ = X_.values

        if cat_features is None or len(cat_features) == 0:
            assert not encoder.features_.any()
            assert (X == X_).all()
        elif type(cat_features[0]) is np.bool_:
            assert encoder.features_.sum() == sum(cat_features)
        else:
            assert encoder.features_.sum() == len(cat_features)

        n_feats = sum([len(np.unique(feat)) for feat in X[:, encoder.features_].T])
        assert X_.shape == (X.shape[0], n_feats + sum(~encoder.features_))


@pytest.mark.parametrize(
    "sklearn_encoder, X, y, categorical_features",
    [
        (sklearn_encoder, *dataset)
        for sklearn_encoder, dataset in product(sklearn_encoders, generate_datasets())
    ],
)
def test_encoder(sklearn_encoder, X, y, categorical_features):
    # Check ordinal and One-Hot encoder
    encoder = PipelineEncoder(features=categorical_features, encoder=sklearn_encoder)

    if (
        categorical_features is not None
        and len(categorical_features) > 0
        and type(X) is np.ndarray
        and type(categorical_features[0]) is str
    ):
        with pytest.raises(TypeError):
            encoder.fit_transform(X, y)

    elif categorical_features is None or len(categorical_features) == 0:
        with pytest.warns(UserWarning):
            encoder.fit_transform(X, y)

    else:

        if type(X) is pd.DataFrame:
            cat_cols = X.columns[X.columns.str.startswith("cat_")]
            for cat_col in cat_cols:
                X[cat_col] = X[cat_col].astype(str)

        X_ = encoder.fit_transform(X, y)

        # Check if type of data structure is preserved
        assert type(X) is type(X_)

        if type(X) is pd.DataFrame:
            num_cols = X.columns[~encoder.features_]
            assert (X.dtypes[num_cols] == X_.dtypes[num_cols]).all()

            X = X.values
            X_ = X_.values

        if sklearn_encoder.__class__.__name__ == "OrdinalEncoder":
            assert X.shape == X_.shape
        else:
            n_feats = sum([len(np.unique(feat)) for feat in X[:, encoder.features_].T])
            assert X_.shape == (
                X.shape[0],
                n_feats + sum(~encoder.features_),
            )

        # Test if self.encoded_features_idx_out_ is being defined correctly (#43)
        # (only if there are features to transform)
        if not encoder.features_.any():
            metric_features = [
                i
                for i in range(X_.shape[1])
                if i not in encoder.encoded_features_idx_out_
            ]
            assert (X_[:, metric_features] == X[:, ~encoder.features_]).all()


@pytest.mark.parametrize(
    "sklearn_encoder, X, y, categorical_features",
    [
        (sklearn_encoder, *dataset)
        for sklearn_encoder, dataset in product(sklearn_encoders, generate_datasets())
    ],
)
def test_pipeline_encoder(sklearn_encoder, X, y, categorical_features):
    # Check in Pipeline
    pipeline = make_pipeline(
        PipelineEncoder(features=categorical_features, encoder=sklearn_encoder),
        DecisionTreeClassifier(),
    )
    if (
        categorical_features is not None
        and len(categorical_features) > 0
        and type(X) is np.ndarray
        and type(categorical_features[0]) is str
    ):
        with pytest.raises(AttributeError):
            pipeline.fit_transform(X, y)

    elif categorical_features is None or len(categorical_features) == 0:
        with pytest.warns(UserWarning):
            pipeline.fit(X, y)

    else:
        pipeline.fit(X, y)
        pipeline.predict(X)


def test_pipeline_encoder_errors():
    X, y, _ = generate_datasets()[3]

    err_cat_features = [[0, True, "cat_2"], [{}, {}, {}]]
    for cat_features in err_cat_features:
        encoder = PipelineEncoder(features=cat_features)
        with pytest.raises(TypeError):
            encoder.fit_transform(X, y)

    key_err_features = [["test"], ["test1", "test2"], "test"]
    for cat_features in key_err_features:
        encoder = PipelineEncoder(features=cat_features)
        with pytest.raises(KeyError):
            encoder.fit_transform(X, y)

    err_type_features = [["cat_0"], "cat_0"]
    for cat_features in err_type_features:
        encoder = PipelineEncoder(features=cat_features)
        with pytest.raises(TypeError):
            encoder.fit_transform(X.values, y)
