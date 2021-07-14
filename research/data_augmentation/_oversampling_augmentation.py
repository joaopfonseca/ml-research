"""
A wrapper to allow the use of oversampling algorithms for data augmentation
in Active Learning experiments with multiple datasets.
"""
from collections import Counter, OrderedDict
import numpy as np
from sklearn.base import clone
from imblearn.over_sampling.base import BaseOverSampler

AUGMENTATION_STRATEGIES = [
    'oversampling',
    'constant',
    'proportional'
]


class OverSamplingAugmentation(BaseOverSampler):
    """
    A wrapper to facilitate the use of `imblearn.over_sampling` objects for
    data augmentation.

    Parameters
    ----------
    oversampler : oversampler estimator, default=None
        Over-sampler to be used for data augmentation.

    augmentation_strategy : float, dict or {'oversampling', 'constant', 'proportional'}\
        , default='oversampling'
        Specifies how the data augmentation is done.

        - When ``float`` or ``int``, each class' frequency is augmented
          according to the specified ratio.

        - When ``oversampling``, the data augmentation is done according to the
          sampling strategy passed in the ``oversampler`` object.

        - When ``constant``, each class frequency is augmented to match
          the value passed in the parameter ``value``.

        - When ``proportional``, relative class frequencies are preserved and the
          number of samples in the dataset is matched with the value passed in the
          parameter ``value``.

    value : int, float, default=None
        Value to be used as the new absolute frequency of each class. It is
        ignored unless the augmentation strategy is set to 'constant'.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    """

    def __init__(
        self,
        oversampler=None,
        augmentation_strategy='oversampling',
        value=None,
        random_state=None
    ):
        super(OverSamplingAugmentation, self).__init__(
            sampling_strategy='auto'
        )
        self.oversampler = oversampler
        self.augmentation_strategy = augmentation_strategy
        self.value = value
        self.random_state = random_state

    def fit(self, X, y):
        """
        Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
            (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """

        X, y, _ = self._check_X_y(X, y)

        if type(self.augmentation_strategy) not in [int, float, dict] \
                and self.augmentation_strategy not in AUGMENTATION_STRATEGIES:
            raise ValueError(
                f"When 'augmentation_strategy' in neither an int or float,"
                f" it needs to be one of {AUGMENTATION_STRATEGIES}. Got "
                f"'{self.augmentation_strategy}' instead."
            )

        if (type(self.value) not in [int, float]) \
                and (self.augmentation_strategy in ['constant', 'proportional']):
            raise ValueError(
                f"When 'augmentation_strategy' is 'constant' or 'proportional',"
                f" 'value' needs to be an int or float. Got "
                f"{self.value} instead."
            )

        # Setup the sampling strategy based on the augmentation strategy
        if self.augmentation_strategy == 'constant':
            counts = OrderedDict(Counter(y))
            self.sampling_strategy_ = {
                k: int(np.round(self.value))
                if self.value > freq
                else freq
                for k, freq in counts.items()
            }
        elif self.augmentation_strategy == 'proportional':
            counts = OrderedDict(Counter(y))
            ratio = self.value / y.shape[0]
            if ratio > 1:
                self.sampling_strategy_ = {
                    k: int(np.round(freq*ratio))
                    for k, freq in counts.items()
                }
            else:
                raise ValueError(
                    "The new size of the augmented dataset must be larger than the"
                    f" original dataset. Originally, there are {y.shape[0]} samples"
                    f" and {self.value} samples are asked."
                )
        elif self.augmentation_strategy == 'oversampling':
            self.sampling_strategy_ = self.oversampler.sampling_strategy

        elif type(self.augmentation_strategy) in [int, float]:
            counts = OrderedDict(Counter(y))
            self.sampling_strategy_ = {
                k: int(np.round(v*self.augmentation_strategy))
                for k, v in counts.items()
            }
        else:
            self.sampling_strategy_ = self.augmentation_strategy

        return self

    def fit_resample(self, X, y, **fit_params):
        """
        Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
            (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
            (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """

        self.fit(X, y)

        if self.oversampler is not None:
            self.oversampler_ = clone(self.oversampler)\
                .set_params(
                    random_state=self.random_state,
                    sampling_strategy=self.sampling_strategy_
                )
            return self.oversampler_.fit_resample(X, y, **fit_params)
        else:
            return X, y

    def _fit_resample(self, X, y):
        """A placeholder. It was overriden by the self.fit_resample method."""
        return
