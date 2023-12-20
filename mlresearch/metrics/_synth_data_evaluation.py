"""
Implementation of the 3-dimensional metric from the paper 'How Faithful is your Synthetic
Data? Sample-level Metrics for Evaluating and Auditing Generative Models' from Alaa et al
(2022).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class AlphaPrecision:
    """
    Measures synthetic data fidelity. It estimates the probability that a synthetic
    sample resides in the $\\alpha$-support of the real distribution.

    This is an implementation of the metric proposed in [1]_.

    .. warning::
        This metric is not listed in the ``get_scorer_names`` function since it is
        following an unconventional structure.

    Parameters
    ----------
    scorer_real : function
        Method used to map a dataset into a score, or a 1-dimensional projection of
        itself. The mapping should be modelled over the original (real) dataset.

    Attributes
    ----------
    center_ : float
        Value of the center of the euclidean ball.

    References
    ----------

    .. [1] Alaa, A., Van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022, June).
        How faithful is your synthetic data? sample-level metrics for evaluating and
        auditing generative models. In International Conference on Machine Learning
        (pp. 290-306). PMLR.

    """

    def __init__(self, scorer_real):
        self.scorer_real = scorer_real

    def fit(self, X_real):
        """
        Compute statistics necessary to calculate $\\alpha$-precision.

        Parameters
        ----------
        X_real : array-like or pd.DataFrame, shape (n_samples, n_features)
            The real (original) dataset used to fit `self.scorer_real`.

        Returns
        -------
        self : object
            Returns an instance of the class.
        """
        original_scores = self.scorer_real(X_real)
        self.center_ = np.median(original_scores)
        self._dist = np.abs(original_scores - self.center_)
        return self

    def score(self, X, alpha=0.05):
        """
        Returns 1 if a sample resides in the $\\alpha$-support of the original
        distribution, 0 otherwise.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data over which $\\alpha$-precision will be calculated.

        alpha : float, default=0.05
            Percentile used to determine the radius of the euclidean ball.

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            $\\alpha$-precision scores.
        """
        radius = np.quantile(self._dist, 1 - alpha)
        within_ball = np.abs(self.scorer_real(X) - self.center_) < radius
        return within_ball.astype(int)


class BetaRecall:
    """
    Checks whether the synthetic data is diverse enough to cover the variability of real
    data, i.e., a model should be able to generate a wide variety of good samples.

    This is an implementation of the metric proposed in [1]_.

    .. warning::
        This metric is not listed in the ``get_scorer_names`` function since it is
        following an unconventional structure.

    Parameters
    ----------
    scorer_synth : function
        Method used to map a dataset into a score, or a 1-dimensional projection of
        itself. The mapping should be modelled over the synthetic dataset.

    Attributes
    ----------
    center_ : float
        Value of the center of the euclidean ball.

    References
    ----------

    .. [1] Alaa, A., Van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022, June).
        How faithful is your synthetic data? sample-level metrics for evaluating and
        auditing generative models. In International Conference on Machine Learning
        (pp. 290-306). PMLR.

    """

    def __init__(self, scorer_synth):
        self.scorer_synth = scorer_synth

    def fit(self, X_synth):
        """
        Compute statistics necessary to calculate $\\beta$-recall.

        Parameters
        ----------
        X_synth : array-like or pd.DataFrame, shape (n_samples, n_features)
            The synthetic dataset used to fit `self.scorer_synth`.

        Returns
        -------
        self : object
            Returns an instance of the class.
        """
        original_scores = self.scorer_synth(X_synth)
        self.center_ = np.median(original_scores)
        self._dist = np.abs(original_scores - self.center_)
        return self

    def score(self, X, beta=0.05):
        """
        Returns 1 if a sample resides in the $\\beta$-support of the synthetic
        distribution, 0 otherwise.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data over which $\\beta$-recall will be calculated.

        beta : float, default=0.05
            Percentile used to determine the radius of the euclidean ball.

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            $\\beta$-recall scores.
        """
        radius = np.quantile(self._dist, 1 - beta)
        within_ball = np.abs(self.scorer_synth(X) - self.center_) < radius
        return within_ball.astype(int)


class Authenticity:
    """
    Quantifies the rate by which a model generates new samples. In other words, this
    scorer assesses whether a sample is non-memorized.

    This is an implementation of the metric proposed in [1]_.

    .. warning::
        This metric is not listed in the ``get_scorer_names`` function since it is
        following an unconventional structure.

    Parameters
    ----------
    metric : str or callable, default='euclidean'
        Metric to use for distance computation. Default is "euclidean", which
        results in the standard Euclidean distance. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    nn_ : estimator object
        Validated k-nearest neighbours algorithm. Used to find the nearest neighbors of
        the synthetic and the original data using the original data as a reference.

    distances_real_ : np.ndarray, shape (n_samples,)
        Distance to the nearest neighbor for each sample in `X_real`.

    References
    ----------

    .. [1] Alaa, A., Van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022, June).
        How faithful is your synthetic data? sample-level metrics for evaluating and
        auditing generative models. In International Conference on Machine Learning
        (pp. 290-306). PMLR.

    """

    def __init__(self, metric="euclidean", n_jobs=None):
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X_real):
        """
        Compute statistics necessary to calculate Authenticity.

        Parameters
        ----------
        X_real : array-like or pd.DataFrame, shape (n_samples, n_features)
            The real (original) dataset used to fit `self.scorer_real`.

        Returns
        -------
        self : object
            Returns an instance of the class.
        """
        self.nn_ = NearestNeighbors(
            n_neighbors=2, metric=self.metric, n_jobs=self.n_jobs
        ).fit(X_real)
        distances, neighbors = self.nn_.kneighbors(X_real)
        self.distances_real_ = distances[:, 1]
        return self

    def score(self, X):
        """
        Returns 1 if an observation is deemed authentic, 0 otherwise.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data over which Authenticity will be calculated.

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Authenticity scores.
        """
        distances, neighbors = self.nn_.kneighbors(X)
        distances = distances[:, 0]
        neighbors = neighbors[:, 0]
        a_j = 1 - (distances < self.distances_real_[neighbors]).astype(int)
        return a_j
