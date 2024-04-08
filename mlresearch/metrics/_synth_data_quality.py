"""
Implementation of the 3 synthetic data quality metrics from the paper 'How Faithful is
your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models'
from Alaa et al (2022).
"""

import numpy as np
from sklearn.metrics._scorer import _BaseScorer
from sklearn.neighbors import NearestNeighbors


class _BaseSynthQualityScorer(_BaseScorer):
    def __repr__(self):
        kwargs_string = "".join([f", {k}={v}" for k, v in self.__dict__.items()])
        return f"make_scorer({self.__class__.__name__}{kwargs_string})"

    def set_score_request(self):
        """
        Placeholder to overwrite sklearn's ``_BaseScorer.set_score_request`` function.
        It is not used and was raising a docstring error with scikit-learn v1.3.0.

        Note
        ----
        This placeholder will be removed soon
        """
        pass


class AlphaPrecision(_BaseSynthQualityScorer):
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

    alpha : float, default=0.05
        Percentile used to determine the radius of the euclidean ball.

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

    def __init__(self, scorer_real, alpha=0.05):
        self.scorer_real = scorer_real
        self.alpha = alpha

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

    def score(self, X_synth):
        """
        Returns 1 if a sample resides in the $\\alpha$-support of the original
        distribution, 0 otherwise.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data over which $\\alpha$-precision will be calculated.

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            $\\alpha$-precision scores.
        """
        radius = np.quantile(self._dist, 1 - self.alpha)
        within_ball = np.abs(self.scorer_real(X_synth) - self.center_) < radius
        return within_ball.astype(int)


class BetaRecall(_BaseSynthQualityScorer):
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

    beta : float, default=0.05
        Percentile used to determine the radius of the euclidean ball.

    n_neighbors : int, default=5
        Number of neighbors to use by default for computing the radius for each sample
        in `X_real` for scoring. Ignored if `scorer_synth` is not `None`.

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

        Ignored if `scorer_synth` is not `None`.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. Ignored if `scorer_synth` is not `None`.

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

    def __init__(
        self,
        scorer_synth=None,
        beta=0.05,
        n_neighbors=5,
        metric="euclidean",
        n_jobs=None,
    ):
        self.scorer_synth = scorer_synth
        self.beta = beta
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs

    def _fit_with_knn(self, X_synth):
        self.center_ = X_synth.mean(axis=0)
        self._dist = np.linalg.norm(X_synth - self.center_, axis=1)

        self.radius_ = np.quantile(self._dist, 1 - self.beta)
        within_ball = np.linalg.norm(X_synth - self.center_, axis=1) < self.radius_
        self.X_synth_ = X_synth[within_ball].copy()
        return self

    def _fit_with_support_estimation(self, X_synth):
        original_scores = self.scorer_synth(X_synth).reshape(-1, 1)
        self.center_ = np.median(original_scores, axis=0)
        self._dist = np.linalg.norm(original_scores - self.center_, axis=1)

        self.radius_ = np.quantile(self._dist, 1 - self.beta)
        return self

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
        if self.scorer_synth is None:
            self._fit_with_knn(X_synth)
        else:
            self._fit_with_support_estimation(X_synth)
        return self

    def _score_with_knn(self, X_real):
        nn_real_ = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=self.n_jobs
        ).fit(X_real)

        radius = nn_real_.kneighbors(X_real)[1][:, -1]

        nn_synth_ = NearestNeighbors(
            n_neighbors=1, metric=self.metric, n_jobs=self.n_jobs
        ).fit(self.X_synth_)
        dists = nn_synth_.kneighbors(X_real)[1][:, -1]

        return dists < radius

    def _score_with_support_estimation(self, X_real):
        scores = self.scorer_synth(X_real).reshape(-1, 1)
        within_ball = np.linalg.norm(scores - self.center_, axis=1) < self.radius_
        return within_ball

    def score(self, X_real):
        """
        Returns 1 if a sample resides in the $\\beta$-support of the synthetic
        distribution, 0 otherwise.

        Parameters
        ----------
        X_real : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data over which $\\beta$-recall will be calculated.

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            $\\beta$-recall scores.
        """
        if self.scorer_synth is None:
            scores = self._score_with_knn(X_real)
        else:
            scores = self._score_with_support_estimation(X_real)
        return scores.astype(int)


class Authenticity(_BaseSynthQualityScorer):
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
