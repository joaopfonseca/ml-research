"""
It includes utilities to search the parameter and model space.

Extracted from the no longer maintained ``research-learn`` library.
"""

# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV

from ..utils._check_pipelines import check_estimator_type, check_param_grids


class MultiEstimatorMixin(_BaseComposition):
    """Mixin class for multi estimator."""

    def __init__(self, estimators, est_name=None):

        self.estimators = estimators
        self.est_name = est_name

    def _validate_estimators(self):
        error_msg = "Invalid `estimators` attribute, `estimators` should"
        " be a list of (string, estimator) tuples."
        try:
            if len(self.estimators) == 0:
                raise TypeError(error_msg)
            for name, est in self.estimators:
                is_str = isinstance(name, str)
                is_est = isinstance(est, BaseEstimator)
                if not (is_str and is_est):
                    raise TypeError(error_msg)
        except TypeError:
            raise TypeError(error_msg)
        self.est_names_ = [est_name for est_name, _ in self.estimators]
        super(MultiEstimatorMixin, self)._validate_names(self.est_names_)
        if self.est_name not in self.est_names_:
            raise ValueError(
                f'Attribute `est_name` should be one of {", ".join(self.est_names_)}. '
                f"Got `{self.est_name}` instead."
            )

    def set_params(self, **params):
        """Set the parameters.
        Valid parameter keys can be listed with get_params().

        Parameters
        ----------
        params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``MultiEstimatorMixin``,
            the individual estimators of the ``MultiEstimatorMixin`` can also be
            set or replaced by setting them to None.
        """
        super(MultiEstimatorMixin, self)._set_params("estimators", **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters.

        Parameters
        ----------
        deep: bool
            Setting it to True gets the various estimators and the parameters
            of the estimators as well
        """
        return super(MultiEstimatorMixin, self)._get_params("estimators", deep=deep)

    def fit(self, X, y, **fit_params):
        """Fit the selected estimator."""

        # Validate estimators
        self._validate_estimators()

        # Copy one of the estimators
        estimator = clone(dict(self.estimators)[self.est_name])

        # Fit estimator
        self.estimator_ = estimator.fit(X, y, **fit_params)

        if hasattr(estimator, "classes_"):
            self.classes_ = estimator.classes_

        return self

    def predict(self, X):
        """Predict with the selected estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


class MultiClassifier(MultiEstimatorMixin, ClassifierMixin):
    """The functionality of a collection of classifiers is provided as
    a single metaclassifier. The classifier to be fitted is selected using a
    parameter."""

    def predict_proba(self, X):
        """Predict the probability with the selected estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


class MultiRegressor(MultiEstimatorMixin, RegressorMixin):
    """The functionality of a collection of regressors is provided as
    a single metaregressor. The regressor to be fitted is selected using a
    parameter."""

    pass


class ModelSearchCV(GridSearchCV):
    """Exhaustive search over specified parameter values for a collection of estimators.

    Important members are fit, predict.

    ModelSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimators used.

    The parameters of the estimators used to apply these methods are optimized
    by cross-validated grid-search over their parameter grids.

    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    estimators :  list of (string, estimator) tuples
        Each estimator is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grids : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, list/tuple, dict or None, default=None
        A single string or a callable to evaluate the predictions on the
        test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        Note that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        If ``None``, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int or string, default=None
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - ``None``, in which case all the jobs are immediately created.
        - An int, giving the exact number of total jobs that are spawned.
        - A string, as a function of n_jobs i.e. ``'2*n_jobs'``.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - ``None``, to use the default 3-fold cross validation.
        - integer, to specify the number of folds in a ``(Stratified)KFold``.
        - An object to be used as a cross-validation generator.
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_parameters_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be availble.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``ModelSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer, default=0
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +-------------------+-----------+------------+-----------------+---+---------+
        |param_dtc_criterion|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +===================+===========+============+=================+===+=========+
        |      'entropy'    |     --    |      2     |       0.80      |...|    2    |
        +-------------------+-----------+------------+-----------------+---+---------+
        |      'entropy'    |     --    |      3     |       0.70      |...|    4    |
        +-------------------+-----------+------------+-----------------+---+---------+
        |      'entropy'    |     0.1   |     --     |       0.80      |...|    3    |
        +-------------------+-----------+------------+-----------------+---+---------+
        |      'entropy'    |     0.2   |     --     |       0.93      |...|    1    |
        +-------------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from mlresearch.model_selection import ModelSearchCV
    >>> X, y, *_ = load_breast_cancer().values()
    >>> param_grids = [{'dt__max_depth': [3, 6]}, {'kn__n_neighbors': [3, 5]}]
    >>> estimators = [('dt', DecisionTreeClassifier()), ('kn', KNeighborsClassifier())]
    >>> model_search_cv = ModelSearchCV(estimators, param_grids)
    >>> model_search_cv.fit(X, y)
    ModelSearchCV(...)
    >>> sorted(model_search_cv.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...]
    """

    def __init__(
        self,
        estimators,
        param_grids,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise",
        return_train_score=False,
    ):
        estimator = (
            MultiClassifier(estimators)
            if check_estimator_type(estimators) == "classifier"
            else MultiRegressor(estimators)
        )
        est_names = [est_name for est_name, _ in estimators]
        param_grid = check_param_grids(param_grids, est_names)
        super(ModelSearchCV, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.estimators = estimators
        self.param_grids = param_grids

    def fit(self, X, y=None, groups=None, **fit_params):

        # Call superclass fit method
        super(ModelSearchCV, self).fit(X, y, groups=groups, **fit_params)

        # Recreate best estimator attribute
        if hasattr(self, "best_estimator_"):
            self.best_estimator_ = self.best_estimator_.estimator_

        return self
