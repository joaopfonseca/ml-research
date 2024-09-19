import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, ParameterGrid
import sklearn.model_selection._search_successive_halving

from ._search import MultiClassifier, MultiRegressor
from ..utils._check_pipelines import check_estimator_type, check_param_grids


def _model_top_k(results, k, itr):
    """
    Apply model-wise top-k selection to the results of a search.
    """
    # Return all the candidates of a given iteration
    iteration, est_name, mean_test_score, params = (
        np.asarray(a)
        for a in (
            results["iter"],
            results["param_est_name"],
            results["mean_test_score"],
            results["params"],
        )
    )

    # Get the indices of the results that correspond to the given iteration
    iter_indices = np.flatnonzero(iteration == itr)
    names = est_name[iter_indices]
    scores = mean_test_score[iter_indices]

    # Get the best candidates for each estimator
    unique_names = np.unique(names)
    k_ = int(np.ceil(k / len(unique_names)))
    params_candidates = []
    for name in unique_names:
        scores_ = scores[names == name]
        iter_indices_ = iter_indices[names == name]
        # argsort() places NaNs at the end of the array so we move NaNs to the
        # front of the array so the last `k` items are the those with the
        # highest scores.
        sorted_indices = np.roll(
            np.argsort(scores_), np.count_nonzero(np.isnan(scores_))
        )
        top_params = np.array(params[iter_indices_][sorted_indices[-k_:]])
        params_candidates.append(top_params)

    return np.concatenate(params_candidates)


# Monkey patch the top_k method, not the best way to do this but it works.
# Better than copying the entire _run_search method only to change this function I guess.
sklearn.model_selection._search_successive_halving._top_k = _model_top_k


class HalvingModelSearchCV(HalvingGridSearchCV):
    """Search over specified parameter values for a collection of estimators with
    successive halving.

    The search strategy starts evaluating all the candidates with a small amount of
    resources and iteratively selects the best candidates, using more and more resources.

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

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.

    max_resources : int, default='auto'
        The maximum amount of resource that any candidate is allowed to use
        for a given iteration. By default, this is set to ``n_samples`` when
        ``resource='n_samples'`` (default), else an error is raised.

    min_resources : {'exhaust', 'smallest'} or int, default='exhaust'
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Equivalently, this defines the amount of
        resources `r0` that are allocated for each candidate at the first
        iteration.

        - 'smallest' is a heuristic that sets `r0` to a small value:

            - ``n_splits * 2`` when ``resource='n_samples'`` for a regression
              problem
            - ``n_classes * n_splits * 2`` when ``resource='n_samples'`` for a
              classification problem
            - ``1`` when ``resource != 'n_samples'``

        - 'exhaust' will set `r0` such that the **last** iteration uses as
          much resources as possible. Namely, the last iteration will use the
          highest value smaller than ``max_resources`` that is a multiple of
          both ``min_resources`` and ``factor``. In general, using 'exhaust'
          leads to a more accurate estimator, but is slightly more time
          consuming.

        Note that the amount of resources used at each iteration is always a
        multiple of ``min_resources``.

    aggressive_elimination : bool, default=False
        This is only relevant in cases where there isn't enough resources to
        reduce the remaining candidates to at most `factor` after the last
        iteration. If ``True``, then the search process will 'replay' the
        first iteration for as long as needed until the number of candidates
        is small enough. This is ``False`` by default, which means that the
        last iteration may evaluate more than ``factor`` candidates. See
        :ref:`aggressive_elimination` for more details.

    scoring : string, callable, list/tuple, dict or None, default=None
        A single string or a callable to evaluate the predictions on the
        test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        Note that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        If ``None``, the estimator's score method is used.

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

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Also used for random uniform
        sampling from lists of possible values instead of scipy.stats
        distributions.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : integer, default=0
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``min_resources_``, the actual
        number of resources used at the last iteration may be smaller than
        ``max_resources_``.

    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

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

    All parameter combinations scored with a NaN will share the lowest rank.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from mlresearch.model_selection import HalvingModelSearchCV
    >>> X, y, *_ = load_breast_cancer().values()
    >>> param_grids = [{'dt__max_depth': [3, 6]}, {'kn__n_neighbors': [3, 5]}]
    >>> estimators = [('dt', DecisionTreeClassifier()), ('kn', KNeighborsClassifier())]
    >>> model_search_cv = HalvingModelSearchCV(estimators, param_grids)
    >>> model_search_cv.fit(X, y)
    HalvingModelSearchCV(...)
    >>> sorted(model_search_cv.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...]


    """

    def __init__(
        self,
        estimators,
        param_grids,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score="raise",
        return_train_score=False,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        estimator = (
            MultiClassifier(estimators)
            if check_estimator_type(estimators) == "classifier"
            else MultiRegressor(estimators)
        )
        super(HalvingGridSearchCV, self).__init__(
            estimator=estimator,
            factor=factor,
            resource=resource,
            max_resources=max_resources,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
            cv=cv,
            scoring=scoring,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            random_state=random_state,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.estimators = estimators
        self.param_grids = param_grids

    def _generate_candidate_params(self):
        est_names = [est_name for est_name, _ in self.estimators]
        param_grid = check_param_grids(self.param_grids, est_names)
        return ParameterGrid(param_grid)

    def fit(self, X, y=None, groups=None, **fit_params):

        # Call superclass fit method
        super(HalvingGridSearchCV, self).fit(X, y, groups=groups, **fit_params)

        # Recreate best estimator attribute
        if hasattr(self, "best_estimator_"):
            self.best_estimator_ = self.best_estimator_.estimator_

        return self
