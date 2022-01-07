from copy import deepcopy
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling.base import BaseOverSampler

from .base import BaseActiveLearner


def _random_initialization(self, X=None, y=None, initial_selection=None):
    """Randomly select an initial training dataset."""
    if initial_selection is not None:
        labeled_pool = initial_selection
    else:
        rng = np.random.RandomState(self.random_state)
        labeled_pool = np.zeros(shape=(X.shape[0])).astype(bool)
        ids = rng.choice(np.arange(X.shape[0]), self.n_init_, replace=False)
        if np.unique(y[ids]).shape[0] == 1:
            ids[-1] = rng.choice(
                np.arange(X.shape[0])[y != y[ids][0]], 1, replace=False
            )
        labeled_pool[ids] = True
    return labeled_pool


class StandardAL(BaseActiveLearner):
    """
    Standard Active Learning model with a random initial data selection

    Parameters
    ----------
    classifier : classifier object, default=None
        Classifier or pipeline to be trained in the iterative process. If None, defaults
        to sklearn's RandomForestClassifier with default parameters and uses the
        ``random_state`` passed in the Active Learning model.

    acquisition_func : function or {'entropy', 'breaking_ties',\
        'random'}, default=None
        Method used to quantify the prediction's uncertainty level. All predefined
        functions are set up so that a higher value means higher uncertainty (higher
        likelihood of selection) and vice-versa. The uncertainty estimate is used to
        select the instances to be added to the labeled/training dataset. Acquisition
        functions may be added or changed in the ``UNCERTAINTY_FUNCTIONS`` dictionary.
        If None, defaults to "random".

    n_init : int or float, default=None
        Number of observations to include in the initial training dataset. If
        ``n_init`` < 1, then the corresponding percentage of the original dataset
        will be used as the initial training set. If None, defaults to 2% of the size of
        the original dataset.

    budget : int or float, default=None
        Number of observations to be added to the training dataset at each iteration. If
        ``budget`` < 1, then the corresponding percentage of the original dataset will be
        used as the initial training set. If None, defaults to 2% of the size of the
        original dataset.

    max_iter : int, default=None
        Maximum number of iterations allowed. If None, the experiment will run until 100%
        of the dataset is added to the training set.

    evaluation_metric : string, default='accuracy'
        Metric used to calculate the test scores. See
        ``research.metrics`` for info on available
        performance metrics.

    continue_training : bool, default=False
        If ``False``, fit a new classifier at each iteration. If ``True``, the
        classifier fitted in the previous iteration is used for further training in
        subsequent iterations.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    Attributes
    ----------
    acquisition_func_ : function
        Method used to calculate the classification uncertainty at each iteration.
    evaluation_metric_ : scorer
        Metric used to estimate the performance of the AL classifier at each iteration.
    classifier_ : estimator object
        The classifier used in the iterative process. It is the classifier fitted in the
        last iteration.
    metadata_ : dict
        Contains the performance estimations, classifiers, labeled pool mask and original
        dataset.
    n_init_ : int
        Number of observations included in the initial training dataset.
    budget_ : int
        Number of observations to be added to the training set per iteration. Also known
        as budget.
    max_iter_ : int
        Maximum number of iterations allowed.
    labeled_pool_ : array-like of shape (n_samples,)
        Mask that filters the labeled observations from the original dataset.
    """

    def _initialization(self, X=None, y=None, initial_selection=None):
        labeled_pool = _random_initialization(self, X, y, initial_selection)
        return labeled_pool

    def _iteration(self, X, y, **kwargs):
        self.classifier_.fit(X[self.labeled_pool_], y[self.labeled_pool_])
        return self.classifier_.predict_proba(X[~self.labeled_pool_])

    def _oracle(self, probabilities):
        uncertainties = self.acquisition_func_(probabilities)
        unlabeled_ids = np.argwhere(~self.labeled_pool_).squeeze()

        ids = (
            unlabeled_ids[np.argsort(uncertainties)[::-1][: self.budget_]]
            if unlabeled_ids.ndim >= 1
            else unlabeled_ids.flatten()[0]
        )

        self.labeled_pool_[ids] = True
        return self


class AugmentationAL(BaseActiveLearner):
    """
    Active Learning with pipelined Data Augmentation. This method is implemented and
    analysed in a working paper.

    Parameters
    ----------
    classifier : classifier object, default=None
        Classifier or pipeline to be trained in the iterative process. If None, defaults
        to sklearn's RandomForestClassifier with default parameters and uses the
        ``random_state`` passed in the Active Learning model.

    generator : generator estimator, default=None
        Generator to be used for artificial data generation within Active
        Learning iterations.

    param_grid : dict or list of dictionaries
        Used to optimize the classifier and generator hyperparameters at each iteration
        via cross-validated grid-search. If None, parameter tuning is skipped.
        Dictionary with parameters names (``str``) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which case the
        grids spanned by each dictionary in the list are explored. This enables searching
        over any sequence of parameter settings.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Used to optimize the
        classifier and generator hyperparameters at each iteration.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    acquisition_func : function or {'entropy', 'breaking_ties',\
        'random'}, default=None
        Method used to quantify the prediction's uncertainty level. All predefined
        functions are set up so that a higher value means higher uncertainty (higher
        likelihood of selection) and vice-versa. The uncertainty estimate is used to
        select the instances to be added to the labeled/training dataset. Acquisition
        functions may be added or changed in the ``UNCERTAINTY_FUNCTIONS`` dictionary.
        If None, defaults to "random".

    n_init : int or float, default=None
        Number of observations to include in the initial training dataset. If
        ``n_init`` < 1, then the corresponding percentage of the original dataset
        will be used as the initial training set. If None, defaults to 2% of the size of
        the original dataset.

    budget : int or float, default=None
        Number of observations to be added to the training dataset at each iteration. If
        ``budget`` < 1, then the corresponding percentage of the original dataset will be
        used as the initial training set. If None, defaults to 2% of the size of the
        original dataset.

    max_iter : int, default=None
        Maximum number of iterations allowed. If None, the experiment will run until 100%
        of the dataset is added to the training set.

    evaluation_metric : string, default='accuracy'
        Metric used to calculate the test scores. See
        ``research.metrics`` for info on available
        performance metrics.

    continue_training : bool, default=False
        If ``False``, fit a new classifier at each iteration. If ``True``, the
        classifier fitted in the previous iteration is used for further training in
        subsequent iterations.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    Attributes
    ----------
    acquisition_func_ : function
        Method used to calculate the classification uncertainty at each iteration.
    evaluation_metric_ : scorer
        Metric used to estimate the performance of the AL classifier at each iteration.
    classifier_ : estimator object
        The classifier used in the iterative process. It is the classifier fitted in the
        last iteration.
    metadata_ : dict
        Contains the performance estimations, classifiers, labeled pool mask and original
        dataset.
    n_init_ : int
        Number of observations included in the initial training dataset.
    budget_ : int
        Number of observations to be added to the training set per iteration.
    max_iter_ : int
        Maximum number of iterations allowed.
    labeled_pool_ : array-like of shape (n_samples,)
        Mask that filters the labeled observations from the original dataset.
    """

    def __init__(
        self,
        classifier: (BaseEstimator, ClassifierMixin) = None,
        generator: BaseOverSampler = None,
        param_grid: dict = None,
        cv=None,
        acquisition_func=None,
        n_init: (int, float) = None,
        budget: (int, float) = None,
        max_iter: int = None,
        evaluation_metric=None,
        continue_training: bool = False,
        random_state: int = None,
    ):
        super().__init__(
            classifier=classifier,
            acquisition_func=acquisition_func,
            n_init=n_init,
            budget=budget,
            max_iter=max_iter,
            evaluation_metric=evaluation_metric,
            continue_training=continue_training,
            random_state=random_state,
        )
        self.generator = generator
        self.param_grid = param_grid
        self.cv = cv

    def _check(self, X, y):
        super()._check(X, y)

        # Generator
        if (
            self.generator is not None
            and hasattr(self.generator, "random_state")
            and self.generator.random_state is None
            and self.random_state is not None
        ):
            # Check random state
            self._generator = clone(self.generator)
            self._generator.set_params(random_state=self.random_state)

            # Add generator to classifier as a pipeline
            generator = clone(self._generator)
            classifier = clone(self._classifier)
            self._classifier = Pipeline(
                [("generator", generator), ("classifier", classifier)]
            )

        # Check if parameters in param_grid are valid
        if type(self.param_grid) == dict:
            for key in self.param_grid.keys():
                if key not in self._classifier.get_params():
                    raise ValueError(
                        f"Invalid parameter {key} for generator or classifier in {self} "
                        "check the list of available parameters with "
                        "`almodel._classifier.get_params().keys()`."
                    )
        elif self.param_grid is not None:
            raise TypeError(
                f"``param_grid`` must be a dict or None. Got {self.param_grid} instead."
            )

    def _save_metadata(self, X, y, **kwargs):
        super()._save_metadata(X, y, **kwargs)
        if hasattr(self, "classifier_") and type(self.classifier_) == GridSearchCV:
            self.metadata_[self._current_iter]["parameters"] = {
                k: v for k, v in self.classifier_.best_estimator_.get_params().items()
                if k in self.param_grid.keys()
            }

    def _check_cross_validation(self, y):
        """Define cross-validation object"""

        min_frequency = np.unique(y, return_counts=True)[-1].min()
        cv = deepcopy(self.cv)

        if hasattr(self.cv, "n_splits"):
            cv.n_splits = min(min_frequency, cv.n_splits)
        elif type(self.cv) == int:
            cv = min(min_frequency, cv)
        elif cv is None:
            cv = min(min_frequency, 5)
        else:
            raise TypeError(
                "``cv`` object must be of type int or cross-validation generator. Got "
                f"{self.cv} instead"
            )
        return cv

    def _initialization(self, X=None, y=None, initial_selection=None):
        labeled_pool = _random_initialization(self, X, y, initial_selection)
        return labeled_pool

    def _iteration(self, X, y, **kwargs):

        # Set up parameter tuning within iterations
        cv = self._check_cross_validation(y[self.labeled_pool_])
        if self.param_grid is not None and cv != 1:
            self.classifier_ = GridSearchCV(
                estimator=self.classifier_,
                param_grid=self.param_grid,
                scoring=self.evaluation_metric_,
                cv=cv,
                refit=True,
            )
        self.classifier_.fit(X[self.labeled_pool_], y[self.labeled_pool_])
        return self.classifier_.predict_proba(X[~self.labeled_pool_])

    def _oracle(self, probabilities):
        uncertainties = self.acquisition_func_(probabilities)
        unlabeled_ids = np.argwhere(~self.labeled_pool_).squeeze()

        ids = (
            unlabeled_ids[np.argsort(uncertainties)[::-1][: self.budget_]]
            if unlabeled_ids.ndim >= 1
            else unlabeled_ids.flatten()[0]
        )

        self.labeled_pool_[ids] = True
        return self
