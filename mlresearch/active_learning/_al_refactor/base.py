"""
Base class for Active Learning models
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from abc import abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.utils import check_X_y
from sklearn.ensemble import RandomForestClassifier

from ._acquisition_functions import ACQUISITION_FUNCTIONS
from ...metrics import SCORERS


class BaseActiveLearner(BaseEstimator, ClassifierMixin):
    """
    Base class to implement Active Learning models.

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

    def __init__(
        self,
        classifier: (BaseEstimator, ClassifierMixin) = None,
        acquisition_func=None,
        n_init: (int, float) = None,
        budget: (int, float) = None,
        max_iter: int = None,
        evaluation_metric=None,
        continue_training: bool = False,
        random_state: int = None,
    ):
        self.classifier = classifier
        self.acquisition_func = acquisition_func
        self.n_init = n_init
        self.budget = budget
        self.max_iter = max_iter
        self.evaluation_metric = evaluation_metric
        self.continue_training = continue_training
        self.random_state = random_state

    @abstractmethod
    def _initialization(self, X, y=None, initial_selection=None):
        pass

    @abstractmethod
    def _iteration(self, X_lab, y_lab, X_pool=None, y_pool=None):
        pass

    @abstractmethod
    def _oracle(self, probabilities):
        pass

    def _early_stop(self):
        # more stopping conditions may be added here in the future
        return self.labeled_pool_.all()

    def _check(self, X, y):
        """Check initialization parameters to run an AL model."""

        # Classifier
        if self.classifier is None:
            self._classifier = RandomForestClassifier(random_state=self.random_state)
        else:
            self._classifier = clone(self.classifier)

        # Random state
        if (
            hasattr(self._classifier, "random_state")
            and self._classifier.random_state is None
            and self.random_state is not None
        ):
            self._classifier.set_params(random_state=self.random_state)

        # Acquisition function
        if self.acquisition_func is None:
            self.acquisition_func_ = ACQUISITION_FUNCTIONS["random"]
        elif type(self.acquisition_func) == str:
            self.acquisition_func_ = ACQUISITION_FUNCTIONS[self.acquisition_func]
        else:
            self.acquisition_func_ = self.acquisition_func

        # Number of initial observations
        if self.n_init is None or self.n_init < 1:
            perc = 0.02 if self.budget is None else self.budget
            n_init_ = int(np.round(perc * X.shape[0]))
        else:
            n_init_ = self.n_init
        self.n_init_ = n_init_ if n_init_ >= 2 else 2

        # Budget
        if self.budget is None or self.budget < 1:
            perc = 0.02 if self.budget is None else self.budget
            budget_ = int(np.round(perc * X.shape[0]))
        else:
            budget_ = self.budget
        self.budget_ = budget_ if budget_ >= 1 else 1

        # Maximum iterations
        self.max_iter_ = (
            self.max_iter
            if self.max_iter is not None
            else int(np.round((X.shape[0] - self.n_init_) / self.budget_))
        )

        # Evaluation metric
        if self.evaluation_metric is None:
            self.evaluation_metric_ = SCORERS["accuracy"]
        elif type(self.evaluation_metric) == str:
            self.evaluation_metric_ = SCORERS[self.evaluation_metric]
        else:
            self.evaluation_metric_ = self.evaluation_metric

        # Train a different classifier per iteration (as an alternative continue training
        # the classifier from previous iterations)
        if type(self.continue_training) == bool:
            self._continue_training = self.continue_training
        else:
            raise TypeError(
                "``continue_training`` must be of type ``bool``. Got"
                f" {self.continue_training} instead."
            )

        # Set up basic attributes
        if not hasattr(self, "_current_iter"):
            self._current_iter = 0
            self.metadata_ = {
                "data": (X, y),
                "performance_metric": self.evaluation_metric_._score_func.__name__,
            }
        else:
            # Raise error, model is already fitted
            raise StopIteration(f"Active Learning model {self} is already initialized.")

    def _save_metadata(self, X, y, **kwargs):
        self.metadata_[self._current_iter] = {"labeled_pool": self.labeled_pool_.copy()}

        # Save performance in the training set
        if hasattr(self, "classifier_"):
            self.metadata_[self._current_iter][
                "train_performance"
            ] = self.evaluation_metric_(
                self.classifier_, X[self.labeled_pool_], y[self.labeled_pool_]
            )

        # Save performance in the test set
        if hasattr(self, "classifier_") and self._has_test:
            self.metadata_[self._current_iter][
                "test_performance"
            ] = self.evaluation_metric_(
                self.classifier_, X[self.labeled_pool_], y[self.labeled_pool_]
            )

        # Save classifier
        if hasattr(self, "classifier_") and not self._continue_training:
            self.metadata_[self._current_iter]["classifier"] = self.classifier_

    def initialization(self, X, y, initial_selection=None, **kwargs):
        self._check(X, y)
        self.labeled_pool_ = self._initialization(
            X=X, y=y, initial_selection=initial_selection
        )
        self._save_metadata(X[self.labeled_pool_], y[self.labeled_pool_], **kwargs)
        return self

    def iteration(self, X, y, **kwargs):
        if not hasattr(self, "_current_iter"):
            # Raise error, model is not initialized
            raise StopIteration(
                f"Active Learning model {self.__name__} is not initialized yet."
            )

        # Create iteration's classifier if the previous one is not going to be
        # used
        if not self._continue_training:
            self.classifier_ = clone(self._classifier)

        # Run iteration
        probabilities = self._iteration(X, y, **kwargs)

        # Update labeled pool
        self._oracle(probabilities)

        # Save results from iteration
        self._current_iter += 1
        self._save_metadata(X, y, **kwargs)

        return self

    def fit(self, X, y, **kwargs):
        """
        Fit an Active Learning model from training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : Active Learning Classifier
            Fitted Active Learning model.
        """

        # Check if parameters are properly set up
        X, y = check_X_y(X, y, **kwargs)

        # Check if there is a Test set
        self._has_test = True if "X_test" in kwargs else False

        # Initialize the AL model
        self.initialization(X, y, **kwargs)

        # Create base classifier if it is going to be trained across iterations
        if self._continue_training and not hasattr(self, "classifier_"):
            self.classifier_ = clone(self._classifier)

        # Iterate
        for iter_ in range(1, self.max_iter_ + 1):

            self.iteration(X, y, **kwargs)

            if self._early_stop():
                break

        return self

    def predict(self, X):
        """
        Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        if not self._continue_training and self._has_test:
            iter_perf = np.array(
                [
                    [i, self.metadata_[i]["test_performance"]]
                    for i in self.metadata_.keys()
                ]
            )
            index = iter_perf[np.argmax(iter_perf, axis=0)[1]][0]
            return self.metadata_[index]["classifier"].predict(X)
        else:
            return self.classifier_.predict(X)
