"""
A wrapper to allow an automated Active Learning procedure for an
experimental environment.
"""

# Author: Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import numpy as np
from sklearn.base import clone
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from ..metrics import SCORERS
from ._selection_methods import UNCERTAINTY_FUNCTIONS
from ._init_methods import init_strategy


class ALWrapper(ClassifierMixin, BaseEstimator):
    """
    Class to perform Active Learning experiments.

    This algorithm is an implementation of an Active Learning framework as
    presented in [1]_. The initialization strategy is WIP.

    Parameters
    ----------
    classifier : classifier object, default=None
        Classifier to be used as Chooser and Predictor.

    generator : generator estimator, default=None
        Generator to be used for artificial data generation within Active
        Learning iterations.

    use_sample_weight : bool, default=False
        Pass ``sample_weights`` as a fit parameter to the generator object. Used to
        generate artificial data around samples with high uncertainty. ``sample_weights``
        is an array-like of shape (n_samples,) containing the probabilities (based on
        uncertainty) for selecting a sample as a center point.

    init_clusterer : clusterer estimator, default=None
        WIP

    init_strategy : WIP, default='random'
        WIP

    selection_strategy : function or {'entropy', 'breaking_ties',\
        'random'}, default='entropy'
        Method used to quantify the chooser's uncertainty level. All predefined functions
        are set up so that a higher value means higher uncertainty (higher likelihood of
        selection) and vice-versa. The uncertainty estimate is used to select the
        instances to be added to the labeled/training dataset.

    max_iter : int, default=None
        Maximum number of iterations allowed.

    n_initial : int, default=100
        Number of observations to include in the initial training dataset.

    increment : int, default=50
        Number of observations to be added to the training dataset at each
        iteration.

    save_classifiers : bool, default=False
        Save classifiers fit at each iteration. These classifiers are stored
        in a list ``self.classifiers_``.

    save_test_scores : bool, default=True
        If ``True``, test scores are saved in the list ``self.test_scores_``.
        Size of the test set is defined with the ``test_size`` parameter.

    auto_load : bool, default=True
        If `True`, the classifier with the best training score is saved in the
        method ``self.classifier_``. It's the classifier object used in the
        ``predict`` method.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to 0.25.

    evaluation_metric : string, default='accuracy'
        Metric used to calculate the test scores. See
        ``research.metrics`` for info on available
        performance metrics.

    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    References
    ----------
    .. [1] Fonseca, J., Douzas, G., Bacao, F. (2021). Increasing the
       Effectiveness of Active Learning: Introducing Artificial Data Generation
       in Active Learning for Land Use/Land Cover Classification. Remote
       Sensing, 13(13), 2619. https://doi.org/10.3390/rs13132619
    """

    def __init__(
        self,
        classifier=None,
        generator=None,
        use_sample_weight=False,
        init_clusterer=None,
        init_strategy='random',
        selection_strategy='entropy',
        max_iter=None,
        n_initial=100,
        increment=50,
        save_classifiers=False,
        save_test_scores=True,
        auto_load=True,
        test_size=None,
        evaluation_metric='accuracy',
        random_state=None
    ):
        self.classifier = classifier
        self.generator = generator
        self.use_sample_weight = use_sample_weight
        self.init_clusterer = init_clusterer
        self.init_strategy = init_strategy
        self.selection_strategy = selection_strategy
        self.max_iter = max_iter
        self.n_initial = n_initial
        self.increment = increment

        # Used to find the optimal classifier
        self.auto_load = auto_load
        self.test_size = test_size
        self.save_classifiers = save_classifiers
        self.save_test_scores = save_test_scores
        self.evaluation_metric = evaluation_metric

        self.random_state = random_state

    def _check(self, X, y):

        if self.evaluation_metric is None:
            self.evaluation_metric_ = SCORERS['accuracy']
        elif type(self.evaluation_metric) == str:
            self.evaluation_metric_ = SCORERS[self.evaluation_metric]
        else:
            self.evaluation_metric_ = self.evaluation_metric

        if self.classifier is None:
            self._classifier = RandomForestClassifier(
                random_state=self.random_state
            )
        else:
            self._classifier = clone(self.classifier)

        if type(self.selection_strategy) == str:
            self.selection_strategy_ = UNCERTAINTY_FUNCTIONS[
                self.selection_strategy
            ]
        else:
            self.selection_strategy_ = self.selection_strategy

        if type(self.use_sample_weights) != bool:
            raise TypeError("``use_sample_weights`` must be of type ``bool``. Got"
                            f" {self.use_sample_weights} instead.")

        self.max_iter_ = self.max_iter \
            if self.max_iter is not None \
            else np.inf

        if self.save_classifiers or self.save_test_scores:
            self.data_utilization_ = []

        if self.save_classifiers:
            self.classifiers_ = []

        if self.save_test_scores:
            self.test_scores_ = []

        if self.auto_load:
            self.classifier_ = None
            self._top_score = 0

        if self.auto_load or self.save_test_scores:
            X, X_test, y, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        else:
            X_test, y_test = (None, None)

        self.increment_ = self.increment
        return X, X_test, y, y_test

    def _get_performance_scores(self):
        data_utilization = [
            i[1] for i in self.data_utilization_
        ]
        test_scores = self.test_scores_
        return data_utilization, test_scores

    def _save_metadata(self, iter_n, classifier, X_test, y_test, selection):
        """Save metadata from a completed iteration."""

        # Get score for current iteration
        if self.save_test_scores or self.auto_load:
            score = self.evaluation_metric_(
                classifier,
                X_test,
                y_test
            )

        # Save classifier
        if self.save_classifiers:
            self.classifiers_.append(classifier)

        # Save test scores
        if self.save_test_scores:
            self.test_scores_.append(score)

            self.data_utilization_.append(
                (selection.sum(), selection.sum()/selection.shape[0])
            )

        # Replace top classifier
        if self.auto_load:
            if score > self._top_score:
                self._top_score = score
                self.classifier_ = classifier
                self.top_score_iter_ = iter_n

    def fit(self, X, y):
        """
        Run an Active Learning procedure from training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : ALWrapper
            Completed Active Learning procedure
        """

        # Original "unlabeled" dataset
        iter_n = 0
        X, X_test, y, y_test = self._check(X, y)
        selection = np.zeros(shape=(X.shape[0])).astype(bool)
        sample_weight = None

        # Supervisor - Get data according to passed initialization method
        self.init_clusterer_, ids = init_strategy(
            X=X,
            n_initial=self.n_initial,
            clusterer=self.init_clusterer,
            selection_method=self.init_strategy,
            random_state=self.random_state
        )

        selection[ids] = True

        while iter_n < self.max_iter_:

            # Generator + Chooser (in this case chooser==Predictor)
            generator = (
                None if self.generator is None else clone(self.generator)
            )
            chooser = clone(self._classifier)

            classifier = Pipeline([
                ('generator', generator),
                ('chooser', chooser)
            ])

            # Generate artificial data and train classifier
            if self.use_sample_weight:
                classifier.fit(X[selection], y[selection], sample_weight)

                # Compute the class probabilities of labeled observations
                labeled_ids = np.argwhere(selection).squeeze()
                probabs_labeled = classifier.predict_proba(X[selection])
                probabs_labeled = np.where(probabs_labeled == 0., 1e-10, probabs_labeled)
            else:
                classifier.fit(X[selection], y[selection])

            # Save metadata from current iteration
            self._save_metadata(
                iter_n, classifier, X_test, y_test, selection
            )

            # Compute the class probabilities of unlabeled observations
            unlabeled_ids = np.argwhere(~selection).squeeze()
            probabs = classifier.predict_proba(X[~selection])
            probabs = np.where(probabs == 0., 1e-10, probabs)

            # Calculate uncertainty
            uncertainty = self.selection_strategy_(probabs)
            if self.use_sample_weight:
                uncertainty_labeled = self.selection_strategy_(probabs_labeled)

            # Get data according to passed selection criterion
            if self.selection_strategy != 'random':
                ids = unlabeled_ids[np.argsort(uncertainty)[::-1][:self.increment]]
            else:
                rng = np.random.RandomState(self.random_state)
                ids = rng.choice(unlabeled_ids, self.increment, replace=False)

            selection[ids] = True

            # Update sample weights for the following iteration
            if self.use_sample_weight:
                sample_weight = np.zeros(selection.shape)
                sample_weight[labeled_ids] = uncertainty_labeled
                sample_weight[ids] = uncertainty
                sample_weight = sample_weight[selection]

            # keep track of iter_n
            iter_n += 1

            # stop if all examples have been included
            if selection.all():
                break
            elif selection.sum()+self.increment_ > y.shape[0]:
                self.increment_ = y.shape[0] - selection.sum()

        return self

    def load_best_classifier(self, X, y):
        """
        Loads the best classifier in the ``self.classifiers_`` list.

        The best classifier is used in the predict method according to the
        performance metric passed.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The test input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : ALWrapper
            Completed Active Learning procedure
        """
        scores = []
        for classifier in self.classifiers_:
            scores.append(
                self.evaluation_metric_(classifier, X, y)
            )

        self.classifier_ = self.classifiers_[np.argmax(scores)]
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
        return self.classifier_.predict(X)
