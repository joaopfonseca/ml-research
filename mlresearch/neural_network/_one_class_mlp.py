from itertools import chain
import numpy as np
from sklearn.base import OutlierMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.metaestimators import available_if
from sklearn.neural_network._multilayer_perceptron import (
    BaseMultilayerPerceptron,
    _STOCHASTIC_SOLVERS,
)
from sklearn.neural_network._base import DERIVATIVES


def hypersphere_loss(outputs, scores, center=1, nu=0.01, radius=0):
    loss = radius**2 + (1 / nu) * np.mean(
        np.hstack((np.zeros(scores.shape), scores)).max(1)
    )
    return loss


class OneClassMLP(OutlierMixin, BaseMultilayerPerceptron):
    """
    Unsupervised One-Class neural network. Can be used to project n-dimensional data into
    a single dimension and for outlier detection. Also used to calculate the $\\alpha$
    and $\\beta$ supports for the performance metrics used to assess synthetic data
    quality.

    Uses the loss function proposed in [1]_, which is optimized using LBFGS or stochastic
    gradient descent. This implementation refers to the architecture described in [2]_
    using the soft-boundary objective.

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(500, 500)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    nu : float, default=0.01
        Hyperparameter used in the loss function.

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term
        is divided by the sample size when added to the loss.

    batch_size : int, default=64
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate at each
          time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when ``solver='sgd'``.

    learning_rate_init : float, default=1e-5
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.

    tol : float, default=0
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'.

    n_iter_no_change : int, default=15
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of loss function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of loss function calls.
        Note that number of loss function calls will be greater than or equal
        to the number of iterations for the `OneClassMLP`.

    Attributes
    ----------

    center_ : float
        The center of the euclidean ball used to calculate the loss.

    radius_ : float
        The radius of the euclidean ball used to calculate the loss.

    loss_ : float
        The current loss computed with the loss function.

    best_loss_ : float or None
        The minimum loss reached by the solver throughout fitting.
        If `early_stopping=True`, this attribute is set to `None`. Refer to
        the `best_validation_score_` fitted attribute instead.

    loss_curve_ : list of shape (`n_iter_`,)
        The ith element in the list represents the loss at the ith iteration.

    t_ : int
        The number of training samples seen by the solver during fitting.

    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        The number of iterations the solver has run.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : str
        Name of the output activation function.

    References
    ----------

    .. [1] Scholkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson,
        R. C. Estimating the support of a high- dimensional distribution. Neural
        computation, 13(7): 1443–1471, 2001.

    .. [2] Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder,
        A., Muller, E., and Kloft, M. Deep one-class classification. In International
        conference on machine learning, pp. 4393–4402. PMLR, 2018.
    """

    def __init__(
        self,
        hidden_layer_sizes=(500, 500),
        activation="relu",
        nu=0.01,
        *,
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate="constant",
        learning_rate_init=1e-5,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=15,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss=None,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=False,
            validation_fraction=None,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.nu = nu

    def _init_center(self, y_pred):
        """
        Initialize hypersphere center c as the mean from an initial forward pass on the
        data.
        """
        self.center_ = y_pred.mean()

    def _init_radius(self, y_pred):
        dist = np.sum((y_pred - self.center_) ** 2, axis=1).reshape(-1, 1)
        self.radius_ = np.quantile(np.sqrt(dist), 1 - self.nu)

    def _initialize(self, layer_units, dtype):
        # set all attributes, allocate weights etc. for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        self.out_activation_ = "identity"

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        dist = np.sum((activations[-1] - self.center_) ** 2, axis=1).reshape(-1, 1)
        scores = dist - self.radius_**2

        # Get loss
        loss = hypersphere_loss(
            outputs=activations[-1],
            scores=scores,
            center=self.center_,
            nu=self.nu,
            radius=self.radius_,
        )

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = scores

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads

    @available_if(lambda est: est._check_solver)
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X):
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Trained MLP model.
        """
        return self._fit(X, incremental=True)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model to data matrix X.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        return self._fit(X, incremental=False)

    def _fit(self, X, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X = self._validate_input(X, incremental, reset=first_pass)

        n_samples, n_features = X.shape

        self.n_outputs_ = 1

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(layer_units, X.dtype)

            y_pred = self._forward_pass_fast(X)
            self._init_center(y_pred)
            self._init_radius(y_pred)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                np.ones(X.shape[0]),
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X,
                np.ones(X.shape[0]),
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
            )

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

    def _validate_input(self, X, incremental, reset):
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        return X

    def predict(self, X):
        """Predict using the multi-layer perceptron model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        """Private predict method with optional input validation"""
        scores = self._forward_pass_fast(X, check_input=check_input)

        if scores.shape[1] == 1:
            return scores.ravel()
        return scores
