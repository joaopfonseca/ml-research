from itertools import product
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import ParameterGrid
from imblearn.utils import Substitution
from imblearn.pipeline import Pipeline
from imblearn.utils._docstring import _random_state_docstring


@Substitution(
    random_state=_random_state_docstring,
)
def check_random_states(random_state, n_runs):
    """
    Create random states for experiments. Used to create seeds for different
    initializations.

    Parameters
    ----------
    {random_state}

    n_runs : int
        Number of initializations.

    Returns
    -------
    random_states : list
        A list of random states with length ``n_runs``.
    """
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2**32 - 1, dtype="uint32") for _ in range(n_runs)]


@Substitution(
    random_state=_random_state_docstring,
)
def check_pipelines(*objects_list, random_state, n_runs):
    """
    Extract estimators and parameter grids to be passed to ModelSearchCV. This enables
    searching over any sequence of parameter settings and objects.

    Parameters
    ----------
    *objects_list : sequence of lists
        Lists of objects to be chained in a pipeline in the passed order. Each list
        must contain tuples composed of (``<obj_name>``, ``<object>``,
        ``<parameter_values_dict>``).

    {random_state}

    n_runs : int
        Number of initializations.

    Returns
    -------
    estimators : List of Pipelines with all combinations among the passed lists of
        objects.

    param_grids : List of dictionaries with estimator and parameter names (``str``) as
        keys and lists of parameter settings to try as values.
    """

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb, rs in product(product(*objects_list), random_states):
        name = "|".join([i[0] for i in comb])

        # name, object, sub grid
        comb = [
            (
                (nm, ob, ParameterGrid(sg))
                if ob is not None
                else (nm, FunctionTransformer(), ParameterGrid(sg))
            )
            for nm, ob, sg in comb
        ]

        # Create estimator
        if name not in [n[0] for n in pipelines]:
            est = Pipeline([(nm, ob) for nm, ob, _ in comb])
            pipelines.append((name, est))

        # Create intermediate parameter grids
        sub_grids = [
            [{f"{nm}__{k}": v for k, v in param_def.items()} for param_def in sg]
            for nm, obj, sg in comb
        ]

        # Create parameter grids
        for sub_grid in product(*sub_grids):
            param_prefix = f"{name}__"
            grid = {"est_name": [name]}
            grid.update(
                {f"{param_prefix}{k}": [v] for d in sub_grid for k, v in d.items()}
            )
            random_states = {
                f"{param_prefix}{param}": [rs]
                for param in est.get_params()
                if "random_state" in param
            }
            grid.update(random_states)

            # Avoid multiple runs over pipelines without random state
            if grid not in param_grid:
                param_grid.append(grid)

    return pipelines, param_grid


def check_pipelines_wrapper(
    *objects_list,
    wrapper,
    random_state,
    n_runs,
    estimator_param="classifier",
    wrapped_only=True,
):
    """
    Extract estimators within a wrapper object and parameter grids to be passed to
    ModelSearchCV. This enables searching over any sequence of parameter settings and
    objects.

    Parameters
    ----------
    *objects_list : sequence of lists
        Lists of objects to be chained in a pipeline in the passed order. Each list
        must contain tuples composed of (``<obj_name>``, ``<object>``,
        ``<parameter_values_dict>``).

    wrapper : tuple or tuple
        Wrapper object to which the lists of objects will be passed. Must be structured
        as (``<obj_name>``, ``<object>``, ``<parameter_values_dict>``) and .

    {random_state}

    n_runs : int
        Number of initializations.

    estimator_param : str, default="classifier"
        Name of the parameter in the wrapper object where the estimators will be passed.

    wrapped_only : bool, default=True
        Return only the wrapped estimators. If ``False``, returns both the wrapped and
        the original objects.

    Returns
    -------
    wrapped_estimators : List of Pipelines with all combinations among the passed lists
        of objects.

    wrapped_param_grids : List of dictionaries with estimator and parameter names
        (``str``) as keys and lists of parameter settings to try as values.
    """

    wrapper_label = wrapper[0]
    wrapper_obj = wrapper[1]
    wrapper_grid = wrapper[2]

    estimators, param_grids = check_pipelines(
        *objects_list, random_state=random_state, n_runs=n_runs
    )

    wrapped_estimators = [
        (
            f"{wrapper_label}|{name}",
            clone(wrapper_obj).set_params(**{estimator_param: pipeline}),
        )
        for name, pipeline in estimators
    ]

    def _format_param(param):
        return "__".join(param.split("__")[1:])

    wrapped_param_grids = [
        {
            "est_name": [f'{wrapper_label}|{d["est_name"][0]}'],
            **{
                f'{wrapper_label}|{d["est_name"][0]}__{estimator_param}__'
                + f"{_format_param(k)}": v
                for k, v in d.items()
                if k != "est_name"
            },
            **{
                f'{wrapper_label}|{d["est_name"][0]}__{k}': v
                for k, v in wrapper_grid.items()
            },
        }
        for d in param_grids
    ]
    if wrapped_only:
        return wrapped_estimators, wrapped_param_grids
    else:
        return (estimators + wrapped_estimators, param_grids + wrapped_param_grids)


def check_param_grids(param_grids, est_names):
    """Check the parameters grids to use with
    parametrized estimators."""

    # Check the parameters grids
    flat_param_grids = [
        param_grid for param_grid in list(ParameterGrid(param_grids)) if param_grid
    ]

    # Append existing estimators names
    param_grids = []
    for param_grid in flat_param_grids:

        # Get estimator name
        est_name = param_grid.pop("est_name", None)

        # Modify values
        param_grid = {param: [val] for param, val in param_grid.items()}

        # Check estimators prefixes
        params_prefixes = set([param.split("__")[0] for param in param_grid.keys()])
        if not params_prefixes.issubset(est_names):
            raise ValueError(
                "Parameters prefixes are not subset of parameter `est_names`."
            )
        if len(params_prefixes) > 1:
            raise ValueError("Parameters prefixes are not unique.")
        if est_name is not None and len(params_prefixes.union([est_name])) > 1:
            raise ValueError(
                "Parameters prefixes and parameter `est_name` are not unique."
            )
        param_grid["est_name"] = (
            [est_name] if est_name is not None else list(params_prefixes)
        )

        # Append parameter grid
        param_grids.append(param_grid)

    # Append missing estimators names
    current_est_names = set([param_grid["est_name"][0] for param_grid in param_grids])
    missing_est_names = set(est_names).difference(current_est_names)
    for est_name in missing_est_names:
        param_grids.append({"est_name": [est_name]})

    return param_grids


def check_estimator_type(estimators):
    """Returns the type of estimators."""
    estimator_types = set([estimator._estimator_type for _, estimator in estimators])
    if len(estimator_types) > 1:
        raise ValueError(
            "Both classifiers and regressors were found. "
            "A single estimator type should be included."
        )
    return estimator_types.pop()
