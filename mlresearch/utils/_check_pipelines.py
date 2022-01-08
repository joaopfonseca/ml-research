from itertools import product
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import ParameterGrid
from imblearn.pipeline import Pipeline
from rlearn.utils import check_random_states


def check_pipelines(objects_list, random_state, n_runs):
    """Extract estimators and parameters grids."""

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb, rs in product(product(*objects_list), random_states):
        name = "|".join([i[0] for i in comb])

        # name, object, sub grid
        comb = [
            (nm, ob, ParameterGrid(sg))
            if ob is not None
            else (nm, FunctionTransformer(), ParameterGrid(sg))
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
    objects_list, wrapper, random_state, n_runs, wrapped_only=False
):
    wrapper_label = wrapper[0]
    wrapper_obj = wrapper[1]
    wrapper_grid = wrapper[2]

    estimators, param_grids = check_pipelines(objects_list, random_state, n_runs)

    wrapped_estimators = [
        (
            f"{wrapper_label}|{name}",
            clone(wrapper_obj).set_params(**{"classifier": pipeline}),
        )
        for name, pipeline in estimators
    ]

    def _format_param(param):
        return "__".join(param.split("__")[1:])

    wrapped_param_grids = [
        {
            "est_name": [f'{wrapper_label}|{d["est_name"][0]}'],
            **{
                f'{wrapper_label}|{d["est_name"][0]}__classifier__{_format_param(k)}': v
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
