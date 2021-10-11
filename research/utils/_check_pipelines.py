from rlearn.utils import check_random_states
from itertools import product
from imblearn.pipeline import Pipeline
from sklearn.base import clone


def check_pipelines(objects_list, random_state, n_runs):
    """Extract estimators and parameters grids."""

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb, rs in product(product(*objects_list), random_states):
        name = '|'.join([i[0] for i in comb])

        # name, object, sub grid
        comb = [(nm, ob, sg) for nm, ob, sg in comb if ob is not None]

        if name not in [n[0] for n in pipelines]:
            pipelines.append((name, Pipeline([(nm, ob) for nm, ob, _ in comb])))

        grid = {'est_name': [name]}
        for obj_name, obj, sub_grid in comb:
            param_prefix = f'{obj_name}' if len(comb) == 1 else '{name}__{obj_name}'

            if 'random_state' in obj.get_params().keys():
                grid[f'{param_prefix}__random_state'] = [rs]
            for param, values in sub_grid.items():
                grid[f'{param_prefix}__{param}'] = values

        # Avoid multiple runs over pipelines without random state
        if grid not in param_grid:
            param_grid.append(grid)

    return pipelines, param_grid


def check_pipelines_wrapper(
    objects_list,
    wrapper,
    random_state,
    n_runs,
    wrapped_only=False
):
    wrapper_label = wrapper[0]
    wrapper_obj = wrapper[1]
    wrapper_grid = wrapper[2]

    estimators, param_grids = check_pipelines(
        objects_list, random_state, n_runs
    )

    wrapped_estimators = [
        (
            f'{wrapper_label}|{name}',
            clone(wrapper_obj).set_params(**{'classifier': pipeline})
        )
        for name, pipeline in estimators
    ]

    wrapped_param_grids = [
        {
            'est_name': [f'{wrapper_label}|{d["est_name"][0]}'],
            **{
                f'{wrapper_label}|{d["est_name"][0]}__classifier__{k}': v
                for k, v in d.items() if k != 'est_name'},
            **{f'{wrapper_label}|{d["est_name"][0]}__{k}': v
                for k, v in wrapper_grid.items()}
        } for d in param_grids
    ]
    if wrapped_only:
        return wrapped_estimators, wrapped_param_grids
    else:
        return (
            estimators + wrapped_estimators,
            param_grids + wrapped_param_grids
        )
