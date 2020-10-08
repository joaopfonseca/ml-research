

def check_pipelines(objects_list, random_state, n_runs):
    """Extract estimators and parameters grids."""

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb in product(*objects_list):
        name  = '|'.join([i[0] for i in comb])
        comb = [(nm,ob,grd) for nm,ob,grd in comb if ob is not None] # name, object, grid

        pipelines.append((name, Pipeline([(nm,ob) for nm,ob,_ in comb])))

        grids = {'est_name': [name]}
        for obj_name, obj, sub_grid in comb:
            if 'random_state' in obj.get_params().keys():
                grids[f'{name}__{obj_name}__random_state'] = random_states
            for param, values in sub_grid.items():
                grids[f'{name}__{obj_name}__{param}'] = values
        param_grid.append(grids)

    return pipelines, param_grid

def check_pipelines_wrapper(
    objects_list,
    wrapper,
    random_state,
    n_runs,
):
    wrapper_label = wrapper[0]
    wrapper_obj = wrapper[1]
    wrapper_grid = wrapper[2]

    estimators, param_grids = check_pipelines(
        objects_list, random_state, n_runs
    )

    wrapped_estimators = [
        (f'{wrapper_label}|{name}', clone(wrapper_obj).set_params(**{'classifier':pipeline})) for name, pipeline in estimators
    ]

    wrapped_param_grids = [
        {
            'est_name': [f'{wrapper_label}|{d["est_name"][0]}'],
            **{k.replace(d["est_name"][0], f'{wrapper_label}|{d["est_name"][0]}__classifier'):v
                for k, v in d.items() if k!='est_name'},
            **{f'{wrapper_label}|{d["est_name"][0]}__{k}':v for k, v in wrapper_grid.items()}
        } for d in param_grids
    ]

    return estimators + wrapped_estimators, param_grids + wrapped_param_grids
