"""
Analyze the experimental results.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join
from zipfile import ZipFile
import numpy as np
import pandas as pd
from rlearn.tools import summarize_datasets
from research.utils import generate_paths, load_datasets
from research.datasets import MulticlassDatasets

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

DATASETS_NAMES = [
    d.replace('fetch_', '')
    for d in dir(MulticlassDatasets())
    if d.startswith('fetch_')
]

DATASETS_MAPPING = dict([
    (d, ''.join([i[0] for i in d.split('_')]).upper())
    if (len(d.split('_')) > 1)
    else (d, d.title())
    for d
    in DATASETS_NAMES
])

METRICS_MAPPING = dict([
    ('accuracy', 'Accuracy'),
    ('f1_macro', 'F-score'),
    ('geometric_mean_score_macro', 'G-mean')
])

GROUP_KEYS = [
    'Dataset',
    'Estimator',
    'Evaluation Metric',
    'Selection Criterion'
]


def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={
            'Dataset name': 'Dataset', 'Imbalance Ratio': 'IR',
        }).set_index('Dataset')\
        .join(pd.Series(dict(
            [(name, dat[-1].unique().size) for name, dat in datasets]
        ), name='Classes'))\
        .reset_index()
    return summarized


def select_results(results):
    """
    Computes mean and std across all splits and runs from the original
    experiment's data.
    """

    results = results.copy()

    # Extract info from the params dict
    for param in ['evaluation_metric', 'selection_strategy']:
        results[param] = results.params.apply(
            lambda x: (
                x[param]
                if param in x.keys()
                else np.nan
            )
        )

    # Format column names
    results.rename(columns={
        'param_est_name': 'Estimator',
        'evaluation_metric': 'Evaluation Metric',
        'selection_strategy': 'Selection Criterion'
    }, inplace=True)

    # Drop random states from params
    # Convert to params to string in order to use groupby
    results.params = results.params\
        .apply(lambda x: {
            k: v
            for k, v in x.items()
            if ('random_state' not in k)
            and ('evaluation_metric' not in k)
            and ('selection_strategy' not in k)
        })\
        .astype(str)

    scoring_cols = {
        col: '_'.join(col.split('_')[2:])
        for col in results.columns
        if 'mean_test' in col
    }

    # Group data using GROUP_KEYS
    scoring_mapping = {
        scorer_name: [np.mean, np.std]
        for scorer_name in scoring_cols.values()
    }

    results_ = results.rename(columns=scoring_cols)\
        .groupby(GROUP_KEYS, dropna=False)

    # Get standard deviations
    stds = results_.apply(
        lambda dat: [
            np.std(
                dat[
                    dat.columns[dat.columns.str.contains(scorer)
                                & dat.columns.str.contains('split')]
                ].values.flatten()
            )
            for scorer in scoring_mapping.keys()
        ]
    )

    results = results_.agg(scoring_mapping)

    mask_cols = np.array(list(results.columns))[:, 1] == 'std'
    values_arr = results.values
    values_arr[:, mask_cols] = np.array(stds.tolist())

    return pd.DataFrame(
        values_arr,
        columns=results.columns,
        index=results.index
    )


def calculate_wide_optimal(results):
    core_metrics = results\
        .reset_index()['Evaluation Metric']\
        .dropna().unique()

    res_ = []
    for m in ['mean', 'std']:
        res = results.loc[
            :, results.columns.get_level_values(1) == m
        ]
        res.columns = res.columns.get_level_values(0)
        res = res.reset_index()\
            .drop(columns=['Evaluation Metric', 'Selection Criterion'])\
            .loc[:, ['Dataset', 'Estimator', *core_metrics]]\
            .melt(id_vars=['Dataset', 'Estimator'])\
            .rename(columns={'value': m})\
            .set_index(['Dataset', 'Estimator', 'variable'])
        res_.append(res)

    wide_optimal = pd.concat(res_, axis=1).reset_index()\
        .groupby(
            ['Dataset', 'Estimator', 'variable']
        )\
        .apply(
            lambda dat: dat.iloc[np.argmax(dat['mean'])]
        ).reset_index(drop=True)

    (
        _,
        wide_optimal['Generator'],
        wide_optimal['Classifier']
    ) = np.array(
        wide_optimal.Estimator.apply(
            lambda x: x.split('|')
            if len(x.split('|')) == 3
            else [np.nan, np.nan, x.split('|')[1]]
        ).tolist()
    ).T

    wide_optimal = wide_optimal\
        .drop(columns='Estimator')\
        .pivot(
            ['Dataset', 'Classifier', 'variable'],
            'Generator',
            ['mean', 'std']
        )

    return (
        wide_optimal['mean'].drop(columns='SMOTE'),
        wide_optimal['std'].drop(columns='SMOTE')
    )


def calculate_wide_optimal_al(results):
    core_metrics = results\
        .reset_index()['Evaluation Metric']\
        .dropna().unique()

    res_ = []
    for m in ['mean', 'std']:
        res = results.loc[
            :, results.columns.get_level_values(1) == m
        ]
        res.columns = res.columns.get_level_values(0)
        res = res.reset_index()\
            .drop(columns=[*core_metrics, 'Selection Criterion'])\
            .melt(id_vars=['Dataset', 'Estimator', 'Evaluation Metric'])\
            .set_index(
                ['Dataset', 'Estimator', 'Evaluation Metric', 'variable']
            ).rename(columns={'value': m})
        res_.append(res)

    wide_optimal = pd.concat(res_, axis=1).reset_index()\
        .groupby(
            ['Dataset', 'Estimator', 'variable', 'Evaluation Metric']
        )\
        .apply(
            lambda dat: (
                dat.iloc[np.argmax(dat['mean'])]
                if not dat.variable.iloc[0].startswith('dur')
                else dat.iloc[np.argmin(dat['mean'])]
            )
        ).reset_index(drop=True)

    (
        wide_optimal['AL'],
        wide_optimal['Generator'],
        wide_optimal['Classifier']
    ) = np.array(
        wide_optimal.Estimator.apply(
            lambda x: x.split('|')
            if len(x.split('|')) == 3
            else [np.nan, np.nan, x.split('|')[1]]
        ).tolist()
    ).T

    wide_optimal = wide_optimal\
        .drop(columns='Estimator')\
        .pivot(
            ['Dataset', 'Classifier', 'Evaluation Metric', 'variable'],
            ['AL', 'Generator'],
            ['mean', 'std']
        )

    return (
        wide_optimal['mean'].drop(columns='SMOTE'),
        wide_optimal['std'].drop(columns='SMOTE')
    )


def generate_main_results(results):
    """Generate the main results of the experiment."""

    wide_optimal_al = calculate_wide_optimal_al(results)
    wide_optimal = calculate_wide_optimal(results)

    # Wide optimal AULC
    wide_optimal_aulc = generate_mean_std_tbl_bold(
        *(
            df.loc[
                df.index.get_level_values(3) == 'area_under_learning_curve'
            ].droplevel('variable', axis=0)
            for df in wide_optimal_al
        ),
        decimals=3
    )
    wide_optimal_aulc.index.rename(
        ['Dataset', 'Classifier', 'Metric'], inplace=True
    )

    # Mean ranking analysis
    mean_std_aulc_ranks = generate_mean_std_tbl_bold(
        *mean_std_ranks_al(wide_optimal_al[0], 'area_under_learning_curve'),
        maximum=False,
        decimals=2
    )

    # Mean scores analysis
    optimal_mean_std_scores = generate_mean_std_tbl_bold(
        *calculate_mean_std_table(wide_optimal),
        maximum=True,
        decimals=3
    )

    mean_std_aulc_scores = generate_mean_std_tbl_bold(
        *calculate_mean_std_table_al(
            wide_optimal_al, 'area_under_learning_curve'
        ),
        maximum=True,
        decimals=3
    )

    # Deficiency scores analysis
    mean_std_deficiency = generate_mean_std_tbl_bold(
        *deficiency_scores(wide_optimal, wide_optimal_al),
        maximum=False,
        decimals=3,
        threshold=.5
    )

    # Return results and names
    main_results_names = (
        'wide_optimal_aulc',
        'mean_std_aulc_ranks',
        'mean_std_aulc_scores',
        'optimal_mean_std_scores',
        'mean_std_deficiency'
    )

    return zip(
        main_results_names,
        (
            wide_optimal_aulc,
            mean_std_aulc_ranks,
            mean_std_aulc_scores,
            optimal_mean_std_scores,
            mean_std_deficiency
        )
    )


if __name__ == '__main__':

    # extract and load datasets
    ZipFile(join(DATA_PATH, 'active_learning_augmentation.db.zip'), 'r')\
        .extract('active_learning_augmentation.db', path=DATA_PATH)

    datasets = load_datasets(data_dir=DATA_PATH)

    # remove uncompressed database file
    os.remove(join(DATA_PATH, 'active_learning_augmentation.db'))

    # datasets_description
    summarize_multiclass_datasets(datasets).to_csv(
        join(ANALYSIS_PATH, 'datasets_description.csv'), index=False
    )

    # load results
    res_names = [
        r for r in os.listdir(RESULTS_PATH)
        if r.endswith('.pkl')
    ]
    results = []
    for name in res_names:
        file_path = join(RESULTS_PATH, name)
        df_results = pd.read_pickle(file_path)
        df_results['Dataset'] = name\
            .replace('_base.pkl', '')\
            .replace('_ceiling.pkl', '')\
            .replace('_proposed.pkl', '')
        results.append(df_results)

    # Combine and select results
    results = select_results(pd.concat(results))

    # Main results - dataframes
    main_results = generate_main_results(results)
