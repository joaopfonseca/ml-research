"""
Analyze the experimental results.
"""

# Author: João Fonseca <jpmrfonseca@gmail.com>
#         Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import listdir
from os.path import join
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from rlearn.tools import (
    summarize_datasets
)
from research.datasets import RemoteSensingDatasets
from research.utils import (
    generate_paths,
    generate_mean_std_tbl_bold,
    load_datasets,
    load_plt_sns_configs
)
from rlearn.tools.reporting import _extract_pvalue
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

DATASETS_NAMES = [
    d.replace('fetch_', '')
    for d in dir(RemoteSensingDatasets())
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

GENERATOR_NAMES = [
    'NONE',
    'SMOTE',
    'G-SMOTE'
]


def _make_bold_stat_signif(value, sig_level=.05):
    """Make bold the lowest or highest value(s)."""

    val = '%.1e' % value
    val = (
        '\\textbf{%s}' % val
        if value <= sig_level
        else val
    )
    return val


def generate_pvalues_tbl_bold(tbl, sig_level=.05):
    """Format p-values."""
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(
            lambda pvalue: _make_bold_stat_signif(pvalue, sig_level)
        )
    return tbl


def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={
            'Dataset name': 'Dataset',
            'Imbalance Ratio': 'IR',
            'Minority instances': 'Min. Instances',
            'Majority instances': 'Maj. Instances'
        })\
        .set_index('Dataset')\
        .join(pd.Series(dict(
            [(name, dat[-1].unique().size) for name, dat in datasets]
        ), name='Classes'))\
        .reset_index()
    summarized.loc[:, 'Dataset'] = summarized.loc[:, 'Dataset']\
        .apply(lambda x: x.title())
    return summarized


def plot_lulc_images():
    arrays_x = []
    arrays_y = []
    for dat_name in DATASETS_NAMES:
        X, y = RemoteSensingDatasets()._load_gic_dataset(dat_name)
        arrays_x.append(X[:, :, 100])
        arrays_y.append(np.squeeze(y))

    for X, y, figname in zip(arrays_x, arrays_y, DATASETS_NAMES):
        plt.figure(
            figsize=(20, 10),
            dpi=320
        )
        if figname == 'kennedy_space_center':
            X = np.clip(X, 0, 350)
        for i, (a, cmap) in enumerate(zip([X, y], ['gist_gray', 'terrain'])):
            plt.subplot(2, 1, i+1)
            plt.imshow(
                a, cmap=plt.get_cmap(cmap)
            )
            plt.axis('off')
        plt.savefig(
            join(analysis_path, figname),
            bbox_inches='tight',
            pad_inches=0
        )


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


def get_mean_std_data(results):
    mask = results.columns.get_level_values(1).isin(['mean', ''])
    df_mean = results.iloc[
        :, mask
    ].copy()
    df_mean.columns = df_mean.columns.get_level_values(0)
    df_std = results.iloc[
        :, ~mask
    ].copy()
    df_std.columns = df_std.columns.get_level_values(0)
    return df_mean, df_std


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

    return wide_optimal['mean'], wide_optimal['std']


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
            ['Dataset', 'Classifier', 'Evaluation Metric', 'variable'],
            'Generator',
            ['mean', 'std']
        )

    return wide_optimal['mean'], wide_optimal['std']


def calculate_mean_std_table(wide_optimal):
    df = wide_optimal[0].copy()
    df_grouped = df.reset_index()\
        .rename(columns={'variable': 'Evaluation Metric'})\
        .groupby(['Classifier', 'Evaluation Metric'])

    return df_grouped.mean(), df_grouped.std(ddof=0)


def calculate_mean_std_table_al(
    wide_optimal_al, al_metric='area_under_learning_curve'
):
    df = wide_optimal_al[0].copy()
    df_grouped = df.loc[df.index.get_level_values(3) == al_metric]\
        .reset_index().groupby(['Classifier', 'Evaluation Metric'])

    return df_grouped.mean(), df_grouped.std(ddof=0)


def mean_std_ranks(
    wide_optimal
):
    ranks = wide_optimal.rank(axis=1, ascending=False)\
        .reset_index()\
        .groupby(['Classifier', 'variable'])
    return ranks.mean(), ranks.std(ddof=0)


def mean_std_ranks_al(
    wide_optimal, al_metric='area_under_learning_curve'
):
    asc = False if not al_metric.startswith('dur') else True

    ranks = wide_optimal.loc[
            wide_optimal.index.get_level_values(3) == al_metric
        ].rank(axis=1, ascending=asc)\
        .reset_index()\
        .groupby(['Classifier', 'Evaluation Metric'])

    return ranks.mean(), ranks.std(ddof=0)


def data_utilization_rate(*wide_optimal):

    df = wide_optimal[0]
    df = df.div(df['NONE'], axis=0)

    dur_grouped = df.loc[
            df.index.get_level_values(3).str.startswith('dur')
        ].reset_index()\
        .melt(id_vars=df.index.names)\
        .pivot(
            ['Dataset', 'Classifier', 'Evaluation Metric', 'Generator'],
            'variable',
            'value'
        ).reset_index()\
        .groupby(['Classifier', 'Evaluation Metric', 'Generator'])

    return dur_grouped.mean(), dur_grouped.std(ddof=0)


def deficiency_scores(wide_optimal, wide_optimal_al):
    wo_mp = wide_optimal[0]['nan'].to_frame()

    wo_al = wide_optimal_al[0].loc[
        wide_optimal_al[0]
        .index
        .get_level_values('variable') == 'area_under_learning_curve'
    ].droplevel('variable', axis=0)

    wo_a = wo_al.drop(columns='NONE')
    wo_b = wo_al['NONE'].to_frame()

    deficiency = (wo_mp.values - wo_a.values)\
        / (2*wo_mp.values - wo_a.values - wo_b.values)

    deficiency = pd.DataFrame(
            deficiency,
            columns=wo_a.columns,
            index=wo_a.index
        ).reset_index()\
        .groupby(['Classifier', 'Evaluation Metric'])

    return deficiency.mean(), deficiency.std(ddof=0)


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


def generate_dur_visualization(wide_optimal_al):
    """Visualize data utilization rates"""
    dur = data_utilization_rate(*wide_optimal_al)
    dur_mean, dur_std = (
        df.rename(
            columns={
                col: int(col.replace('dur_', ''))
                for col in df.columns
            }
        ) for df in dur
    )

    load_plt_sns_configs()

    col_values = dur_mean.index.get_level_values('Evaluation Metric').unique()
    row_values = dur_mean.index.get_level_values('Classifier').unique()

    # Set and format main content of the visualization
    fig, axes = plt.subplots(
        row_values.shape[0],
        col_values.shape[0],
        figsize=(10, 6),
        sharex='col',
        sharey='row',
        constrained_layout=True
    )
    for (row, clf), (col, metric) in product(
            enumerate(row_values),
            enumerate(col_values)
    ):
        ax = axes[row, col]
        dur_mean.loc[(clf, metric)].T.plot.line(
            ax=ax,
            xlabel='',
            color={
                'SMOTE': 'steelblue',
                'G-SMOTE': 'burlywood',
                'NONE': 'indianred'
            }
        )

        err_fills = dur_std.loc[(clf, metric)].T
        for col_ in err_fills.columns:
            ax.fill_between(
                err_fills.index,
                (dur_mean.loc[(clf, metric, col_)].T - err_fills[col_]),
                (dur_mean.loc[(clf, metric, col_)].T + err_fills[col_]),
                alpha=0.1,
                color={
                    'SMOTE': 'steelblue',
                    'G-SMOTE': 'burlywood',
                    'NONE': 'indianred'
                }[col_]
            )

        ax.set_ylabel(clf)
        ax.set_ylim(
            bottom=dur_mean.loc[clf].values.min(),
            top=dur_mean.loc[clf].values.max()
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xticks(dur_mean.columns)

        # Set x label
        if (row == 2) and (col == 1):
            ax.set_xlabel('Performance Thresholds')

        # Set legend
        if (row == 1) and (col == 2):
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1, .5),
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize=10
            )
        else:
            ax.get_legend().remove()

    for ax, metric in zip(axes[0, :], col_values):
        ax.set_title(METRICS_MAPPING[metric])

    fig.savefig(join(analysis_path, 'data_utilization_rate.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close()


def generate_mean_rank_bar_chart(wide_optimal_al):
    """Generates bar chart."""

    load_plt_sns_configs()

    ranks, ranks_std = (
        df.reset_index()
        for df in mean_std_ranks_al(
            wide_optimal_al[0],
            'area_under_learning_curve'
        )
    )

    ranks['Evaluation Metric'] = ranks['Evaluation Metric'].apply(
        lambda x: METRICS_MAPPING[x]
    )

    fig, axes = plt.subplots(
        ranks['Classifier'].unique().shape[0],
        ranks['Evaluation Metric'].unique().shape[0],
        figsize=(5, 6)
    )
    lranks = ranks.set_index(['Classifier', 'Evaluation Metric'])
    for (row, clf), (col, metric) in product(
            enumerate(ranks['Classifier'].unique()),
            enumerate(ranks['Evaluation Metric'].unique())
    ):
        dat = len(GENERATOR_NAMES) - lranks.loc[
            (clf, metric)
        ].loc[list(GENERATOR_NAMES[::-1])]
        axes[row, col].bar(
            dat.index,
            dat.values,
            color=[
                'steelblue'
                for i in range(len(GENERATOR_NAMES)-1)
            ]+['indianred']
        )
        plt.sca(axes[row, col])
        plt.yticks(
            range(len(GENERATOR_NAMES)),
            [None]+list(range(1, len(GENERATOR_NAMES)))[::-1]
        )
        plt.xticks(rotation=90)
        if row == 0:
            plt.title(metric)
        if col == 0:
            plt.ylabel(f'{clf}')
        if row != len(ranks.Classifier.unique())-1:
            plt.xticks(range(len(GENERATOR_NAMES)), [])
        if col != 0:
            plt.yticks(range(len(GENERATOR_NAMES)), [])
        sns.despine(left=True)
        plt.grid(b=None, axis='x')

    fig.savefig(join(analysis_path, 'mean_rankings_bar_chart.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close()


def apply_wilcoxon_test(wide_optimal, dep_var, OVRS_NAMES, alpha):
    """Performs a Wilcoxon signed-rank test"""
    pvalues = []
    for ovr in OVRS_NAMES:
        mask = np.repeat(True, len(wide_optimal))

        pvalues.append(wilcoxon(
            wide_optimal.loc[mask, ovr],
            wide_optimal.loc[mask, dep_var]).pvalue
        )
    wilcoxon_results = pd.DataFrame({
        'Oversampler': OVRS_NAMES,
        'p-value': pvalues,
        'Significance': np.array(pvalues) < alpha
    })
    return wilcoxon_results


def generate_statistical_results(
    wide_optimal_al, alpha=.1, control_method='NONE'
):
    """Generate the statistical results of the experiment."""

    # Get results
    results = wide_optimal_al[0][GENERATOR_NAMES]\
        .reset_index()[
            wide_optimal_al[0].reset_index().variable
            ==
            'area_under_learning_curve'
        ].drop(columns=['variable'])\
        .rename(columns={'Evaluation Metric': 'Metric'})

    # Calculate rankings
    ranks = results\
        .set_index(['Dataset', 'Classifier', 'Metric'])\
        .rank(axis=1, ascending=0).reset_index()

    # Friedman test
    friedman_test = ranks.groupby(['Classifier', 'Metric'])\
        .apply(_extract_pvalue)\
        .reset_index()\
        .rename(columns={0: 'p-value'})

    friedman_test['Significance'] = friedman_test['p-value'] < alpha
    friedman_test['p-value'] = friedman_test['p-value'].apply(
        lambda x: '{:.1e}'.format(x)
    )

    # Wilcoxon signed rank test
    # Optimal proposed framework vs baseline framework
    results['Optimal'] = results[['SMOTE', 'G-SMOTE']].max(1)
    wilcoxon_test = []
    for dataset in results.Dataset.unique():
        wilcoxon_results = apply_wilcoxon_test(
            results[results['Dataset'] == dataset],
            'Optimal',
            ['NONE'],
            alpha
        ).drop(columns='Oversampler')
        wilcoxon_results['Dataset'] = dataset.replace('_', ' ').title()
        wilcoxon_test.append(wilcoxon_results[
            ['Dataset', 'p-value', 'Significance']
        ])

    wilcoxon_test = pd.concat(wilcoxon_test, axis=0)
    wilcoxon_test['p-value'] = wilcoxon_test['p-value'].apply(
        lambda x: '{:.1e}'.format(x)
    )

    statistical_results_names = (
        'friedman_test', 'wilcoxon_test'
    )

    statistical_results = zip(
        statistical_results_names, (friedman_test, wilcoxon_test)
    )
    return statistical_results


if __name__ == '__main__':

    data_path, results_path, analysis_path = generate_paths(__file__)

    # load datasets
    datasets = load_datasets(data_dir=data_path)

    # datasets description
    summarize_multiclass_datasets(datasets).to_csv(
        join(analysis_path, 'datasets_description.csv'), index=False
    )

    # datasets visualization
    # plot_lulc_images()

    # load results
    res_names = [
        r for r in listdir(results_path)
        if r.endswith('.pkl')
    ]
    results = []
    for name in res_names:
        file_path = join(results_path, name)
        df_results = pd.read_pickle(file_path)
        df_results['Dataset'] = name\
            .replace('_base.pkl', '')\
            .replace('_al.pkl', '')
        results.append(df_results)

    # Combine and select results
    results = select_results(pd.concat(results))

    # Main results - dataframes
    main_results = generate_main_results(results)
    for name, result in main_results:
        # Format results
        result = result\
            .rename(index={
                **METRICS_MAPPING, **DATASETS_MAPPING
            })\
            .rename(columns={'nan': 'MP'})
        result = (
            result[[
                col
                for col in ['MP']+GENERATOR_NAMES
                if col in result.columns
            ]]
        )

        # Export LaTeX-ready dataframe
        result.to_csv(join(analysis_path, f'{name}.csv'))

    # Main results - visualizations
    wide_optimal_al = calculate_wide_optimal_al(results)
    generate_dur_visualization(wide_optimal_al)
    # generate_mean_rank_bar_chart(wide_optimal_al)

    # Statistical results
    statistical_results = generate_statistical_results(
        wide_optimal_al, alpha=.1, control_method='NONE'
    )
    for name, result in statistical_results:
        if 'Metric' in result.columns:
            result['Metric'] = result['Metric'].map(METRICS_MAPPING)
        result = result.rename(columns={'Metric': 'Evaluation Metric'})
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False)
