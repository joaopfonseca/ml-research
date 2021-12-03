"""
Analyze the experimental results.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join
from zipfile import ZipFile
from itertools import product
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from rlearn.tools.reporting import _extract_pvalue
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from rlearn.tools import summarize_datasets
from research.utils import (
    generate_paths,
    load_datasets,
    load_plt_sns_configs,
    make_bold,
)

DATA_PATH, RESULTS_PATH, ANALYSIS_PATH = generate_paths(__file__)

DATASETS_NAMES = [
    "baseball",
    "gas_drift",
    "image_segmentation",
    "japanese_vowels",
    "mfeat_zernike",
    "mice_protein",
    "pendigits",
    "texture",
    "vehicle",
    "waveform",
]

DATASETS_MAPPING = dict(
    [
        (d, "".join([i[0] for i in d.split("_")]).upper())
        if (len(d.split("_")) > 1)
        else (d, d.title())
        for d in DATASETS_NAMES
    ]
)

METRICS_MAPPING = dict(
    [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F-score"),
        ("geometric_mean_score_macro", "G-mean"),
    ]
)

GENERATOR_NAMES = ["NONE", "G-SMOTE", "G-SMOTE-AUGM"]

GROUP_KEYS = ["Dataset", "Estimator", "Evaluation Metric", "Selection Criterion"]


def generate_mean_std_tbl_bold(
    mean_vals, std_vals, maximum=True, decimals=2, threshold=None
):
    """
    Generate table that combines mean and sem values. This is the same
    function of the ml-research library but with the bug of the decimals on
    the standard deviations fixed (which will be already fixed in version
    0.3.4).
    """
    mean_bold = mean_vals.apply(
        lambda row: make_bold(row, maximum, decimals, threshold, with_sem=True)[0],
        axis=1,
    )
    mask = mean_vals.apply(
        lambda row: make_bold(row, maximum, decimals, threshold, with_sem=True)[1],
        axis=1,
    ).values

    formatter = "{0:.%sf}" % decimals
    std_bold = std_vals.applymap(lambda x: formatter.format(x))
    std_bold = np.where(mask, std_bold + "}", std_bold)
    scores = mean_bold + r" $\pm$ " + std_bold
    return scores


def _make_bold_stat_signif(value, sig_level=0.05):
    """Make bold the lowest or highest value(s)."""

    val = "{%.1e}" % value
    val = "\\textbf{%s}" % val if value <= sig_level else val
    return val


def generate_pvalues_tbl_bold(tbl, sig_level=0.05):
    """Format p-values."""
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(
            lambda pvalue: _make_bold_stat_signif(pvalue, sig_level)
        )
    return tbl


def summarize_multiclass_datasets(datasets):
    summarized = (
        summarize_datasets(datasets)
        .rename(
            columns={
                "Dataset name": "Dataset",
                "Imbalance Ratio": "IR",
            }
        )
        .set_index("Dataset")
        .join(
            pd.Series(
                dict([(name, dat[-1].unique().size) for name, dat in datasets]),
                name="Classes",
            )
        )
        .reset_index()
    )
    return summarized


def select_results(results):
    """
    Computes mean and std across all splits and runs from the original
    experiment's data.
    """

    results = results.copy()

    # Extract info from the params dict
    for param in ["evaluation_metric", "selection_strategy"]:
        results[param] = results.params.apply(
            lambda x: (x[param] if param in x.keys() else np.nan)
        )

    # Format column names
    results.rename(
        columns={
            "param_est_name": "Estimator",
            "evaluation_metric": "Evaluation Metric",
            "selection_strategy": "Selection Criterion",
        },
        inplace=True,
    )

    # Drop random states from params
    # Convert to params to string in order to use groupby
    results.params = results.params.apply(
        lambda x: {
            k: v
            for k, v in x.items()
            if ("random_state" not in k)
            and ("evaluation_metric" not in k)
            and ("selection_strategy" not in k)
        }
    ).astype(str)

    scoring_cols = {
        col: "_".join(col.split("_")[2:])
        for col in results.columns
        if "mean_test" in col
    }

    # Group data using GROUP_KEYS
    scoring_mapping = {
        scorer_name: [np.mean, np.std] for scorer_name in scoring_cols.values()
    }

    results_ = results.rename(columns=scoring_cols).groupby(GROUP_KEYS, dropna=False)

    # Get standard deviations
    stds = results_.apply(
        lambda dat: [
            np.std(
                dat[
                    dat.columns[
                        dat.columns.str.contains(scorer)
                        & dat.columns.str.contains("split")
                    ]
                ].values.flatten()
            )
            for scorer in scoring_mapping.keys()
        ]
    )

    results = results_.agg(scoring_mapping)

    mask_cols = np.array(list(results.columns))[:, 1] == "std"
    values_arr = results.values
    values_arr[:, mask_cols] = np.array(stds.tolist())

    return pd.DataFrame(values_arr, columns=results.columns, index=results.index)


def calculate_wide_optimal(results):
    core_metrics = results.reset_index()["Evaluation Metric"].dropna().unique()

    res_ = []
    for m in ["mean", "std"]:
        res = results.loc[:, results.columns.get_level_values(1) == m]
        res.columns = res.columns.get_level_values(0)
        res = (
            res.reset_index()
            .drop(columns=["Evaluation Metric", "Selection Criterion"])
            .loc[:, ["Dataset", "Estimator", *core_metrics]]
            .melt(id_vars=["Dataset", "Estimator"])
            .rename(columns={"value": m})
            .set_index(["Dataset", "Estimator", "variable"])
        )
        res_.append(res)

    wide_optimal = (
        pd.concat(res_, axis=1)
        .reset_index()
        .groupby(["Dataset", "Estimator", "variable"])
        .apply(lambda dat: dat.iloc[np.argmax(dat["mean"])])
        .reset_index(drop=True)
    )

    (
        wide_optimal["AL"],
        wide_optimal["Generator"],
        wide_optimal["Classifier"],
    ) = np.array(
        wide_optimal.Estimator.apply(
            lambda x: x.split("|")
            if len(x.split("|")) == 3
            else [np.nan, np.nan, x.split("|")[1]]
        ).tolist()
    ).T

    wide_optimal = wide_optimal.drop(columns="Estimator").pivot(
        ["Dataset", "Classifier", "variable"], ["AL", "Generator"], ["mean", "std"]
    )

    return [
        df.drop(("AL-GEN", "G-SMOTE"), axis=1).droplevel(0, axis=1)
        for df in [wide_optimal["mean"], wide_optimal["std"]]
    ]


def calculate_wide_optimal_al(results):
    core_metrics = results.reset_index()["Evaluation Metric"].dropna().unique()

    res_ = []
    for m in ["mean", "std"]:
        res = results.loc[:, results.columns.get_level_values(1) == m]
        res.columns = res.columns.get_level_values(0)
        res = (
            res.reset_index()
            .drop(columns=[*core_metrics, "Selection Criterion"])
            .melt(id_vars=["Dataset", "Estimator", "Evaluation Metric"])
            .set_index(["Dataset", "Estimator", "Evaluation Metric", "variable"])
            .rename(columns={"value": m})
        )
        res_.append(res)

    wide_optimal = (
        pd.concat(res_, axis=1)
        .reset_index()
        .groupby(["Dataset", "Estimator", "variable", "Evaluation Metric"])
        .apply(
            lambda dat: (
                dat.iloc[np.argmax(dat["mean"])]
                if not dat.variable.iloc[0].startswith("dur")
                else dat.iloc[np.argmin(dat["mean"])]
            )
        )
        .reset_index(drop=True)
    )

    (
        wide_optimal["AL"],
        wide_optimal["Generator"],
        wide_optimal["Classifier"],
    ) = np.array(
        wide_optimal.Estimator.apply(
            lambda x: x.split("|")
            if len(x.split("|")) == 3
            else [np.nan, np.nan, x.split("|")[1]]
        ).tolist()
    ).T

    wide_optimal = wide_optimal.drop(columns="Estimator").pivot(
        ["Dataset", "Classifier", "Evaluation Metric", "variable"],
        ["AL", "Generator"],
        ["mean", "std"],
    )

    return [
        df.drop(("AL-GEN", "G-SMOTE"), axis=1).droplevel(0, axis=1)
        for df in [wide_optimal["mean"], wide_optimal["std"]]
    ]


def calculate_mean_std_table(wide_optimal):
    df = wide_optimal[0].copy()
    df_grouped = (
        df.reset_index()
        .rename(columns={"variable": "Evaluation Metric"})
        .groupby(["Classifier", "Evaluation Metric"])
    )

    return df_grouped.mean(), df_grouped.std(ddof=0)


def mean_std_ranks_al(wide_optimal, al_metric="area_under_learning_curve"):
    asc = False if not al_metric.startswith("dur") else True

    ranks = (
        wide_optimal.loc[wide_optimal.index.get_level_values(3) == al_metric]
        .rank(axis=1, ascending=asc)
        .reset_index()
        .groupby(["Classifier", "Evaluation Metric"])
    )

    return ranks.mean(), ranks.std(ddof=0)


def calculate_mean_std_table_al(wide_optimal_al, al_metric="area_under_learning_curve"):
    df = wide_optimal_al[0].copy()
    df_grouped = (
        df.loc[df.index.get_level_values(3) == al_metric]
        .reset_index()
        .groupby(["Classifier", "Evaluation Metric"])
    )

    return df_grouped.mean(), df_grouped.std(ddof=0)


def generate_main_results(results):
    """Generate the main results of the experiment."""

    wide_optimal_al = calculate_wide_optimal_al(results)
    wide_optimal = calculate_wide_optimal(results)

    # Wide optimal AULC
    wide_optimal_aulc = generate_mean_std_tbl_bold(
        *(
            df.loc[
                df.index.get_level_values(3) == "area_under_learning_curve"
            ].droplevel("variable", axis=0)
            for df in wide_optimal_al
        ),
        decimals=3,
    )
    wide_optimal_aulc.index.rename(["Dataset", "Classifier", "Metric"], inplace=True)

    # Mean ranking analysis
    mean_std_aulc_ranks = generate_mean_std_tbl_bold(
        *mean_std_ranks_al(wide_optimal_al[0], "area_under_learning_curve"),
        maximum=False,
        decimals=2,
    )

    # Mean scores analysis
    optimal_mean_std_scores = generate_mean_std_tbl_bold(
        *calculate_mean_std_table(wide_optimal), maximum=True, decimals=3
    )

    mean_std_aulc_scores = generate_mean_std_tbl_bold(
        *calculate_mean_std_table_al(wide_optimal_al, "area_under_learning_curve"),
        maximum=True,
        decimals=3,
    )

    # Return results and names
    main_results_names = (
        "wide_optimal_aulc",
        "mean_std_aulc_ranks",
        "mean_std_aulc_scores",
        "optimal_mean_std_scores",
    )

    return zip(
        main_results_names,
        (
            wide_optimal_aulc,
            mean_std_aulc_ranks,
            mean_std_aulc_scores,
            optimal_mean_std_scores,
        ),
    )


def data_utilization_rate(*wide_optimal):

    df = wide_optimal[0]
    df = df.div(df["NONE"], axis=0)

    dur_grouped = (
        df.loc[df.index.get_level_values(3).str.startswith("dur")]
        .reset_index()
        .melt(id_vars=df.index.names)
        .pivot(
            ["Dataset", "Classifier", "Evaluation Metric", "Generator"],
            "variable",
            "value",
        )
        .reset_index()
        .groupby(["Classifier", "Evaluation Metric", "Generator"])
    )

    return dur_grouped.mean(), dur_grouped.std(ddof=0)


def generate_data_utilization_tables(wide_optimal_al):

    data_utilization = wide_optimal_al[0].reset_index()

    # Data utilization per dataset and performance threshold
    optimal_du = data_utilization[
        (data_utilization["Evaluation Metric"] == "geometric_mean_score_macro")
        & (data_utilization.variable.str.startswith("dur_"))
    ].drop(columns="Evaluation Metric")

    optimal_du = (
        optimal_du.groupby(["Classifier", "variable"])
        .mean()
        .apply(lambda row: make_bold(row * 100, maximum=False, num_decimals=1), axis=1)
        .reset_index()
    )

    optimal_du["G-mean Score"] = optimal_du.variable.str.replace("dur_", "")
    optimal_du["G-mean Score"] = (optimal_du["G-mean Score"].astype(int) / 100).apply(
        lambda x: "{0:.2f}".format(x)
    )

    for generator in GENERATOR_NAMES:
        optimal_du[generator] = optimal_du[generator].apply(
            lambda x: x[:-1] + "\\%}" if x.endswith("}") else x + "\\%"
        )

    return optimal_du[
        ["G-mean Score", "Classifier", "NONE", "G-SMOTE", "G-SMOTE-AUGM"]
    ].sort_values(["G-mean Score", "Classifier"])


def generate_dur_visualization(wide_optimal_al):
    """Visualize data utilization rates"""
    dur = data_utilization_rate(*wide_optimal_al)
    dur_mean, dur_std = (
        df.loc[
            df.index.get_level_values("Evaluation Metric").isin(
                ["geometric_mean_score_macro", "f1_macro"]
            )
        ]
        .rename(columns={col: int(col.replace("dur_", "")) for col in df.columns})
        .rename(
            index={
                "NONE": "Standard",
                "G-SMOTE": "Oversampling",
                "G-SMOTE-AUGM": "Proposed",
            }
        )
        for df in dur
    )

    load_plt_sns_configs(10)

    col_values = dur_mean.index.get_level_values("Evaluation Metric").unique()
    row_values = dur_mean.index.get_level_values("Classifier").unique()

    # Set and format main content of the visualization
    fig, axes = plt.subplots(
        row_values.shape[0],
        col_values.shape[0],
        figsize=(7, 7),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for (row, clf), (col, metric) in product(
        enumerate(row_values), enumerate(col_values)
    ):
        ax = axes[row, col]
        dur_mean.loc[(clf, metric)].T.plot.line(
            ax=ax,
            xlabel="",
            color={
                "Standard": "indianred",
                "Oversampling": "burlywood",
                "Proposed": "steelblue",
            },
        )

        ax.set_ylabel(clf)
        ax.set_ylim(
            bottom=(dur_mean.loc[clf].values.min() - 0.05),
            top=(
                dur_mean.loc[clf].values.max()
                if dur_mean.loc[clf].values.max() >= 1.05
                else 1.05
            ),
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticks(dur_mean.columns, minor=True)

        # Set legend
        if (row == 1) and (col == 1):
            handles, labels = ax.get_legend_handles_labels()
            order = [2, 0, 1]
            ax.legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize=10,
            )
        else:
            ax.get_legend().remove()

    fig.text(0.45, -0.025, "Performance Thresholds", ha="center", va="bottom")

    for ax, metric in zip(axes[0, :], col_values):
        ax.set_title(METRICS_MAPPING[metric])

    fig.savefig(
        join(ANALYSIS_PATH, "data_utilization_rate.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def apply_wilcoxon_test(wide_optimal, dep_var, OVRS_NAMES, alpha):
    """Performs a Wilcoxon signed-rank test"""
    pvalues = []
    for ovr in OVRS_NAMES:
        mask = np.repeat(True, len(wide_optimal))

        pvalues.append(
            wilcoxon(
                wide_optimal.loc[mask, ovr], wide_optimal.loc[mask, dep_var]
            ).pvalue
        )
    wilcoxon_results = pd.DataFrame(
        {
            "Oversampler": OVRS_NAMES,
            "p-value": pvalues,
            "Significance": np.array(pvalues) < alpha,
        }
    )
    return wilcoxon_results


def generate_statistical_results(wide_optimal_al, alpha=0.1, control_method="NONE"):
    """Generate the statistical results of the experiment."""

    # Get results
    results = (
        wide_optimal_al[0]
        .reset_index()[wide_optimal_al[0].reset_index().variable.str.startswith("dur_")]
        .drop(columns=["variable"])
        .rename(columns={"Evaluation Metric": "Metric"})
    )

    # Calculate rankings
    ranks = (
        results.set_index(["Dataset", "Classifier", "Metric"])
        .rank(axis=1, ascending=0)
        .reset_index()
    )

    # Friedman test
    friedman_test = (
        ranks.groupby(["Classifier", "Metric"])
        .apply(_extract_pvalue)
        .reset_index()
        .rename(columns={0: "p-value"})
    )

    friedman_test["Significance"] = friedman_test["p-value"] < alpha
    friedman_test["p-value"] = friedman_test["p-value"].apply(
        lambda x: "{:.1e}".format(x)
    )

    # Wilcoxon signed rank test
    # Optimal proposed framework vs oversampling framework
    wilcoxon_test = []
    for dataset in results.Dataset.unique():
        wilcoxon_results = apply_wilcoxon_test(
            results[results["Dataset"] == dataset],
            "G-SMOTE-AUGM",
            ["NONE", "G-SMOTE"],
            alpha,
        ).drop(columns="Significance")
        wilcoxon_results["Dataset"] = dataset.replace("_", " ").title()
        wilcoxon_test.append(
            wilcoxon_results.pivot("Dataset", "Oversampler", "p-value")
        )

    wilcoxon_test = pd.concat(wilcoxon_test, axis=0)
    wilcoxon_test = generate_pvalues_tbl_bold(
        wilcoxon_test.reset_index(), sig_level=alpha
    )

    # Holms test
    # Optimal proposed framework vs baseline framework
    ovrs_names = list(results.columns[3:])
    ovrs_names.remove(control_method)

    # Define empty p-values table
    pvalues = pd.DataFrame()

    # Populate p-values table
    for name in ovrs_names:
        pvalues_pair = results.groupby(["Classifier", "Metric"])[
            [name, control_method]
        ].apply(lambda df: ttest_rel(df[name], df[control_method])[1])
        pvalues_pair = pd.DataFrame(pvalues_pair, columns=[name])
        pvalues = pd.concat([pvalues, pvalues_pair], axis=1)

    # Corrected p-values
    holms_test = pd.DataFrame(
        pvalues.apply(
            lambda col: multipletests(col, method="holm")[1], axis=1
        ).values.tolist(),
        columns=ovrs_names,
    )
    holms_test = generate_pvalues_tbl_bold(
        holms_test.set_index(pvalues.index).reset_index(), sig_level=alpha
    )

    # Return statistical analyses
    statistical_results_names = ("friedman_test", "wilcoxon_test", "holms_test")
    statistical_results = zip(
        statistical_results_names, (friedman_test, wilcoxon_test, holms_test)
    )
    return statistical_results


if __name__ == "__main__":

    # extract and load datasets
    ZipFile(join(DATA_PATH, "active_learning_augmentation.db.zip"), "r").extract(
        "active_learning_augmentation.db", path=DATA_PATH
    )

    datasets = load_datasets(data_dir=DATA_PATH)

    # remove uncompressed database file
    os.remove(join(DATA_PATH, "active_learning_augmentation.db"))

    # datasets_description
    summarized_datasets = summarize_multiclass_datasets(datasets)
    summarized_datasets["Dataset"] = summarized_datasets["Dataset"].str.title()
    mask = (
        summarized_datasets["Dataset"]
        .str.lower()
        .str.replace(" ", "_")
        .isin(DATASETS_NAMES)
    )
    summarized_datasets[mask].to_csv(
        join(ANALYSIS_PATH, "datasets_description.csv"), index=False
    )

    # load results
    res_names = [
        r
        for r in os.listdir(RESULTS_PATH)
        if r.endswith(".pkl") and any(x in r for x in DATASETS_NAMES)
    ]
    results = []
    for name in res_names:
        file_path = join(RESULTS_PATH, name)
        df_results = pd.read_pickle(file_path)
        df_results["Dataset"] = (
            name.replace("_al_base.pkl", "")
            .replace("_ceiling.pkl", "")
            .replace("_al_proposed.pkl", "")
        )
        results.append(df_results)

    # Combine and select results
    results = select_results(pd.concat(results).copy())

    # Main results - dataframes
    main_results = generate_main_results(results)

    for name, result in main_results:
        # Format results
        result = result.rename(index={**METRICS_MAPPING, **DATASETS_MAPPING}).rename(
            columns={"nan": "MP"}
        )
        result = result[
            [col for col in ["MP"] + GENERATOR_NAMES if col in result.columns]
        ]

        result.reset_index(inplace=True)
        # Keep only G-mean and F-score
        # if ('Evaluation Metric' in result.columns or
        #         'Metric' in result.columns):

        #     query_col = 'Evaluation Metric'\
        #         if 'Evaluation Metric' in result.columns\
        #         else 'Metric'

        #     result = result[
        #         result[query_col].isin(['G-mean', 'F-score'])
        #     ]

        # Export LaTeX-ready dataframe
        result.rename(
            columns={
                "NONE": "Standard",
                "G-SMOTE": "Oversampling",
                "G-SMOTE-AUGM": "Proposed",
            }
        ).to_csv(join(ANALYSIS_PATH, f"{name}.csv"), index=False)

    # Main results - visualizations
    wide_optimal_al = calculate_wide_optimal_al(results)
    generate_dur_visualization(wide_optimal_al)

    # Data utilization - dataframes
    optimal_data_utilization = generate_data_utilization_tables(wide_optimal_al)
    optimal_data_utilization[
        (optimal_data_utilization["G-mean Score"].astype(float) >= 0.6)
        & (optimal_data_utilization["G-mean Score"].apply(lambda x: x[-1] in "06"))
    ].rename(
        columns={
            "NONE": "Standard",
            "G-SMOTE": "Oversampling",
            "G-SMOTE-AUGM": "Proposed",
        }
    ).to_csv(
        join(ANALYSIS_PATH, "optimal_data_utilization.csv"), index=False
    )

    # Statistical results
    statistical_results = generate_statistical_results(
        wide_optimal_al, alpha=0.05, control_method="NONE"
    )
    for name, result in statistical_results:
        if "Metric" in result.columns:
            result["Metric"] = result["Metric"].map(METRICS_MAPPING)
        result = result.rename(
            columns={
                "Metric": "Evaluation Metric",
                "NONE": "Standard",
                "G-SMOTE": "Oversampling",
                "G-SMOTE-AUGM": "Proposed",
            }
        )
        result.to_csv(join(ANALYSIS_PATH, f"{name}.csv"), index=False)
