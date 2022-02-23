"""
Download, transform and simulate various datasets.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import os
from os.path import join
import pickle
from urllib.parse import urljoin
from string import ascii_lowercase
from sqlite3 import connect

from rich.progress import track
import numpy as np
import pandas as pd

from .base import Datasets, get_data_home, FETCH_URLS


class ContinuousCategoricalDatasets(Datasets):
    """Class to download, transform and save datasets with both continuous
    and categorical features."""

    @staticmethod
    def _modify_columns(data, categorical_features):
        """Rename and reorder columns of dataframe."""
        X_metric, X_cat, y = (
            data.drop(columns=categorical_features + ["target"]),
            data[categorical_features],
            data.target,
        )
        X_metric.columns = range(len(X_metric.columns))
        X_cat.columns = [f"cat_{i}" for i in range(len(X_cat.columns))]
        return pd.concat([X_metric, X_cat, y], axis=1)

    def download(self):
        """Download the datasets."""
        self.data_home_ = get_data_home(data_home=self.data_home)
        dataset_prefix = self.__class__.__name__.lower().replace("datasets", "")

        if self.names == "all":
            func_names = [func_name for func_name in dir(self) if "fetch_" in func_name]
        else:
            func_names = [
                f"fetch_{name}".lower().replace(" ", "_") for name in self.names
            ]
        self.content_ = []
        for func_name in track(func_names, description="Datasets"):
            dat_name = func_name.replace("fetch_", "")
            name = dat_name.upper().replace("_", " ")
            file_name = f"{dataset_prefix}_{dat_name}.csv"

            if (
                file_name not in os.listdir(self.data_home_)
                and self.download_if_missing
            ):
                df, categorical_features = getattr(self, func_name)()
                df = self._modify_columns(df, list(categorical_features))
                df.to_csv(join(self.data_home_, file_name), index=False)

            data = pd.read_csv(join(self.data_home_, file_name))

            self.content_.append((name, data))
        return self

    def fetch_adult(self):
        """Download and transform the Adult Data Set.

        https://archive.ics.uci.edu/ml/datasets/Adult
        """
        data = pd.read_csv(FETCH_URLS["adult"], header=None, na_values=" ?").dropna()
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        return data, categorical_features

    def fetch_abalone(self):
        """Download and transform the Abalone Data Set.

        https://archive.ics.uci.edu/ml/datasets/Abalone
        """
        data = pd.read_csv(FETCH_URLS["abalone"], header=None)
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [0]
        return data, categorical_features

    def fetch_acute(self):
        """Download and transform the Acute Inflammations Data Set.

        https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
        """
        data = pd.read_csv(
            FETCH_URLS["acute"], header=None, sep="\t", decimal=",", encoding="UTF-16"
        )
        data["target"] = data[6].str[0] + data[7].str[0]
        data.drop(columns=[6, 7], inplace=True)
        categorical_features = list(range(1, 6))
        return data, categorical_features

    def fetch_annealing(self):
        """Download and transform the Annealing Data Set.

        https://archive.ics.uci.edu/ml/datasets/Annealing
        """
        data = pd.read_csv(FETCH_URLS["annealing"], header=None, na_values="?")

        # some features are dropped; they have too many missing values
        missing_feats = (data.isnull().sum(0) / data.shape[0]) < 0.1
        data = data.iloc[:, missing_feats.values]
        data[2].fillna(data[2].mode().squeeze(), inplace=True)

        data = data.T.reset_index(drop=True).T
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)

        categorical_features = [0, 1, 5, 9]
        return data, categorical_features

    def fetch_census(self):
        """Download and transform the Census-Income (KDD) Data Set.

        https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
        """
        data = pd.read_csv(FETCH_URLS["census"], header=None)

        categorical_features = (
            list(range(1, 5))
            + list(range(6, 16))
            + list(range(19, 29))
            + list(range(30, 38))
            + [39]
        )

        # some features are dropped; they have too many missing values
        cols_ids = [1, 6, 9, 13, 14, 20, 21, 29, 31, 37]
        categorical_features = np.argwhere(
            np.delete(
                data.rename(columns={k: f"nom_{k}" for k in categorical_features})
                .columns.astype("str")
                .str.startswith("nom_"),
                cols_ids,
            )
        ).squeeze()
        data = data.drop(columns=cols_ids).T.reset_index(drop=True).T
        # some rows are dropped; they have rare missing values
        data = data.iloc[
            data.applymap(lambda x: x != " Not in universe").all(1).values, :
        ]

        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        return data, categorical_features

    def fetch_contraceptive(self):
        """Download and transform the Contraceptive Method Choice Data Set.

        https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
        """
        data = pd.read_csv(FETCH_URLS["contraceptive"], header=None)
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [4, 5, 6, 8]
        return data, categorical_features

    def fetch_covertype(self):
        """Download and transform the Covertype Data Set.

        https://archive.ics.uci.edu/ml/datasets/Covertype
        """
        data = pd.read_csv(FETCH_URLS["covertype"], header=None)
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        wilderness_area = pd.Series(
            np.argmax(data.iloc[:, 10:14].values, axis=1), name=10
        )
        soil_type = pd.Series(np.argmax(data.iloc[:, 14:54].values, axis=1), name=11)
        data = (
            data.drop(columns=list(range(10, 54)))
            .join(wilderness_area)
            .join(soil_type)[list(range(0, 12)) + ["target"]]
        )
        categorical_features = [10, 11]
        return data, categorical_features

    def fetch_credit_approval(self):
        """Download and transform the Credit Approval Data Set.

        https://archive.ics.uci.edu/ml/datasets/Credit+Approval
        """
        data = pd.read_csv(
            FETCH_URLS["credit_approval"], header=None, na_values="?"
        ).dropna()
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12]
        return data, categorical_features

    def fetch_dermatology(self):
        """Download and transform the Dermatology Data Set.

        https://archive.ics.uci.edu/ml/datasets/Dermatology
        """
        data = pd.read_csv(
            FETCH_URLS["dermatology"], header=None, na_values="?"
        ).dropna()
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = list(range(data.shape[1] - 1))
        categorical_features.remove(33)
        return data, categorical_features

    def fetch_echocardiogram(self):
        """Download and transform the Echocardiogram Data Set.

        https://archive.ics.uci.edu/ml/datasets/Echocardiogram
        """
        data = pd.read_csv(
            FETCH_URLS["echocardiogram"],
            header=None,
            on_bad_lines="skip",
            na_values="?",
        )
        data.drop(columns=[10, 11], inplace=True)
        data.dropna(inplace=True)
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [1, 3]
        return data, categorical_features

    def fetch_flags(self):
        """Download and transform the Flags Data Set.

        https://archive.ics.uci.edu/ml/datasets/Flags
        """
        data = pd.read_csv(FETCH_URLS["flags"], header=None)
        target = data[6].rename("target")
        data = data.drop(columns=[0, 6]).T.reset_index(drop=True).T.join(target)
        categorical_features = [
            0,
            1,
            4,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]
        return data, categorical_features

    def fetch_heart_disease(self):
        """Download and transform the Heart Disease Data Set.

        https://archive.ics.uci.edu/ml/datasets/Heart+Disease
        """
        data = (
            pd.concat(
                [
                    pd.read_csv(url, header=None, na_values="?")
                    for url in FETCH_URLS["heart_disease"]
                ],
                ignore_index=True,
            )
            .drop(columns=[10, 11, 12])
            .dropna()
        )
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [1, 2, 5, 6, 8]
        return data, categorical_features

    def fetch_hepatitis(self):
        """Download and transform the Hepatitis Data Set.

        https://archive.ics.uci.edu/ml/datasets/Hepatitis
        """
        data = (
            pd.read_csv(FETCH_URLS["hepatitis"], header=None, na_values="?")
            .drop(columns=[15, 18])
            .dropna()
        )
        target = data[0].rename("target")
        data = data.drop(columns=[0]).T.reset_index(drop=True).T.join(target)
        categorical_features = list(range(1, 13)) + [16]
        return data, categorical_features

    def fetch_german_credit(self):
        """Download and transform the German Credit Data Set.

        https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
        """
        data = pd.read_csv(FETCH_URLS["german_credit"], header=None, sep=" ")
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = (
            np.argwhere(data.iloc[0, :-1].apply(lambda x: str(x)[0] == "A").values)
            .squeeze()
            .tolist()
        )
        return data, categorical_features

    def fetch_heart(self):
        """Download and transform the Heart Data Set.

        http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
        """
        data = pd.read_csv(FETCH_URLS["heart"], header=None, delim_whitespace=True)
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        categorical_features = [1, 2, 5, 6, 8, 10, 12]
        return data, categorical_features

    def fetch_thyroid(self):
        """Download and transform the Thyroid Disease Data Set.
        Label 0 corresponds to no disease found.
        Label 1 corresponds to one or multiple diseases found.

        https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
        """
        data = (
            pd.read_csv(FETCH_URLS["thyroid"], header=None, na_values="?")
            .drop(columns=27)
            .dropna()
            .T.reset_index(drop=True)
            .T
        )
        data.rename(columns={data.columns[-1]: "target"}, inplace=True)
        data["target"] = (
            data["target"].apply(lambda x: x.split("[")[0]) != "-"
        ).astype(int)
        categorical_features = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            20,
            22,
            24,
            26,
            27,
        ]
        return data, categorical_features


class MultiClassDatasets(Datasets):
    """Class to download, transform and save multi-class datasets."""

    def fetch_first_order_theorem(self):
        """Download and transform the First Order Theorem Data Set.

        https://www.openml.org/d/1475
        """
        data = pd.read_csv(FETCH_URLS["first_order_theorem"])
        data.rename(columns={"Class": "target"}, inplace=True)
        return data

    def fetch_gas_drift(self):
        """Download and transform the Gas Drift Data Set.

        https://www.openml.org/d/1476
        """
        data = pd.read_csv(FETCH_URLS["gas_drift"])
        data.rename(columns={"Class": "target"}, inplace=True)
        return data

    def fetch_autouniv_au7(self):
        """Download and transform the AutoUniv au7 Data Set

        https://www.openml.org/d/1552
        """
        data = pd.read_csv(FETCH_URLS["autouniv_au7"])
        data.rename(columns={"Class": "target"}, inplace=True)
        data.target = data.target.apply(lambda x: x.replace("class", "")).astype(int)

        mask = (data.iloc[:, :-1].nunique() > 10).tolist()
        mask.append(True)
        data = data.loc[:, mask].copy()
        return data

    def fetch_autouniv_au4(self):
        """Download and transform the AutoUniv au4 Data Set

        https://www.openml.org/d/1548
        """
        data = pd.read_csv(FETCH_URLS["autouniv_au4"])
        data.rename(columns={"Class": "target"}, inplace=True)
        data.target = data.target.apply(lambda x: x.replace("class", "")).astype(int)

        mask = (data.iloc[:, :-1].nunique() > 10).tolist()
        mask.append(True)
        data = data.loc[:, mask].copy()
        return data

    def fetch_mice_protein(self):
        """Download and transform the Mice Protein Data Set

        https://www.openml.org/d/40966
        """
        data = pd.read_csv(FETCH_URLS["mice_protein"])
        data.rename(columns={"class": "target"}, inplace=True)
        data.drop(columns=["MouseID"], inplace=True)
        data.replace("?", np.nan, inplace=True)

        mask = (data.iloc[:, :-1].nunique() > 10).tolist()
        mask.append(True)
        mask2 = data.isna().sum() < 10
        data = data.loc[:, mask & mask2].dropna().copy()

        data.iloc[:, :-1] = data.iloc[:, :-1].astype(float)

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)
        return data

    def fetch_steel_plates(self):
        """Download and transform the Steel Plates Fault Data Set.

        https://www.openml.org/d/40982
        """
        data = pd.read_csv(FETCH_URLS["steel_plates"])

        mask = (data.iloc[:, :-1].nunique() > 10).tolist()
        mask.append(True)
        data = data.loc[:, mask].copy()

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)
        return data

    def fetch_cardiotocography(self):
        """Download and transform the Cardiotocography Data Set.

        https://www.openml.org/d/1560
        """
        data = pd.read_csv(FETCH_URLS["cardiotocography"])
        data.rename(columns={"Class": "target"}, inplace=True)

        mask = (data.iloc[:, :-1].nunique() > 10).tolist()
        mask.append(True)
        data = data.loc[:, mask].copy()

        return data

    def fetch_waveform(self):
        """Download and transform the Waveform Database Generator (version 2) Data Set.

        https://www.openml.org/d/60
        """
        data = pd.read_csv(FETCH_URLS["waveform"])
        data.rename(columns={"class": "target"}, inplace=True)
        return data

    def fetch_volkert(self):
        """Download and transform the Volkert Data Set.

        https://www.openml.org/d/41166
        """
        data = pd.read_csv(FETCH_URLS["volkert"])
        data.rename(columns={"class": "target"}, inplace=True)

        mask = (data.iloc[:, 1:].nunique() > 100).tolist()
        mask.insert(0, True)
        data = data.loc[:, mask].copy()
        return data

    def fetch_vehicle(self):
        """Download and transform the Vehicle Silhouettes Data Set.

        https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
        """
        data = pd.DataFrame()
        for letter in ascii_lowercase[0:9]:
            partial_data = pd.read_csv(
                urljoin(FETCH_URLS["vehicle"], "xa%s.dat" % letter),
                header=None,
                delim_whitespace=True,
            )
            partial_data = partial_data.rename(columns={18: "target"})
            data = data.append(partial_data)

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)

        return data

    def fetch_asp_potassco(self):
        """Download and transform the ASP-POTASSCO Data Set.

        https://www.openml.org/d/41705
        """
        data = pd.read_csv(FETCH_URLS["asp_potassco"], na_values="?")
        data.dropna(inplace=True)
        data["target"] = data["algorithm"]
        data.drop(columns=["instance_id", "algorithm"], inplace=True)

        mask = (data.iloc[:, :-1].nunique() > 100).tolist()
        mask.append(True)
        data = data.loc[:, mask].copy()

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)

        return data

    def fetch_wine_quality(self):
        """Download and transform the Wine Quality Data Set.

        https://www.openml.org/d/40691
        """
        data = pd.read_csv(FETCH_URLS["wine_quality"])
        data.rename(columns={"class": "target"}, inplace=True)
        return data

    def fetch_mfeat_zernike(self):
        """Download and transform the Multiple Features Dataset: Zernike Data Set.

        https://www.openml.org/d/22
        """
        data = pd.read_csv(FETCH_URLS["mfeat_zernike"])
        data.drop_duplicates(inplace=True)
        data.rename(columns={"class": "target"}, inplace=True)
        return data

    def fetch_gesture_segmentation(self):
        """Download and transform the Gesture Phase Segmentation Data Set.

        https://www.openml.org/d/4538
        """
        data = pd.read_csv(FETCH_URLS["gesture_segmentation"])
        data.rename(columns={"Phase": "target"}, inplace=True)

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)
        return data

    def fetch_texture(self):
        """Download and transform the Texture Data Set.

        https://www.openml.org/d/40499
        """
        data = pd.read_csv(FETCH_URLS["texture"])
        data.drop_duplicates(inplace=True)
        data.rename(columns={"Class": "target"}, inplace=True)
        return data

    def fetch_usps(self):
        """Download and transform the USPS Data Set.

        https://www.openml.org/data/get_csv/19329737/usps.arff
        """
        data = pd.read_csv(FETCH_URLS["usps"])
        data.rename(columns={"int0": "target"}, inplace=True)
        return data

    def fetch_vowels(self):
        """Download and transform the Vowels Data Set.

        https://www.openml.org/d/375
        """
        data = pd.read_csv(FETCH_URLS["vowels"])
        data.rename(columns={"speaker": "target"}, inplace=True)
        data.drop(columns=["utterance", "frame"], inplace=True)
        return data

    def fetch_pendigits(self):
        """Download and transform the Pen-Based Recognition of Handwritten
        Digits Data Set.

        https://www.openml.org/d/32
        """
        data = pd.read_csv(FETCH_URLS["pendigits"])
        data.rename(columns={"class": "target"}, inplace=True)
        return data

    def fetch_image_segmentation(self):
        """Download and transform the Image Segmentation Data Set.

        https://www.openml.org/d/40984
        """
        data = pd.read_csv(FETCH_URLS["image_segmentation"])
        data.drop(columns=data.columns[:5], inplace=True)
        data.rename(columns={"class": "target"}, inplace=True)

        mapper = {v: k for k, v in enumerate(data.target.unique())}
        data.target = data.target.map(mapper)
        return data

    def fetch_baseball(self):
        """Download and transform the Baseball Hall of Fame Data Set.

        https://www.openml.org/d/185
        """
        data = pd.read_csv(FETCH_URLS["baseball"], na_values="?")
        data.drop(columns=["Player", "Position"], inplace=True)
        data.rename(columns={"Hall_of_Fame": "target"}, inplace=True)
        data.dropna(inplace=True)
        return data
