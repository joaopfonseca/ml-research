"""
Download, transform and simulate various datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from typing import Optional, Union
import os
from os.path import expanduser, join
from collections import Counter
from urllib.parse import urljoin
from sqlite3 import connect
from rich.progress import track
import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from imblearn.datasets import make_imbalance

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
GIC_URL = "http://www.ehu.eus/ccwintco/uploads/"
OPENML_URL = "https://www.openml.org/data/get_csv/"
FETCH_URLS = {
    "breast_tissue": urljoin(UCI_URL, "00192/BreastTissue.xls"),
    "ecoli": urljoin(UCI_URL, "ecoli/ecoli.data"),
    "eucalyptus": urljoin(OPENML_URL, "3625/dataset_194_eucalyptus.arff"),
    "glass": urljoin(UCI_URL, "glass/glass.data"),
    "haberman": urljoin(UCI_URL, "haberman/haberman.data"),
    "heart": urljoin(UCI_URL, "statlog/heart/heart.dat"),
    "iris": urljoin(UCI_URL, "iris/bezdekIris.data"),
    "libras": urljoin(UCI_URL, "libras/movement_libras.data"),
    "liver": urljoin(UCI_URL, "liver-disorders/bupa.data"),
    "pima": "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f"
    "/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv",
    "vehicle": urljoin(UCI_URL, "statlog/vehicle/"),
    "wine": urljoin(UCI_URL, "wine/wine.data"),
    "new_thyroid": urljoin(UCI_URL, "thyroid-disease/new-thyroid.data"),
    "cleveland": urljoin(UCI_URL, "heart-disease/processed.cleveland.data"),
    "led": urljoin(OPENML_URL, "4535757/phpSj3fWL"),
    "page_blocks": urljoin(OPENML_URL, "30/dataset_30_page-blocks.arff"),
    "yeast": urljoin(UCI_URL, "yeast/yeast.data"),
    "banknote_authentication": urljoin(
        UCI_URL, "00267/data_banknote_authentication.txt"
    ),
    "arcene": urljoin(UCI_URL, "arcene/"),
    "audit": urljoin(UCI_URL, "00475/audit_data.zip"),
    "spambase": urljoin(UCI_URL, "spambase/spambase.data"),
    "parkinsons": urljoin(UCI_URL, "parkinsons/parkinsons.data"),
    "ionosphere": urljoin(UCI_URL, "ionosphere/ionosphere.data"),
    "breast_cancer": urljoin(UCI_URL, "breast-cancer-wisconsin/wdbc.data"),
    "adult": urljoin(UCI_URL, "adult/adult.data"),
    "abalone": urljoin(UCI_URL, "abalone/abalone.data"),
    "acute": urljoin(UCI_URL, "acute/diagnosis.data"),
    "annealing": urljoin(UCI_URL, "annealing/anneal.data"),
    "census": urljoin(UCI_URL, "census-income-mld/census-income.data.gz"),
    "contraceptive": urljoin(UCI_URL, "cmc/cmc.data"),
    "covertype": urljoin(UCI_URL, "covtype/covtype.data.gz"),
    "credit_approval": urljoin(UCI_URL, "credit-screening/crx.data"),
    "dermatology": urljoin(UCI_URL, "dermatology/dermatology.data"),
    "echocardiogram": urljoin(UCI_URL, "echocardiogram/echocardiogram.data"),
    "flags": urljoin(UCI_URL, "flags/flag.data"),
    "heart_disease": [
        urljoin(UCI_URL, "heart-disease/processed.cleveland.data"),
        urljoin(UCI_URL, "heart-disease/processed.hungarian.data"),
        urljoin(UCI_URL, "heart-disease/processed.switzerland.data"),
        urljoin(UCI_URL, "heart-disease/processed.va.data"),
    ],
    "hepatitis": urljoin(UCI_URL, "hepatitis/hepatitis.data"),
    "german_credit": urljoin(UCI_URL, "statlog/german/german.data"),
    "thyroid": urljoin(UCI_URL, "thyroid-disease/thyroid0387.data"),
    "first_order_theorem": urljoin(OPENML_URL, "1587932/phpPbCMyg"),
    "gas_drift": urljoin(OPENML_URL, "1588715/phpbL6t4U"),
    "autouniv_au7": urljoin(OPENML_URL, "1593748/phpmRPvKy"),
    "autouniv_au4": urljoin(OPENML_URL, "1593744/phpiubDlf"),
    "mice_protein": urljoin(OPENML_URL, "17928620/phpchCuL5"),
    "steel_plates": urljoin(OPENML_URL, "18151921/php5s7Ep8"),
    "cardiotocography": urljoin(OPENML_URL, "1593756/phpW0AXSQ"),
    "waveform": urljoin(OPENML_URL, "60/dataset_60_waveform-5000.arff"),
    "volkert": urljoin(OPENML_URL, "19335689/file1c556e3db171.arff"),
    "asp_potassco": urljoin(OPENML_URL, "21377447/file18547f421393.arff"),
    "wine_quality": urljoin(OPENML_URL, "4965268/wine-quality-red.arff"),
    "mfeat_zernike": urljoin(OPENML_URL, "22/dataset_22_mfeat-zernike.arff"),
    "gesture_segmentation": urljoin(OPENML_URL, "1798765/phpYLeydd"),
    "texture": urljoin(OPENML_URL, "4535764/phpBDgUyY"),
    "usps": urljoin(OPENML_URL, "19329737/usps.arff"),
    "vowels": urljoin(OPENML_URL, "52415/JapaneseVowels.arff"),
    "pendigits": urljoin(OPENML_URL, "32/dataset_32_pendigits.arff"),
    "image_segmentation": urljoin(OPENML_URL, "18151937/phpyM5ND4"),
    "baseball": urljoin(OPENML_URL, "3622/dataset_189_baseball.arff"),
    "indian_pines": [
        urljoin(GIC_URL, "2/22/Indian_pines.mat"),
        urljoin(GIC_URL, "c/c4/Indian_pines_gt.mat"),
    ],
    "salinas": [
        urljoin(GIC_URL, "f/f1/Salinas.mat"),
        urljoin(GIC_URL, "f/fa/Salinas_gt.mat"),
    ],
    "salinas_a": [
        urljoin(GIC_URL, "d/df/SalinasA.mat"),
        urljoin(GIC_URL, "a/aa/SalinasA_gt.mat"),
    ],
    "pavia_centre": [
        urljoin(GIC_URL, "e/e3/Pavia.mat"),
        urljoin(GIC_URL, "5/53/Pavia_gt.mat"),
    ],
    "pavia_university": [
        urljoin(GIC_URL, "e/ee/PaviaU.mat"),
        urljoin(GIC_URL, "5/50/PaviaU_gt.mat"),
    ],
    "kennedy_space_center": [
        urljoin(GIC_URL, "2/26/KSC.mat"),
        urljoin(GIC_URL, "a/a6/KSC_gt.mat"),
    ],
    "botswana": [
        urljoin(GIC_URL, "7/72/Botswana.mat"),
        urljoin(GIC_URL, "5/58/Botswana_gt.mat"),
    ],
}
RANDOM_STATE = 0


def get_data_home(data_home: Optional[str] = None) -> str:
    """Return the path of the ml-research data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'ml_research_data' in the
    user home folder.
    Alternatively, it can be set programmatically by giving an explicit folder
    path. The '~' symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str, default=None
        The path to the data directory. If `None`, the default path
        is `~/ml_research_data`.
    """
    if data_home is None:
        data_home = join("~", "mlresearch_data")
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


class Datasets:
    """
    Base class to download and save datasets.
    """

    def __init__(
        self,
        names: Union[str, list] = "all",
        data_home: str = None,
        download_if_missing: bool = True,
    ):
        """
        Parameters
        ----------
        names : str or list, default="all"
            List of dataset names to be downloaded. If ``all``, downloads all datasets.

        data_home : str, default=None
            The path to the data directory. If `None`, the default path
            is `~/ml_research_data`.

        download_if_missing : bool, default=True
            If True, downloads the dataset from the internet and puts it in
            ``data_home``. If the dataset is already downloaded, it is not downloaded
            again.

        Attributes
        ----------
        data_home_ : str
            Path were the data was stored.

        content_ : list
            List of tuples composed of (Dataset name, Dataframe).
        """
        self.names = names
        self.data_home = data_home
        self.download_if_missing = download_if_missing

    @staticmethod
    def _modify_columns(data):
        """Rename and reorder columns of dataframe."""
        X, y = data.drop(columns="target"), data.target
        X.columns = range(len(X.columns))
        return pd.concat([X, y], axis=1)

    @staticmethod
    def _calculate_sampling_strategy(ir, y):
        """Calculate ratio based on the IR."""
        ratio = Counter(y).most_common()
        b = ratio[0][1]
        c_min = ratio[0][1] / ir
        m = (c_min - b) / (len(ratio) - 1)

        sampling_strategy = {}
        for x, (c, f) in enumerate(ratio):
            f_new = int(np.clip(m * x + b, 1, f))
            sampling_strategy[c] = f_new

        return sampling_strategy

    def _make_imbalance(self, data, sampling_strategy, random_state=None):
        """Undersample the minority class."""
        X, y = check_X_y(data.drop(columns="target"), data["target"], dtype=None)
        X, y = make_imbalance(
            X, y, sampling_strategy=sampling_strategy, random_state=random_state
        )
        data = pd.DataFrame(np.column_stack((X, y)))
        data[data.columns[-1]] = data[data.columns[-1]].astype(int)
        return data

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
                df = getattr(self, func_name)()
                df.to_csv(join(self.data_home_, file_name), index=False)

            data = pd.read_csv(join(self.data_home_, file_name))
            data = self._modify_columns(data)
            self.content_.append((name, data))
        return self

    def imbalance_datasets(self, imbalance_ratio: float, random_state: int = None):
        """
        Appends imbalanced versions of datasets with predefined imbalance ratios to
        ``self.content_``.

        $IR = \frac{|C_{maj}|}{|C_{min}|}$

        Parameters
        ----------
        imbalance_ratio : float
            Final Imbalance Ratio expected in the datasets.

        random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.

            - If int, ``random_state`` is the seed used by the random number
              generator;
            - If ``RandomState`` instance, random_state is the random number
              generator;
            - If ``None``, the random number generator is the ``RandomState``
              instance used by ``np.random``.

        Returns
        -------
        self

        """
        imbalanced_content = []
        base_content = [
            dataset for dataset in self.content_ if not dataset[0].endswith(")")
        ]
        for name, data in base_content:
            base_freqs = Counter(data.target).values()
            base_ir = int(max(base_freqs) / min(base_freqs))
            sampling_strategy = self._calculate_sampling_strategy(
                ir=imbalance_ratio, y=data.target
            )
            data_imb = self._make_imbalance(
                data, sampling_strategy=sampling_strategy, random_state=random_state
            )

            data_imb.columns = data.columns

            freqs = Counter(data_imb.target).values()
            new_ir = int(max(freqs) / min(freqs))
            name_imb = f"{name} ({new_ir})"
            if name_imb not in dict(self.content_).keys() and base_ir < new_ir:
                imbalanced_content.append((name_imb, data_imb))

        self.content_.extend(imbalanced_content)
        return self

    def summarize_datasets(self):
        """Create a summary of the downloaded datasets."""

        # Check datasets format
        datasets = [
            (name, (data.drop(columns="target"), data.target))
            if type(data) == pd.DataFrame
            else data
            for name, data in self.content_
        ]

        # Define summary table columns
        summary_columns = [
            "Dataset name",
            "Features",
            "Instances",
            "Minority instances",
            "Majority instances",
            "Imbalance Ratio",
            "Classes",
        ]

        # Define empty summary table
        datasets_summary = []

        # Populate summary table
        for dataset_name, (X, y) in datasets:
            n_instances = Counter(y).values()
            n_minority_instances = min(n_instances)
            n_majority_instances = max(n_instances)
            values = [
                dataset_name,
                X.shape[1],
                len(X),
                n_minority_instances,
                n_majority_instances,
                round(n_majority_instances / n_minority_instances, 2),
                len(n_instances),
            ]
            datasets_summary.append(values)
        datasets_summary = pd.DataFrame(datasets_summary, columns=summary_columns)

        # Cast to integer columns
        int_cols = datasets_summary.columns.drop(["Dataset name", "Imbalance Ratio"])
        datasets_summary.loc[:, int_cols] = datasets_summary.loc[:, int_cols].astype(
            int
        )

        # Sort datasets summary
        datasets_summary = datasets_summary.sort_values("Imbalance Ratio").reset_index(
            drop=True
        )

        return datasets_summary

    def save(self, path, db_name):
        """Save datasets."""
        with connect(join(path, f"{db_name}.db")) as connection:
            for name, data in self.content_:
                data.to_sql(name, connection, index=False, if_exists="replace")
