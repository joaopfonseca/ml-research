"""
Download, transform and simulate various datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

from os.path import join
from urllib.parse import urljoin
from sqlite3 import connect
from rich.progress import track
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
KEEL_URL = "http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/"
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
    "new_thyroid_1": urljoin(
        urljoin(KEEL_URL, "imb_IRlowerThan9/"), "new-thyroid1.zip"
    ),
    "new_thyroid_2": urljoin(
        urljoin(KEEL_URL, "imb_IRlowerThan9/"), "new-thyroid2.zip"
    ),
    "cleveland": urljoin(
        urljoin(KEEL_URL, "imb_IRhigherThan9p2/"), "cleveland-0_vs_4.zip"
    ),
    "led": urljoin(
        urljoin(KEEL_URL, "imb_IRhigherThan9p2/"), "led7digit-0-2-4-5-6-7-8-9_vs_1.zip"
    ),
    "page_blocks_1_3": urljoin(
        urljoin(KEEL_URL, "imb_IRhigherThan9p1/"), "page-blocks-1-3_vs_4.zip"
    ),
    "vowel": urljoin(urljoin(KEEL_URL, "imb_IRhigherThan9p1/"), "vowel0.zip"),
    "yeast_1": urljoin(urljoin(KEEL_URL, "imb_IRlowerThan9/"), "yeast1.zip"),
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
    "japanese_vowels": urljoin(OPENML_URL, "52415/JapaneseVowels.arff"),
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


class Datasets:
    """Base class to download and save datasets."""

    def __init__(self, names="all"):
        self.names = names

    @staticmethod
    def _modify_columns(data):
        """Rename and reorder columns of dataframe."""
        X, y = data.drop(columns="target"), data.target
        X.columns = range(len(X.columns))
        return pd.concat([X, y], axis=1)

    def download(self):
        """Download the datasets."""
        if self.names == "all":
            func_names = [func_name for func_name in dir(self) if "fetch_" in func_name]
        else:
            func_names = [
                f"fetch_{name}".lower().replace(" ", "_") for name in self.names
            ]
        self.content_ = []
        for func_name in track(func_names, description="Datasets"):
            name = func_name.replace("fetch_", "").upper().replace("_", " ")
            fetch_data = getattr(self, func_name)
            data = self._modify_columns(fetch_data())
            self.content_.append((name, data))
        return self

    def save(self, path, db_name):
        """Save datasets."""
        with connect(join(path, f"{db_name}.db")) as connection:
            for name, data in self.content_:
                data.to_sql(name, connection, index=False, if_exists="replace")
