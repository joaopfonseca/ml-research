"""
Download, transform and simulate various datasets.
"""

# Author: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import io
import requests
import numpy as np
from scipy.io import loadmat

from .base import Datasets, FETCH_URLS
from ..utils import image_to_dataframe


class RemoteSensingDatasets(Datasets):
    """Class to download, transform and save remote sensing datasets."""

    def __init__(
        self,
        names: str = "all",
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

    def download(self):
        """Download the datasets and append undersampled versions of them."""
        super(RemoteSensingDatasets, self).download(keep_index=True)
        content_ = []
        for name, data in self.content_:
            data = data.set_index(["h", "w"])
            content_.append((name, data))
        self.content_ = content_

        return self

    def _load_gic_dataset(self, dataset_name):
        for url in FETCH_URLS[dataset_name]:
            r = requests.get(url, stream=True)
            content = loadmat(io.BytesIO(r.content))
            arr = np.array(list(content.values())[-1])
            arr = np.expand_dims(arr, -1) if arr.ndim == 2 else arr
            if arr.shape[-1] != 1:
                indices = np.moveaxis(np.indices(arr.shape[:-1]), 0, -1)
                arr = np.insert(arr, [0, 0], indices, -1)
            yield arr

    def fetch_indian_pines(self):
        """Download and transform the Indian Pines Data Set. Label "0" means
        the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines
        """
        df = image_to_dataframe(*self._load_gic_dataset("indian_pines"))
        return df[df.target != 0]

    def fetch_salinas(self):
        """Download and transform the Salinas Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene
        """
        df = image_to_dataframe(*self._load_gic_dataset("salinas"))
        return df[df.target != 0]

    def fetch_salinas_a(self):
        """Download and transform the Salinas-A Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene
        """
        df = image_to_dataframe(*self._load_gic_dataset("salinas_a"))
        return df[df.target != 0]

    def fetch_pavia_centre(self):
        """Download and transform the Pavia Centre Data Set. Label "0" means the pixel
        is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene
        """
        df = image_to_dataframe(*self._load_gic_dataset("pavia_centre"))
        return df[df.target != 0]

    def fetch_pavia_university(self):
        """Download and transform the Pavia University Data Set. Label "0"
        means the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene
        """
        df = image_to_dataframe(*self._load_gic_dataset("pavia_university"))
        return df[df.target != 0]

    def fetch_kennedy_space_center(self):
        """Download and transform the Kennedy Space Center Data Set. Label "0"
        means the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Kennedy_Space_Center_.28KSC.29
        """
        df = image_to_dataframe(*self._load_gic_dataset("kennedy_space_center"))
        return df[df.target != 0]

    def fetch_botswana(self):
        """Download and transform the Botswana Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Botswana
        """
        df = image_to_dataframe(*self._load_gic_dataset("botswana"))
        return df[df.target != 0]
