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
from ..utils import img_array_to_pandas


class RemoteSensingDatasets(Datasets):
    """Class to download, transform and save remote sensing datasets."""

    def __init__(self, names="all", return_coords=False):
        self.names = names
        self.return_coords = return_coords

    def _load_gic_dataset(self, dataset_name):
        for url in FETCH_URLS[dataset_name]:
            r = requests.get(url, stream=True)
            content = loadmat(io.BytesIO(r.content))
            arr = np.array(list(content.values())[-1])
            arr = np.expand_dims(arr, -1) if arr.ndim == 2 else arr
            if self.return_coords and arr.shape[-1] != 1:
                indices = np.moveaxis(np.indices(arr.shape[:-1]), 0, -1)
                arr = np.insert(arr, [0, 0], indices, -1)
            yield arr

    def fetch_indian_pines(self):
        """Download and transform the Indian Pines Data Set. Label "0" means
        the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines
        """
        df = img_array_to_pandas(*self._load_gic_dataset("indian_pines"))
        return df[df.target != 0]

    def fetch_salinas(self):
        """Download and transform the Salinas Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset("salinas"))
        return df[df.target != 0]

    def fetch_salinas_a(self):
        """Download and transform the Salinas-A Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset("salinas_a"))
        return df[df.target != 0]

    def fetch_pavia_centre(self):
        """Download and transform the Pavia Centre Data Set. Label "0" means the pixel
        is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset("pavia_centre"))
        return df[df.target != 0]

    def fetch_pavia_university(self):
        """Download and transform the Pavia University Data Set. Label "0"
        means the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset("pavia_university"))
        return df[df.target != 0]

    def fetch_kennedy_space_center(self):
        """Download and transform the Kennedy Space Center Data Set. Label "0"
        means the pixel is not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Kennedy_Space_Center_.28KSC.29
        """
        df = img_array_to_pandas(*self._load_gic_dataset("kennedy_space_center"))
        return df[df.target != 0]

    def fetch_botswana(self):
        """Download and transform the Botswana Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Botswana
        """
        df = img_array_to_pandas(*self._load_gic_dataset("botswana"))
        return df[df.target != 0]
