from urllib.request import urlopen
import multiprocessing.dummy as mp
from multiprocessing import cpu_count
import ssl
import numpy as np
import pandas as pd

from ..base import Datasets, FETCH_URLS

ssl._create_default_https_context = ssl._create_unverified_context

MIN_CLASS = 20
MAJ_CLASS = 100
N_FEATURES = 4
X = pd.DataFrame(np.random.random((30 + MIN_CLASS + MAJ_CLASS, N_FEATURES)))
y = pd.Series(
    [
        *[0 for i in range(MIN_CLASS)],
        *[1 for i in range(30)],
        *[2 for i in range(MAJ_CLASS)],
    ],
    name="target",
)


def test_urls():
    """Test whether URLS are working."""
    urls = [
        url
        for sublist in [[url] for url in list(FETCH_URLS.values()) if type(url) == str]
        for url in sublist
    ]

    p = mp.Pool(cpu_count())
    url_status = p.map(lambda url: (urlopen(url).status == 200), urls)

    assert all(url_status)


def test_imbalance_datasets():
    """Test if the imbalance_datasets and summarize_datasets functions are working."""
    base_ir = MAJ_CLASS / MIN_CLASS
    irs = [1, 5, 10, 100]
    exp_irs = [i for i in irs if i > base_ir]

    name = "test"
    data = pd.concat([X, y], axis=1)

    datasets = Datasets()
    datasets.content_ = [(name, data)]
    for ir in irs:
        datasets.imbalance_datasets(ir)

    descr = datasets.summarize_datasets()
    content = datasets.content_

    # imbalance_datasets
    assert len(content) == len(exp_irs) + 1
    assert list(dict(content).keys()) == [name] + [f"{name} ({ir})" for ir in exp_irs]

    # summarize_datasets
    assert descr.shape == (len(exp_irs) + 1, 7)
    assert (descr["Features"] == N_FEATURES).all()
    assert descr["Dataset name"].tolist() == list(dict(content).keys())
    assert descr["Imbalance Ratio"].astype(int).tolist() == [
        i for i in irs if i >= base_ir
    ]
