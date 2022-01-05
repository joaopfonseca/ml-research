import os
from os.path import expanduser, join, exists

# Torchvision is an optional dependency
try:
    from torchvision import datasets
except ModuleNotFoundError:
    pass


def get_data_home(data_home=None) -> str:
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
        data_home = join("~", "ml_research_data")
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


class PytorchDatasets:
    """
    Download and save Pytorch image datasets. The data is stored both locally and in
    memory.
    """

    def __init__(
        self,
        names: list = ["cifar10", "cifar100", "svhn", "fashionmnist"],
        data_home: str = None,
        download_if_missing: bool = True,
    ):
        self.names = names
        self.data_home = data_home
        self.download_if_missing = download_if_missing

    def fetch_cifar10(self):
        """Download the CIFAR-10 Data Set.

        https://www.cs.toronto.edu/~kriz/cifar.html
        """
        train = datasets.CIFAR10(
            root=self.data_home_, train=True, download=self.download_if_missing
        )
        test = datasets.CIFAR10(
            root=self.data_home_, train=False, download=self.download_if_missing
        )
        if exists(join(self.data_home_, "cifar-10-python.tar.gz")):
            os.remove(join(self.data_home_, "cifar-10-python.tar.gz"))
        return (train.data, train.targets), (test.data, test.targets)

    def fetch_cifar100(self):
        """Download the CIFAR-100 Data Set.

        https://www.cs.toronto.edu/~kriz/cifar.html
        """
        train = datasets.CIFAR100(
            root=self.data_home_, train=True, download=self.download_if_missing
        )
        test = datasets.CIFAR100(
            root=self.data_home_, train=False, download=self.download_if_missing
        )
        if exists(join(self.data_home_, "cifar-100-python.tar.gz")):
            os.remove(join(self.data_home_, "cifar-100-python.tar.gz"))
        return (train.data, train.targets), (test.data, test.targets)

    def fetch_svhn(self):
        """Download the Street View House Numbers (SVHN) Data Set. The ``extra`` set is
        not downloaded.

        http://ufldl.stanford.edu/housenumbers/
        """
        train = datasets.SVHN(
            root=join(self.data_home_, "svhn"),
            split="train",
            download=self.download_if_missing,
        )
        test = datasets.SVHN(
            root=join(self.data_home_, "svhn"),
            split="test",
            download=self.download_if_missing,
        )
        return (train.data, train.labels), (test.data, test.labels)

    def fetch_fashionmnist(self):
        """Download the Fashion-MNIST Data Set.

        https://github.com/zalandoresearch/fashion-mnist
        """
        train = datasets.FashionMNIST(
            root=self.data_home_, train=True, download=self.download_if_missing
        )
        test = datasets.FashionMNIST(
            root=self.data_home_, train=False, download=self.download_if_missing
        )
        return (train.data, train.targets), (test.data, test.targets)

    def download(self):
        """Fetch/Download datasets."""
        self.data_home_ = get_data_home(data_home=self.data_home)

        if self.names == "all":
            func_names = [func_name for func_name in dir(self) if "fetch_" in func_name]
        else:
            func_names = [
                f"fetch_{name}".lower().replace(" ", "_") for name in self.names
            ]
        self.content_ = []
        for func_name in func_names:
            name = func_name.replace("fetch_", "").upper().replace("_", " ")
            fetch_data = getattr(self, func_name)
            data = fetch_data()
            self.content_.append((name, *data))
        return self
