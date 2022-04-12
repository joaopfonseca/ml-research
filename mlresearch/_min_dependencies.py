"""All minimum dependencies for ml-research."""
import argparse

NUMPY_MIN_VERSION = "1.14.6"
PANDAS_MIN_VERSION = "1.3.5"
SKLEARN_MIN_VERSION = "1.0.0"
IMBLEARN_MIN_VERSION = "0.8.0"
RICH_MIN_VERSION = "10.16.1"
MATPLOTLIB_MIN_VERSION = "2.2.3"
SEABORN_MIN_VERSION = "0.9.0"
RLEARN_MIN_VERSION = "0.2.1"
PYTORCH_MIN_VERSION = "1.10.1"
TORCHVISION_MIN_VERSION = "0.11.2"
PYTORCH_LIGHTNING_VERSION = "1.5.8"

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "pandas": (PANDAS_MIN_VERSION, "install"),
    "numpy": (NUMPY_MIN_VERSION, "install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    "imbalanced-learn": (IMBLEARN_MIN_VERSION, "install"),
    "rich": (RICH_MIN_VERSION, "install"),
    "matplotlib": (MATPLOTLIB_MIN_VERSION, "install"),
    "seaborn": (SEABORN_MIN_VERSION, "install"),
    "research-learn": (RLEARN_MIN_VERSION, "install"),
    "requests": ("2.26.0", "install"),
    "torch": (PYTORCH_MIN_VERSION, "install"),
    "torchvision": (TORCHVISION_MIN_VERSION, "install"),
    "pytorch-lightning": (PYTORCH_LIGHTNING_VERSION, "install"),
    "pytest-cov": ("3.0.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("22.3", "tests"),
    "pylint": ("2.12.2", "tests"),
    "coverage": ("6.2", "tests"),
    "sphinx": ("4.2.0", "docs"),
    "numpydoc": ("1.0.0", "docs, tests"),
    "sphinx-material": ("0.0.35", "docs"),
    "nbsphinx": ("0.8.7", "docs"),
    "recommonmark": ("0.7.1", "docs"),
    "sphinx-markdown-tables": ("0.0.15", "docs"),
    "sphinx-copybutton": ("0.4.0", "docs"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
