<div align="center">
<img src="docs/_static/logo.png" width="400px">
</div>

______________________________________________________________________

<p align="center">
<a href="https://github.com/joaopfonseca/ml-research/actions/workflows/ci.yml"><img alt="Github Actions" src="https://github.com/joaopfonseca/ml-research/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://codecov.io/gh/joaopfonseca/ml-research"><img alt="Codecov" src="https://codecov.io/gh/joaopfonseca/ml-research/branch/master/graph/badge.svg?token=J2EBA4YTMN"></a>
<a href="https://mlresearch.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/mlresearch/badge/?version=latest"></a>
<a href="https://github.com/psf/black"><img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://img.shields.io/badge/python-3.8%20|%203.9-blue"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue"></a>
<a href="https://doi.org/10.3390/RS13132619"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.3390/RS13132619.svg"></a>
</p>
<table align="center">
  <tr>
    <td>
      <b>PyPI</b>
    </td>
    <td>
      <a href="https://badge.fury.io/py/ml-research"><img alt="Pypi Version" src="https://badge.fury.io/py/ml-research.svg"></a>
      <a href="https://pepy.tech/project/ml-research"><img alt="Downloads" src="https://static.pepy.tech/personalized-badge/ml-research?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads"></a>
    </td>
  </tr>
  <tr>
    <td>
      <b>Anaconda</b>
    </td>
    <td>
      <a href="https://anaconda.org/conda-forge/ml-research"><img alt="Conda Version" src="https://img.shields.io/conda/vn/conda-forge/ml-research.svg"></a>
      <a href="https://anaconda.org/conda-forge/ml-research"><img alt="Conda Downloads" src="https://img.shields.io/conda/dn/conda-forge/ml-research.svg"></a>
    </td>
  </tr>
</table>

``ML-Research`` is an open source library for machine learning research.  It
contains the software implementation of most algorithms used or developed in
my research. Specifically, it contains ``scikit-learn`` compatible
implementations for Active Learning, Oversampling, Datasets and various
utilities to assist in experiment design and results reporting. Other
techniques, such as self-supervised learning and semi-supervised learning are
currently under development and are being implemented in ``pytorch`` and
intended to be ``scikit-learn`` compatible.

The LaTeX and Python code for generating the
paper, experiments' results and visualizations reported in each paper is
available (whenever possible) in the [publications
repo](https://github.com/joaopfonseca/publications).

Contributions at the algorithm level are available in the
package ``mlresearch``.

## Installation

A Python distribution of version 3.8, 3.9 or 3.10 is required to run this
project. Earlier Python versions might work in most cases, but they are not
tested. ``ML-Research`` requires:

- numpy (>= 1.14.6)
- pandas (>= 1.3.5)
- sklearn (>= 1.0.0)
- imblearn (>= 0.8.0)
- rich (>= 10.16.1)
- matplotlib (>= 2.2.3)
- seaborn (>= 0.9.0)
- pytorch (>= 1.10.1)
- torchvision (>= 0.11.2)
- pytorch_lightning (>= 1.5.8)

### User Installation

The easiest way to install ml-research is using ``pip`` :

    pip install -U ml-research

Or ``conda`` :

    conda install -c conda-forge ml-research

The documentation includes more detailed [installation
instructions](https://mlresearch.readthedocs.io/en/latest/getting-started.html).

### Installing from source

The following commands should allow you to setup the development version of the
project with minimal effort:

    # Clone the project.
    git clone https://github.com/joaopfonseca/ml-research.git
    cd ml-research

    # Create and activate an environment 
    make environment 
    conda activate mlresearch # Adapt this line accordingly if you're not running conda

    # Install project requirements and the research package. Dependecy group
    # "all" will also install both dependency groups shown below.
    pip install .[tests,docs] 

## Citing ML-Research

If you use ML-Research in a scientific publication, we would appreciate
citations to the following paper:

    @article{Fonseca2021,
      doi = {10.3390/RS13132619},
      url = {https://doi.org/10.3390/RS13132619},
      keywords = {SMOTE,active learning,artificial data generation,land use/land cover classification,oversampling},
      year = {2021},
      month = {jul},
      publisher = {Multidisciplinary Digital Publishing Institute},
      volume = {13},
      pages = {2619},
      author = {Fonseca, Joao and Douzas, Georgios and Bacao, Fernando},
      title = {{Increasing the Effectiveness of Active Learning: Introducing Artificial Data Generation in Active Learning for Land Use/Land Cover Classification}},
      journal = {Remote Sensing}
    }
