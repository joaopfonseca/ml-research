<div align="center">
<img src="docs/_static/logo.png" width="400px">
</div>

______________________________________________________________________

<p align="center">
<a href="https://github.com/joaopfonseca/ml-research/actions/workflows/ci.yml"><img alt="Github Actions" src="https://github.com/joaopfonseca/ml-research/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://codecov.io/gh/joaopfonseca/ml-research"><img alt="Codecov" src="https://codecov.io/gh/joaopfonseca/ml-research/branch/master/graph/badge.svg?token=J2EBA4YTMN"></a>
<a href="https://mlresearch.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/mlresearch/badge/?version=latest"></a>
<a href="https://github.com/psf/black"><img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"></a>
<a href="https://doi.org/10.1155/2023/7941878"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.1155/2023/7941878.svg"></a>
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

``ML-Research`` is an open source library for machine learning research. It
contains several utilities for Machine Learning research and algorithm
implementations. Specifically, it contains ``scikit-learn`` compatible
implementations for Active Learning and synthetic data generation, as well as
datasets and various utilities to assist in experiment design and results
reporting.

The LaTeX and Python code for generating the paper, experiments' results and
visualizations shown in each paper is available (whenever possible) in the
[publications repo](https://github.com/joaopfonseca/publications).

## Installation

A Python distribution of version >= 3.9 is required to run this
project. Earlier Python versions might work in most cases, but they are not
tested. ``ML-Research`` requires:

- numpy (>= 1.20.0)
- pandas (>= 1.3.5)
- sklearn (>= 1.2.0)
- imblearn (>= 0.8.0)

Some functions in the ``mlresearch.utils`` submodule (the ones in the script
``_visualization.py``) require Matplotlib (>= 2.2.3). The
``mlresearch.datasets`` submodule and ``mlresearch.utils.parallel_apply``
function require tqdm (>= 4.46.0) to display progress bars.

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
    # "all" will also install the dependency groups shown below.
    pip install .[optional,tests,docs] 

## Citing ML-Research

If you use ML-Research in a scientific publication, we would appreciate
citations to the following paper:

    @article{fonseca2023improving,
      title={Improving Active Learning Performance through the Use of Data Augmentation},
      author={Fonseca, Joao and Bacao, Fernando and others},
      journal={International Journal of Intelligent Systems},
      volume={2023},
      year={2023},
      publisher={Hindawi}
    }
