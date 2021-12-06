|CircleCI| |Codecov| |Documentation Status| |Pypi Version| |Black| |Python
Versions| |DOI|

ML-Research
===========

This repository contains the code developed for all the publications I
was involved in. The LaTeX and Python code for generating the paper,
experiments' results and visualizations reported in each paper is
available (whenever possible) in the paper's directory.

Additionally, contributions at the algorithm level are available in the
package ``research``.

Project Organization
--------------------

::

    ├── LICENSE
    │
    ├── Makefile           <- Makefile with basic commands
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── publications       <- Research papers published, submitted, or under development.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── research           <- Source code used in publications.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Installation and Setup
----------------------

A Python distribution of version 3.7 or higher is required to run this
project.

User Installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy, the easiest way
to install scikit-learn is using ``pip`` ::

    pip install -U ml-research

The documentation includes more detailed `installation instructions
<https://mlresearch.readthedocs.io/en/latest/getting-started.html>`_.

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

The following commands should allow you to setup the development version of the
project with minimal effort:

::

    # Clone the project.
    git clone https://github.com/joaopfonseca/research.git
    cd research

    # Create and activate an environment 
    make environment 
    conda activate research # Adapt this line accordingly if you're not running conda

    # Install project requirements and the research package
    make requirements

Citing ML-Research
------------------

If you use ML-Research in a scientific publication, we would appreciate
citations to the following paper::


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


.. |Python Versions| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue

.. |Documentation Status| image:: https://readthedocs.org/projects/mlresearch/badge/?version=latest
   :target: https://mlresearch.readthedocs.io/en/latest/?badge=latest

.. |Pypi Version| image:: https://badge.fury.io/py/ml-research.svg
   :target: https://badge.fury.io/py/ml-research

.. |DOI| image:: https://zenodo.org/badge/DOI/10.3390/RS13132619.svg
   :target: https://doi.org/10.3390/RS13132619

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |CircleCI| image:: https://circleci.com/gh/joaopfonseca/ml-research/tree/master.svg?style=shield
    :target: https://circleci.com/gh/joaopfonseca/ml-research/tree/master

.. |Codecov| image:: https://codecov.io/gh/joaopfonseca/ml-research/branch/master/graph/badge.svg?token=J2EBA4YTMN
      :target: https://codecov.io/gh/joaopfonseca/ml-research
    
