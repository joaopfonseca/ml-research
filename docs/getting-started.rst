Basic installation
------------------

If you only intend to use the ``mlresearch`` package, you can simply install it by
running::

    pip install -U ml-research

Prerequisites
=============

A Python distribution of version 3.7 or higher is required to run this
project. It is generally recommended that you create a separate environment to
use this project, which can be done by running ``make environment`` from the
root of this project. The package's dependencies are listed in the
`requirements.txt file
<https://github.com/joaopfonseca/ml-research/blob/master/requirements.txt>`_.
They can be installed using pip or conda::

    pip install -r requirements.txt

If you intend to compile the documentation locally or install development
utilities (e.g., flake8), you must also install the dependencies listed in the
`requirements.dev.txt file
<https://github.com/joaopfonseca/ml-research/blob/master/requirements.dev.txt>`_::

   pip install -r requirements.dev.txt

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

    git clone https://github.com/joaopfonseca/ml-research.git
    cd ml-research
    pip install .

The project's environment and requirements can also be installed using `make` commands (See section
`Commands <commands.html>`_).
