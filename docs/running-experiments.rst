Running an experiment
---------------------

The experiments described in most `research papers
<https://github.com/joaopfonseca/research/tree/master/publications>`_ (except
one) are replicable. To do so, make sure to install the requirements file in
the corresponding publication's directory. 

=================
Project structure
=================

Every research project contains, when applicable, the scripts, data, results,
analysis and content directories. The README file should contain the
manuscript's title and abstract.

scripts
=======

It is the entry point every project. To install the required dependencies from
the scripts directory run the command::

    pip install -r requirements.txt

In order to generate the content of the publication in a reproducible format,
various scripts are provided.

**data.py**

Download and save the datasets used for the experiments::

    python data.py

**results.py**

Run the experiments and get the results::

    python results.py

**analysis.py**

Analyze the results of experiments::

    python analysis.py

data
====

It contains the experimental data. They are downloaded and
saved, using the ``data.py`` script.

results
=======

It contains the results of experiments as pickled pandas dataframes. They are
generated, using the ``results.py`` script.

analysis
========

It contains the analysis of experiments' results in various formats. They are
generated, using the ``analysis.py`` script.

content
=======

It contains the LaTex source files of the project.

