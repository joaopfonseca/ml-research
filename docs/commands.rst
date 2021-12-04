.. _commands:

Commands
========

The Makefile contains commands for common tasks related the setup and
development of experiments. These commands can be used by running ``make
<command>`` in the root directory of the project.

======================================  =========================================================
 `make` command                          Description
======================================  =========================================================
``clean``                               Delete all compiled Python files
``code-analysis``                       Lint using flake8
``code-format``                         Format code using Black
``environment``                         Set up python interpreter environment
``install-update``                      Install and Update Python Dependencies + ML-Research
``test``                                Run test suite and coverage
``upload-pypi``                         Upload new package version to pypi
======================================  =========================================================
