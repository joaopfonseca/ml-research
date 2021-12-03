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
``environment``                         Set up python interpreter environment
``code-analysis``                       Lint using flake8
``code-format``                         Format code using Black
``install-update``                      Install and Update Python Dependencies + ML-Research
``test_environment``                    Test python environment is setup correctly
======================================  =========================================================
