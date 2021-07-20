.. _api_description:

===
API
===

This is the full API documentation of the `research` package.

:mod:`research.active_learning`
-------------------------------

.. automodule:: research.active_learning
    :no-members:
    :no-inherited-members:

.. currentmodule:: research

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    active_learning.ALSimulation

:mod:`research.data_augmentation`
---------------------------------

.. automodule:: research.data_augmentation
    :no-members:
    :no-inherited-members:

.. currentmodule:: research

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    data_augmentation.GeometricSMOTE
    data_augmentation.OverSamplingAugmentation

:mod:`research.datasets`
------------------------

.. automodule:: research.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: research

.. autosummary::
    :toctree: _generated/
    :template: class.rst

    datasets.Datasets
    datasets.BinaryDatasets
    datasets.ImbalancedBinaryDatasets
    datasets.ContinuousCategoricalDatasets
    datasets.MulticlassDatasets
    datasets.RemoteSensingDatasets

:mod:`research.metrics`
-------------------------------

.. automodule:: research.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: research

.. autosummary::
    :toctree: _generated/
    :template: function.rst

    metrics.geometric_mean_score_macro
    metrics.area_under_learning_curve
    metrics.data_utilization_rate

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    metrics.ALScorer

:mod:`research.utils`
-------------------------------

.. automodule:: research.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: research

.. autosummary::
    :toctree: _generated/
    :template: function.rst

    utils.generate_mean_std_tbl
    utils.generate_pvalues_tbl
    utils.sort_tbl
    utils.generate_paths
    utils.make_bold
    utils.generate_mean_std_tbl_bold
    utils.img_array_to_pandas
    utils.load_datasets
    utils.check_pipelines
    utils.check_pipelines_wrapper
    utils.load_plt_sns_configs
    utils.val_to_color
