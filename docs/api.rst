.. _api_description:

===
API
===

This is the full API documentation of the `mlresearch` package.

:mod:`mlresearch.active_learning`
---------------------------------

.. automodule:: mlresearch.active_learning
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    active_learning.StandardAL
    active_learning.AugmentationAL

:mod:`mlresearch.data_augmentation`
-----------------------------------

.. automodule:: mlresearch.data_augmentation
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    data_augmentation.GeometricSMOTE
    data_augmentation.OverSamplingAugmentation

:mod:`mlresearch.preprocessing`
-------------------------------

.. automodule:: mlresearch.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
   :toctree: _generated/
   :template: class.rst

    preprocessing.PipelineEncoder

:mod:`mlresearch.datasets`
--------------------------

.. automodule:: mlresearch.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: class.rst

    datasets.BinaryDatasets
    datasets.ImbalancedBinaryDatasets
    datasets.ContinuousCategoricalDatasets
    datasets.MultiClassDatasets
    datasets.RemoteSensingDatasets

:mod:`mlresearch.metrics`
-------------------------

.. automodule:: mlresearch.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: function.rst

    metrics.get_scorer
    metrics.get_scorer_names
    metrics.geometric_mean_score_macro
    metrics.area_under_learning_curve
    metrics.data_utilization_rate

.. autosummary::
    :toctree: _generated/
    :template: class.rst
    
    metrics.ALScorer

:mod:`mlresearch.latex`
-----------------------

.. automodule:: mlresearch.latex
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: function.rst

    latex.format_table
    latex.make_bold
    latex.make_mean_sem_table
    latex.export_longtable

:mod:`mlresearch.utils`
-----------------------

.. automodule:: mlresearch.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: mlresearch

.. autosummary::
    :toctree: _generated/
    :template: function.rst

    utils.image_to_dataframe
    utils.dataframe_to_image
    utils.load_datasets
    utils.check_pipelines
    utils.check_pipelines_wrapper
    utils.load_plt_sns_configs
    utils.val_to_color
    utils.generate_paths
