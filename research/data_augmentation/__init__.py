"""
Contains the implementation of variations of oversampling/data augmentation
algorithms, as well as helper classes to use oversampling algorithms as data
augmentation techniques.
"""

from ._oversampling_augmentation import OverSamplingAugmentation
from ._gsmote import GeometricSMOTE

__all__ = ["OverSamplingAugmentation", "GeometricSMOTE"]
