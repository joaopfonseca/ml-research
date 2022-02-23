from sklearn.base import ClassifierMixin
from .base import BaseActiveLearner


class SelfSupervisedAL(BaseActiveLearner, ClassifierMixin):
    """Active Learning with self supervision."""


class LADA(BaseActiveLearner, ClassifierMixin):
    """Look-Ahead Data Augmentation for Active Learning implementation."""


class LearningLoss(BaseActiveLearner, ClassifierMixin):
    """Learning Loss implementation."""


class CoreSet(BaseActiveLearner, ClassifierMixin):
    """Coreset Active Learning implementation."""
