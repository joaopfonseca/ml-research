from .base import BaseActiveLearner


class SelfSupervisedAL(BaseActiveLearner):
    """Active Learning with self supervision."""


class LADA(BaseActiveLearner):
    """Look-Ahead Data Augmentation for Active Learning implementation."""


class LearningLoss(BaseActiveLearner):
    """Learning Loss implementation."""


class CoreSet(BaseActiveLearner):
    """Coreset Active Learning implementation."""
