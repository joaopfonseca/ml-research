from sklearn.base import ClassifierMixin
from .base import BaseActiveLearner


class SelfSupervisedAL(BaseActiveLearner, ClassifierMixin):
    """
    Active Learning with self supervision as presented in [1]_. The model is pretrained
    using SimSiam [2]_. They use the pretrained model to extract features, which are used
    to train a SVM or a Logistic Regression.

    Notes
    -----
    See the original paper: [1]_ for more details.

    References
    ----------
    .. [1] Bengar, J. Z., van de Weijer, J., Twardowski, B., & Raducanu, B. (2021).
       Reducing label effort: Self-supervised meets active learning. In Proceedings of
       the IEEE/CVF International Conference on Computer Vision (pp. 1631-1639).

    .. [2] Chen, X., & He, K. (2021). Exploring simple siamese representation learning.
       In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
       Recognition (pp. 15750-15758).
    """


class LADA(BaseActiveLearner, ClassifierMixin):
    """Look-Ahead Data Augmentation for Active Learning implementation."""


class LearningLoss(BaseActiveLearner, ClassifierMixin):
    """Learning Loss implementation."""


class CoreSet(BaseActiveLearner, ClassifierMixin):
    """Coreset Active Learning implementation."""
