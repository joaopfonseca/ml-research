import warnings
from typing import List
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import models


class MLP(nn.Module):
    """
    Multilayer perceptron implementation used as projector and predictor.
    """

    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    """
    Implementation of the online and target networks for BYOL and SimSiam.
    """

    def __init__(
        self,
        encoder="resnet50",
        encoder_out_dim=2048,
        projector_hidden_size=4096,
        projector_out_dim=256,
    ):
        super().__init__()

        if type(encoder) == str:
            encoder = getattr(models, encoder)(pretrained=True, progress=True)
            encoder.fc = (
                nn.Identity()
                if encoder_out_dim == 2048
                else nn.Sequential(
                    nn.Linear(2048, encoder_out_dim, bias=False),
                    nn.BatchNorm1d(encoder_out_dim),
                    nn.ReLU(inplace=True),
                )
            )

        self.encoder = encoder
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        self.predictor = MLP(
            projector_out_dim, projector_hidden_size, projector_out_dim
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


def linear_warmup_decay(
    step,
    warmup_steps,
    total_steps,
    start_lr=0,
    end_lr=0,
    decay="cosine",
):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay
    to 0 at total_steps.

    decay: "cosine", "linear", None
    """

    if step < warmup_steps:
        x = float(step) / float(warmup_steps)
        step_ratio = 1 / warmup_steps
        m = (1 - start_lr) / (1 - step_ratio)
        return m * (x - step_ratio) + start_lr

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    if decay == "linear":
        base = 1.0 - progress
        return base * (1 - end_lr) + end_lr
    elif decay == "cosine":
        base = 0.5 * (1.0 + np.cos(np.pi * progress))
        return base * (1 - end_lr) + end_lr
    else:
        return 1.0


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for
        :class:`LinearWarmupCosineAnnealingLR` after each iteration as calling it after
        each epoch will keep the starting lr at warmup_start_lr for the first epoch
        which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an
        EPOCH_DEPRECATION_WARNING. It calls the :func:`_get_closed_form_lr()` method for
        this scheduler instead of :func:`get_lr()`. Though this does not change the
        behavior of the scheduler, when passing epoch param to :func:`.step()`, the user
        should call the :func:`.step()` function before calling train and validation
        methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer, warmup_epochs=10, max_epochs=40
        ... )
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup.
                Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr_multiplier(self, step):
        return [
            linear_warmup_decay(
                step=step,
                warmup_steps=self.warmup_epochs,
                total_steps=self.max_epochs,
                start_lr=self.warmup_start_lr / lr,
                end_lr=self.eta_min / lr,
                decay="cosine",
            )
            for lr in self.base_lrs
        ]

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [
            lr_multiplier * group["initial_lr"]
            for lr_multiplier, group in zip(
                self.get_lr_multiplier(self._step_count - 1),
                self.optimizer.param_groups,
            )
        ]
