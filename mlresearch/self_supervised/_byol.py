"""
Work in progress.

Adapted from the PyTorch-Lightning Bolts implementation.
"""

import warnings
from typing import Sequence, Union, Any, List
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torchvision import models
from pytorch_lightning import LightningModule, Trainer, Callback
from ._lars import LARS


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


class BYOLArm(nn.Module):
    """
    Implementation of the online and target networks for BYOL.
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

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - np.cos(np.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


class BYOLMAWeightUpdate(Callback):
    """
    Weight update rule from BYOL.

    Your model should have:
        - ``self.online_network``
        - ``self.target_network``

    Updates the target_network params using an exponential moving average update rule
    weighted by tau. BYOL claims this keeps the online_network from collapsing.

    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training
              step

    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        k = pl_module.global_step
        tau = 1 - (1 - self.initial_tau) * (np.cos(np.pi * k / max_steps) + 1) / 2
        return tau

    def update_weights(
        self,
        online_net: Union[nn.Module, torch.Tensor],
        target_net: Union[nn.Module, torch.Tensor],
    ) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(),
            target_net.named_parameters(),
        ):
            target_p.data = (
                self.current_tau * target_p.data
                + (1 - self.current_tau) * online_p.data
            )


class BYOL(LightningModule):
    """
    Pytorch implementation of Bootstrap Your Own Latent (BYOL).
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        exclude_bias_from_adaption: bool = True,  # Not implemented yet
        input_height: int = 32,  # This is not in the paper's hyperparameters list
        batch_size: int = 32,  # This is not the same value as in the paper
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        base_encoder: Union[str, torch.nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_size: int = 4096,
        projector_out_dim: int = 256,
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
            base_encoder: the base encoder module or resnet name
            encoder_out_dim: output dimension of base_encoder
            projector_hidden_size: hidden layer size of projector MLP
            projector_out_dim: output size of projector MLP
        """
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")

        # TODO: Check where data augmentation is being done (and how).

        self.online_network = BYOLArm(
            encoder=base_encoder,
            encoder_out_dim=encoder_out_dim,
            projector_hidden_size=projector_hidden_size,
            projector_out_dim=projector_out_dim,
        )
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:

        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx
        )

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        imgs, y = batch
        img_1, img_2 = imgs[:2]

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = 2 - 2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = 2 - 2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict(
            {"1_2_loss": loss_a, "2_1_loss": loss_b, "train_loss": total_loss}
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({"1_2_loss": loss_a, "2_1_loss": loss_b, "val_loss": total_loss})

        return total_loss

    def configure_optimizers(self):
        optimizer = LARS(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            exclude_bias_from_adaption=self.hparams.exclude_bias_from_adaption,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]
