from typing import Sequence, Union, Any
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer, Callback
from ._lars import LARS
from ._components import SiameseArm, LinearWarmupCosineAnnealingLR


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

    It is originally trained with 300 epochs on imagenet, when removing the momentum
    encoder and increasing the predictor’s learning rate by 10x.

    The recommended batch sizes are 256 and 512.

    Args:
        datamodule: The datamodule
        learning_rate: the learning rate
        weight_decay: optimizer weight decay
        input_height: image input height
        warmup_epochs: num of epochs for scheduler warm up
        max_epochs: max epochs for scheduler
        base_encoder: the base encoder module or resnet name
        encoder_out_dim: output dimension of base_encoder
        projector_hidden_size: hidden layer size of projector MLP
        projector_out_dim: output size of projector MLP
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        start_lr: float = 0.0,
        final_lr: float = 0.0,
        weight_decay: float = 1.5e-6,
        exclude_bias_from_adaption: bool = True,
        input_height: int = 32,  # This is not in the paper's hyperparameters list, also not implemented
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        base_encoder: Union[str, torch.nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_size: int = 4096,
        projector_out_dim: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.online_network = SiameseArm(
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
