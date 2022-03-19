import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from ._lars import LARS
from ._components import SiameseArm, LinearWarmupCosineAnnealingLR


class SimSiam(LightningModule):
    """
    PyTorch Lightning implementation of Exploring Simple Siamese Representation
    Learning (SimSiam_)

    Paper authors: Xinlei Chen, Kaiming He.

    .. warning:: Work in progress.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet
    Example::
        model = SimSiam()
        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        trainer = Trainer()
        trainer.fit(model, datamodule=dm)
    Train::
        trainer = Trainer()
        trainer.fit(model)

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf

    Args:
        datamodule: The datamodule
        learning_rate: the learning rate
        weight_decay: optimizer weight decay
        input_height: image input height
        batch_size: the batch size
        num_workers: number of workers
        warmup_epochs: num of epochs for scheduler warm up
        max_epochs: max epochs for scheduler
    """

    def __init__(
        self,
        base_encoder: str = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_size: int = 512,
        learning_rate: float = 1e-3,
        start_lr: float = 0.0,
        final_lr: float = 0.0,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base_encoder = base_encoder

        self.encoder_out_dim = encoder_out_dim
        self.projector_hidden_size = projector_hidden_size

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay

        self.start_lr = start_lr
        self.final_lr = final_lr

        # training params, checked
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.online_network = SiameseArm(
            encoder=self.base_encoder,
            encoder_out_dim=self.encoder_out_dim,
            projector_hidden_size=self.projector_hidden_size,
            projector_out_dim=self.encoder_out_dim,
        )

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def training_step(self, batch, batch_idx):
        (img_1, img_2), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        self.log_dict({"val_loss": loss})

        return loss

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=("bias", "bn")
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.hparams.learning_rate, weight_decay=self.weight_decay
            )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )

        return [optimizer], [scheduler]
