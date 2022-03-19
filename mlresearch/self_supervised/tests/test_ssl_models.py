import pytest
import multiprocessing
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
from mlresearch.data_augmentation import ImageTransform
from .._byol import BYOL
from .._simsiam import SimSiam

pl.seed_everything(42, workers=True)

MODELS = {"BYOL": BYOL, "SimSiam": SimSiam}
CPUS = multiprocessing.cpu_count()

DATA_KWARGS = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "kernel_size": 3,
    "solarize_threshold": 0.9,
    "gaussian_prob": 0.5,
    "solarize_prob": 0.5,
}

BASE_KWARGS = {
    "num_classes": 10,
    "no_labels": False,
    "max_epochs": 2,
    "lars": True,
    "lr": 0.01,
    "grad_clip_lars": True,
    "weight_decay": 0.00001,
    "classifier_lr": 0.5,
    "exclude_bias_n_norm": True,
    "accumulate_grad_batches": 1,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "warmup_cosine",
    "min_lr": 0.0,
    "warmup_start_lr": 0.0,
    "warmup_epochs": 10,
    "num_crops_per_aug": [2, 0],  # large crops, small crops
    "num_large_crops": 2,
    "num_small_crops": 0,
    "eta_lars": 0.02,
    "lr_decay_steps": None,
    "batch_size": 32,
    "num_workers": 4,
    "base_tau_momentum": 0.99,
    "final_tau_momentum": 1.0,
}


class FakeCIFAR(Dataset):
    def __init__(
        self, size=512, num_classes=10, img_shape=(32, 32), img_transforms=None
    ):
        self.img_transforms = img_transforms

        imgs = torch.rand(size, 3, *img_shape)
        imgs = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(
            imgs
        )

        self.X = imgs
        self.y = torch.randint(low=0, high=num_classes, size=(size,))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.img_transforms is None:
            return self.X[idx], self.y[idx]
        else:
            return self.img_transforms(self.X[idx]), self.y[idx]


class SSLImageTransform(ImageTransform):
    def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


########################################################################################


@pytest.mark.parametrize("name", MODELS.keys())
def test_models(name):

    data = FakeCIFAR(size=64)
    img_transforms = SSLImageTransform(**DATA_KWARGS)
    model = MODELS[name]()

    # test forward
    out = model(data[: BASE_KWARGS["batch_size"]][0])
    assert isinstance(out, torch.Tensor)
    assert out.size() == (
        BASE_KWARGS["batch_size"],
        model.hparams["encoder_out_dim"],
    )

    # test fitting
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    data.img_transforms = img_transforms

    train_dl = DataLoader(
        data, batch_size=BASE_KWARGS["batch_size"], shuffle=True, num_workers=CPUS
    )

    val_dl = DataLoader(data, batch_size=BASE_KWARGS["batch_size"], num_workers=CPUS)
    trainer.fit(model, train_dl, val_dl)
