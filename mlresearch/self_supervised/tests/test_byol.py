# Retrieved from solo-learn

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from mlresearch.data_augmentation import ImageTransform
from .._byol import BYOL

pl.seed_everything(42, workers=True)

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


def make_dataset(size=512, num_classes=10, img_shape=(32, 32)):
    """Random image dataset generator"""

    im = np.random.rand(size, 3, *img_shape) * 255
    im = torch.from_numpy(im.astype("uint8"))

    idx = torch.arange(size)
    label = torch.randint(low=0, high=num_classes, size=(size,))

    batch = (idx, im, label)

    return batch


def gen_batch(batch_size, num_classes, size=32):

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")

    T = ImageTransform(**DATA_KWARGS)

    x1, x2 = T(im), T(im)
    x1 = x1.unsqueeze(0).repeat(batch_size, 1, 1, 1).requires_grad_(True)
    x2 = x2.unsqueeze(0).repeat(batch_size, 1, 1, 1).requires_grad_(True)

    idx = torch.arange(batch_size)
    label = torch.randint(low=0, high=num_classes, size=(batch_size,))

    batch = [idx, (x1, x2), label]

    return batch


########################################################################################


def test_byol():

    model = BYOL()

    # test forward
    batch = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"])
    out = model(batch[1][0])
    assert isinstance(out, torch.Tensor) and out.size() == (
        BASE_KWARGS["batch_size"],
        model.hparams["encoder_out_dim"],
    )

    # test fitting
    trainer = pl.Trainer()

    # pl.LightningDataModule
    train_dl = DataLoader(
        make_dataset(size=512, num_classes=10, img_shape=(32, 32)),
        batch_size=BASE_KWARGS["batch_size"],
    )
    val_dl = DataLoader(
        make_dataset(size=512, num_classes=10, img_shape=(32, 32)),
        batch_size=BASE_KWARGS["batch_size"],
    )
    # TODO: finish test module and BYOL implementation
    # trainer.fit(model, train_dl, val_dl)
