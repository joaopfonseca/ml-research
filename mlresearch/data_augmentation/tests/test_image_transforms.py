import pytest
import torch
import numpy as np
from PIL import Image
from .._image_transforms import ImageTransform

base_img = (np.random.RandomState(42).rand(32, 32, 3) * 255).astype("uint8")

IMG_FORMATS = {
    "numpy": base_img.copy().astype("uint8"),
    "PIL.Image": Image.fromarray(base_img).convert("RGB"),
    "torch.Tensor": torch.from_numpy(np.moveaxis(base_img, -1, 0)),
}


@pytest.mark.parametrize(
    "dtype",
    IMG_FORMATS.keys(),
)
def test_image_transforms(dtype):
    img = IMG_FORMATS[dtype]

    T = ImageTransform(0, 0, 0, 0, 3, 0)
    assert T(img).size(1) == 32
