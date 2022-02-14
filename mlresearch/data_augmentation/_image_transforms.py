from typing import Sequence
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image


class ImageConvert:
    """Convert a tensor or an ndarray to PIL Image. This transform does not support
    torchscript. If it is already a PIL Image, return itself. This class is equivalent
    to transforms.ToPILImage() but bypasses the conversion if the passed image is already
    a PIL image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the
            input data:
            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (
            i.e ``int``, ``float``, ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        if type(pic) == Image.Image:
            return pic
        else:
            return F.to_pil_image(pic, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={0}".format(self.mode)
        format_string += ")"
        return format_string


class ImageTransform:
    """
    Class that applies basic transformations to image data.
    If you want to do other augmentations, you can just re-write this class.

    Args:
        brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 +
            brightness].
        contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
        saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 +
            saturation].
        hue (float): sampled uniformly in [-hue, hue].
        kernel_size (int): Size of the Gaussian kernel (to apply Gaussian blur)
        solarize_threshold (float): all pixels equal or above this value are inverted.
        color_jitter_prob (float, optional): probability of applying color jitter.
            Defaults to 0.8.
        gray_scale_prob (float, optional): probability of converting to gray scale.
            Defaults to 0.2.
        horizontal_flip_prob (float, optional): probability of flipping horizontally.
            Defaults to 0.5.
        gaussian_prob (float, optional): probability of applying gaussian blur.
            Defaults to 0.0.
        solarize_prob (float, optional): probability of applying solarization.
            Defaults to 0.0.
        min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        crop_size (int, optional): size of the crop. Defaults to 32.
        mean (Sequence[float], optional): mean values for normalization.
            Defaults to (0.485, 0.456, 0.406).
        std (Sequence[float], optional): std values for normalization.
            Defaults to (0.228, 0.224, 0.225).
    """

    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        kernel_size: int,
        solarize_threshold: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarize_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 32,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):

        super().__init__()
        self.transform = transforms.Compose(
            [
                ImageConvert(),
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size)], p=gaussian_prob
                ),
                transforms.RandomSolarize(solarize_threshold, p=solarize_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)
