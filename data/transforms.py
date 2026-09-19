"""Orientation-safe augmentation for the five-class task in paper Section 2.1.

The four colony labels depend on absolute lamellar angle. Flips and rotations
would require verified label permutations, which the paper does not supply.
This module therefore implements translations via paired cropping and RGB
brightness/contrast changes only. Geometry is shared by image, mask, and prior.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def spatial_size(value, name="image_size"):
    """Validate a (height, width) pair without silently truncating floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a (height, width) pair")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in value):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(v) for v in value)


def validate_triplet(image, mask, prior):
    """Reject shape/range errors before data enter a segmentation model."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an H×W×3 RGB numpy array")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be nonempty")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("image must use floating-point values in [0, 1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must contain finite values in [0, 1]")
    if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
        raise ValueError("image and mask spatial dimensions must match")
    if not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0 or mask.max() > 4:
        raise ValueError("mask must contain integer class IDs in [0, 4]")
    if not isinstance(prior, np.ndarray) or prior.shape != mask.shape:
        raise ValueError("image, mask, and prior spatial dimensions must match")
    if not np.isin(prior, (0, 1)).all():
        raise ValueError("prior must contain only binary values 0 and 1")


@dataclass(frozen=True)
class PairedAugment:
    """Random crop plus photometry, with no orientation-changing operations.

    ``crop_size`` is the output (height, width); ``None`` leaves geometry intact.
    ``brightness`` and ``contrast`` specify a nonnegative jitter magnitude: a
    magnitude of 0.1 draws a multiplicative factor uniformly from [0.9, 1.1].
    The caller provides ``rng`` so dataset seed/index/epoch control randomness.
    Prior is a geometric auxiliary mask; photometric jitter does not alter it.
    """

    crop_size: tuple[int, int] | None = None
    brightness: float = 0.1
    contrast: float = 0.1

    def __post_init__(self):
        if self.crop_size is not None:
            object.__setattr__(self, "crop_size", spatial_size(self.crop_size, "crop_size"))
        for name in ("brightness", "contrast"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1]")

    def __call__(self, image, mask, prior, *, rng):
        validate_triplet(image, mask, prior)
        image, mask, prior = image.copy(), mask.copy(), prior.copy()
        if self.crop_size is not None:
            height, width = self.crop_size
            if height > image.shape[0] or width > image.shape[1]:
                raise ValueError("crop_size cannot exceed the input image dimensions")
            top = int(rng.integers(0, image.shape[0] - height + 1))
            left = int(rng.integers(0, image.shape[1] - width + 1))
            region = np.s_[top:top + height, left:left + width]
            image, mask, prior = image[region], mask[region], prior[region]
        if self.contrast:
            factor = float(rng.uniform(1 - self.contrast, 1 + self.contrast))
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = mean + factor * (image - mean)
        if self.brightness:
            image = image * float(rng.uniform(1 - self.brightness, 1 + self.brightness))
        return np.clip(image, 0, 1).astype(np.float32), mask.copy(), prior.copy()


def denormalize_image(image):
    """Undo ImageNet normalization on a CHW or NCHW torch tensor for display."""
    if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError("image must be a CHW or NCHW tensor with three channels")
    shape = (3, 1, 1) if image.ndim == 3 else (1, 3, 1, 1)
    mean = image.new_tensor(IMAGENET_MEAN).view(shape)
    std = image.new_tensor(IMAGENET_STD).view(shape)
    return image * std + mean
