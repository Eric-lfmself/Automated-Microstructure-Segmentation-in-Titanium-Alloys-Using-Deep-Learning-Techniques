"""Five-class TC4 datasets following paper Sections 2.1 and 2.2.

Class IDs are zero-based for PyTorch: 0 = equiaxed alpha, 1–4 = colony
orientations 1–4. Beta is not a separate semantic class. The binary Otsu prior
is returned separately; it is not appended as a fourth input channel.
"""

import csv
import math
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from preprocessing import binarize_otsu, enhance_iwmid, to_grayscale

from .transforms import IMAGENET_MEAN, IMAGENET_STD, spatial_size, validate_triplet

_CONTENT_SALT = 7  # independent deterministic sample-content stream
_AUGMENT_SALT = 19  # independent deterministic augmentation stream

CLASS_NAMES = (
    "Equiaxed alpha", "Colony orientation 1", "Colony orientation 2",
    "Colony orientation 3", "Colony orientation 4",
)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _resize_pair(image, mask, size):
    """Center-crop to target aspect ratio, then resize without changing angles.

    Source crop dimensions are integer multiples of the target aspect ratio,
    guaranteeing identical horizontal and vertical scale factors. Image uses
    bilinear sampling, class mask nearest-neighbor. This can discard margins;
    use ``image_size=None`` to preserve the original full image.
    """
    if size is None or image.shape[:2] == size:
        return image.astype(np.float32) / 255.0, mask.astype(np.int64)
    out_h, out_w = size
    divisor = math.gcd(out_h, out_w)
    unit_h, unit_w = out_h // divisor, out_w // divisor
    factor = min(image.shape[0] // unit_h, image.shape[1] // unit_w)
    if factor < 1:
        raise ValueError("source image is too small for an exact orientation-preserving target aspect ratio")
    crop_h, crop_w = factor * unit_h, factor * unit_w
    top = (image.shape[0] - crop_h) // 2
    left = (image.shape[1] - crop_w) // 2
    image = image[top:top + crop_h, left:left + crop_w]
    mask = mask[top:top + crop_h, left:left + crop_w]
    image = np.asarray(Image.fromarray(image).resize((out_w, out_h), Image.Resampling.BILINEAR))
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((out_w, out_h), Image.Resampling.NEAREST))
    return image.astype(np.float32) / 255.0, mask.astype(np.int64)


class _TC4Base(Dataset):
    """Shared local-only preprocessing and tensor contract for both datasets."""

    def _configure(self, image_size, seed, transform, preprocess, sigmas, weights, normalize):
        self.image_size = None if image_size is None else spatial_size(image_size)
        self.seed = _nonnegative_integer(seed, "seed")
        self.epoch = 0
        if transform is not None and not callable(transform):
            raise ValueError("transform must be callable or None")
        self.transform = transform
        self.preprocess = bool(preprocess)
        self.sigmas = tuple(sigmas)
        self.weights = tuple(weights)
        self.normalize = bool(normalize)

    def set_epoch(self, epoch):
        """Change deterministic augmentation randomness between training epochs.

        Synthetic image content remains fixed; only augmentation changes.
        Call before iterating a new DataLoader (persistent workers retain copies).
        """
        self.epoch = _nonnegative_integer(epoch, "epoch")

    def _index(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} outside dataset of length {len(self)}")
        return index

    def _finish(self, image, mask, identifier, index):
        image, mask = _resize_pair(image, mask, self.image_size)
        if self.preprocess:
            image = enhance_iwmid(image, sigmas=self.sigmas, weights=self.weights)
        prior = binarize_otsu(to_grayscale(image)).astype(np.uint8)
        validate_triplet(image, mask, prior)
        if self.transform is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, _AUGMENT_SALT]))
            result = self.transform(image, mask, prior, rng=rng)
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise ValueError("transform must return (image, mask, prior)")
            image, mask, prior = result
            validate_triplet(image, mask, prior)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(dtype=torch.float32)
        if self.normalize:
            mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return {
            "image": tensor,
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            "prior": torch.from_numpy(np.ascontiguousarray(prior[None], dtype=np.float32)),
            "id": str(identifier),
        }


class DummyTC4Dataset(_TC4Base):
    """Deterministic synthetic microstructure-shaped fixtures, never real data.

    Large central equiaxed grains overlay four colony regions whose lamellae
    have illustrative orientations 22.5°, 67.5°, 112.5°, and 157.5°. These are
    fixture settings independent of the annotation boundaries in our dataset.
    RGB content is fixed by (seed, index); masks contain all five classes at
    the supported image sizes before optional cropping. No network access.
    """

    def __init__(self, length=4, image_size=(128, 128), seed=42, transform=None,
                 preprocess=True, sigmas=(1, 2, 3, 4), weights=(.5, .5, .5, .5),
                 normalize=True):
        self.length = _nonnegative_integer(length, "length")
        if self.length < 1:
            raise ValueError("length must be positive")
        self._configure(image_size, seed, transform, preprocess, sigmas, weights, normalize)
        if self.image_size is None or min(self.image_size) < 16:
            raise ValueError("dummy image_size must have height and width at least 16")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self._index(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, _CONTENT_SALT]))
        height, width = self.image_size
        y, x = np.mgrid[:height, :width]
        mask = 1 + (x >= width // 2).astype(np.int64) + 2 * (y >= height // 2)
        intensity = np.empty((height, width), dtype=np.float64)
        for label, angle in enumerate((22.5, 67.5, 112.5, 157.5), start=1):
            theta = np.deg2rad(angle)
            wavelength = max(3.0, min(height, width) / 13.0) * rng.uniform(.85, 1.15)
            phase = (-x * np.sin(theta) + y * np.cos(theta)) / wavelength
            lamellae = .5 + .5 * np.sin(2 * np.pi * phase + rng.uniform(0, 2 * np.pi))
            region = mask == label
            intensity[region] = (.27 + .39 * lamellae)[region]
        center_x = width * rng.uniform(.46, .54)
        center_y = height * rng.uniform(.46, .54)
        radius = min(height, width) * .23
        equiaxed = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        # A smaller off-center grain makes the fixture less symmetric.
        equiaxed |= ((x - .22 * width) ** 2 + (y - .27 * height) ** 2) <= (.09 * min(height, width)) ** 2
        mask[equiaxed] = 0
        intensity[equiaxed] = .82
        intensity += rng.normal(0, .025, size=(height, width))
        rgb = intensity[..., None] * np.array([1.02, 1.0, .97])
        rgb += rng.normal(0, .004, size=rgb.shape)
        image = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return self._finish(image, mask, f"dummy_{index:04d}", index)


class TC4Dataset(_TC4Base):
    """Read the private dataset from an explicit local CSV; never download it.

    Required columns: ``id,image,mask``. Relative paths resolve relative to the
    CSV directory. Image files are converted to RGB. Masks must be single-band
    integer images (including palette PNGs), with values 0–4, or 1–5 when
    ``label_offset=1``. There is no ignore/background class. Image/mask sizes
    must match *before* cropping/resizing; no automatic mismatch repair occurs.

    Matching dimensions cannot establish semantic alignment: the user must
    supply genuinely corresponding images and annotations. Duplicate IDs,
    duplicate file pairs, reused images/masks, and image-as-mask are rejected.
    """

    def __init__(self, manifest, image_size=(1024, 1024), seed=42, transform=None,
                 preprocess=True, sigmas=(1, 2, 3, 4), weights=(.5, .5, .5, .5),
                 normalize=True, label_offset=0):
        self._configure(image_size, seed, transform, preprocess, sigmas, weights, normalize)
        if isinstance(label_offset, bool) or not isinstance(label_offset, Integral) or label_offset not in (0, 1):
            raise ValueError("label_offset must be 0 (labels 0–4) or 1 (labels 1–5)")
        self.label_offset = int(label_offset)
        self.manifest = Path(manifest).expanduser().resolve()
        self.records = []
        seen_ids, seen_images, seen_masks = set(), set(), set()
        with self.manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"id", "image", "mask"}.issubset(reader.fieldnames):
                raise ValueError("manifest must have id,image,mask columns")
            for row_number, row in enumerate(reader, start=2):
                if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "image", "mask")):
                    raise ValueError(f"manifest row {row_number} has a missing id/image/mask")
                identifier = row["id"].strip()
                image_path = self._local_path(row["image"].strip())
                mask_path = self._local_path(row["mask"].strip())
                if identifier in seen_ids or image_path in seen_images or mask_path in seen_masks:
                    raise ValueError(f"manifest row {row_number} duplicates an ID, image, or mask")
                if image_path == mask_path:
                    raise ValueError("image and mask must refer to different files")
                seen_ids.add(identifier)
                seen_images.add(image_path)
                seen_masks.add(mask_path)
                self.records.append((identifier, image_path, mask_path))
        if not self.records:
            raise ValueError("manifest contains no samples")

    def _local_path(self, value):
        if "://" in value:
            raise ValueError("manifest paths must be local files, not URLs")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.manifest.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest file does not exist: {path}")
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        index = self._index(index)
        identifier, image_path, mask_path = self.records[index]
        with Image.open(image_path) as source:
            if source.mode not in ("RGB", "RGBA", "L", "P", "1"):
                raise ValueError(f"{identifier}: unsupported image mode {source.mode}; convert high-bit-depth images to calibrated 8-bit RGB explicitly")
            image = np.array(source.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as source:
            raw_mask = np.array(source)
        if raw_mask.ndim != 2 or not np.issubdtype(raw_mask.dtype, np.integer):
            raise ValueError(f"{identifier}: mask must be a single-band integer image")
        if image.shape[:2] != raw_mask.shape:
            raise ValueError(f"{identifier}: image/mask dimensions differ before resizing")
        mask = raw_mask.astype(np.int64) - self.label_offset
        if mask.size == 0 or mask.min() < 0 or mask.max() > 4:
            raise ValueError(f"{identifier}: mask class IDs must be {self.label_offset}–{self.label_offset + 4}")
        return self._finish(image, mask, identifier, index)
