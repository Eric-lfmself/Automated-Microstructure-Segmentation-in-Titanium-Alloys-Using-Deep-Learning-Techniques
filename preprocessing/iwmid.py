"""IWMID enhancement and Otsu thresholding: our paper Section 2.2, Eqs. (1)-(5).

Our configurable release defaults are sigmas (1, 2, 3, 4) pixels and weights
(0.5, 0.5, 0.5, 0.5), reflect boundaries, Gaussian truncation at four sigmas,
BT.601 grayscale, and 256 Otsu bins across the observed intensity range.

Inputs are HxW grayscale or HxWx3 RGB NumPy arrays/PIL images. uint8 and uint16
arrays of either byte order are scaled by their dtype maximum; floating inputs
must lie in [0, 1]. Filtering and subtraction use float32 independently for each
RGB channel. We evaluate the equations without intermediate clipping and clip
only the final enhanced image. iwmid_components preserves Eq. (4) overshoot.
The binary output is an auxiliary prior, separate from semantic labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


class IWMIDComponents(TypedDict):
    """Intermediate arrays, with all signed Eq. (2)–(4) values preserved."""

    input: np.ndarray
    blurred: tuple[np.ndarray, ...]
    differences: tuple[np.ndarray, ...]
    enhancement_difference: np.ndarray
    enhanced_unclipped: np.ndarray


def _as_float_image(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image) and image.mode not in {"L", "RGB", "I;16", "F"}:
        raise ValueError("PIL images must use L, RGB, I;16, or F mode; convert explicitly first")
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError("image must have shape (H, W) or (H, W, 3)")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("image spatial dimensions must be nonempty")
    if np.issubdtype(array.dtype, np.uint8) or np.issubdtype(array.dtype, np.uint16):
        result = array.astype(np.float32) / np.iinfo(array.dtype).max
    elif np.issubdtype(array.dtype, np.floating):
        # Validate BEFORE conversion: tiny float64 overshoots must not round into range.
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError("floating image intensities must be finite and in [0, 1]")
        result = array.astype(np.float32, copy=True)
    else:
        raise TypeError("image dtype must be uint8, uint16, or floating point in [0, 1]")
    return result


def _four_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        parsed = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain exactly four finite numbers") from exc
    if parsed.shape != (4,) or not np.isfinite(parsed).all():
        raise ValueError(f"{name} must contain exactly four finite numbers")
    return tuple(float(value) for value in parsed)


def iwmid_components(
    image: np.ndarray | Image.Image,
    sigmas: Sequence[float] = (1, 2, 3, 4),
    weights: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
) -> IWMIDComponents:
    """Evaluate paper Section 2.2 Eqs. (1)–(4) without clipping signed arrays.

    Each ``Bi`` is a Gaussian convolution of the original image, not of ``B(i-1)``.
    Sigmas must be finite, positive, and strictly increasing. Weights must be
    finite and nonnegative; no paper-unsupported upper bound is imposed.
    Large custom weights may overflow float32 and are rejected explicitly.
    """
    scales = _four_values(sigmas, "sigmas")
    coefficients = _four_values(weights, "weights")
    if any(scale <= 0 for scale in scales) or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError("sigmas must be positive and strictly increasing")
    if any(weight < 0 for weight in coefficients):
        raise ValueError("weights must be nonnegative")
    source = _as_float_image(image)
    # Eq. (1); sigma=0 on the last axis forbids mixing RGB channels.
    blurred = tuple(
        gaussian_filter(source, sigma=(sigma, sigma) + ((0,) if source.ndim == 3 else ()),
                        mode="reflect", truncate=4.0)
        for sigma in scales
    )
    # Eq. (2). Source is already floating point: subtraction cannot uint-wrap.
    differences = tuple(previous - following for previous, following in zip((source, *blurred[:-1]), blurred))
    d1, d2, d3, d4 = differences
    w1, w2, w3, w4 = coefficients
    with np.errstate(over="ignore", invalid="ignore"):
        # Eq. (3), including its asymmetric sign-dependent D1 coefficient.
        enhancement = ((1 - w1 * np.sign(d1)) * d1 + w2 * d2 + w3 * d3 + w4 * d4).astype(np.float32)
        enhanced = source + enhancement  # Eq. (4)
    if not np.isfinite(enhancement).all() or not np.isfinite(enhanced).all():
        raise ValueError("weights produce nonfinite float32 enhancement values")
    return {"input": source, "blurred": blurred, "differences": differences,
            "enhancement_difference": enhancement, "enhanced_unclipped": enhanced}


def enhance_iwmid(
    image: np.ndarray | Image.Image,
    sigmas: Sequence[float] = (1, 2, 3, 4),
    weights: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
) -> np.ndarray:
    """Return float32 [0, 1] enhanced image, Section 2.2 Eqs. (1)–(4).

    The input shape is preserved and the input is never mutated. Final clipping
    is a documented engineering choice; use ``iwmid_components`` for raw Eq. (4).
    """
    return np.clip(iwmid_components(image, sigmas, weights)["enhanced_unclipped"], 0, 1).astype(np.float32)


def to_grayscale(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert RGB to float32 grayscale before Eq. (5); we use BT.601 coefficients."""
    image_float = _as_float_image(image)
    if image_float.ndim == 2:
        return image_float
    gray = image_float @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(gray, 0, 1).astype(np.float32)


def otsu_threshold(gray_image: np.ndarray | Image.Image) -> float:
    """Maximize Eq. (5) between-class variance using a 256-bin histogram.

    Input must be H×W grayscale. Ties select the first (lowest) bin center.
    A constant image has no valid two-class split and returns its sole intensity.
    """
    gray = _as_float_image(gray_image)
    if gray.ndim != 2:
        raise ValueError("Otsu input must be grayscale (H, W); use to_grayscale first")
    low, high = float(gray.min()), float(gray.max())
    if low == high:
        return low
    histogram, edges = np.histogram(gray.astype(np.float64), bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    # Integer counts keep an empty right class exactly empty. Computing its
    # mass as 1-cumsum(probabilities) can invent a tiny nonzero population.
    left_count = np.cumsum(histogram, dtype=np.int64)
    right_count = gray.size - left_count
    valid = (left_count > 0) & (right_count > 0)
    # Shift/scale the gray axis before moments; Otsu's optimum is invariant
    # to affine intensity transforms. This avoids cancellation at low contrast.
    scaled_centers = (centers - low) / (high - low)
    moment = histogram * scaled_centers
    left_moment = np.cumsum(moment, dtype=np.float64)
    right_moment = np.r_[np.cumsum(moment[::-1], dtype=np.float64)[::-1][1:], 0.0]
    mu0 = left_moment[valid] / left_count[valid]
    mu1 = right_moment[valid] / right_count[valid]
    between_variance = np.full(256, -np.inf, dtype=np.float64)
    # Equivalent to Eq. (5), without subtracting nearly equal global moments.
    between_variance[valid] = ((left_count[valid] / gray.size) *
                              (right_count[valid] / gray.size) * (mu0 - mu1) ** 2)
    return float(centers[int(np.argmax(between_variance))])


def binarize_otsu(gray_image: np.ndarray | Image.Image) -> np.ndarray:
    """Return uint8 {0, 1} prior from Section 2.2 Eq. (5), with bright pixels=1.

    The convention is ``gray > threshold``. Constant images consequently return
    all zeros (no evidence for two phases), including a constant white image.
    """
    gray = _as_float_image(gray_image)
    return (gray.astype(np.float64) > otsu_threshold(gray)).astype(np.uint8)
