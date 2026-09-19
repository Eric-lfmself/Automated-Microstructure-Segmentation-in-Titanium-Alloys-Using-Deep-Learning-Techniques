"""Numerical, CPU-only checks of paper Section 2.2 Eqs. (1)–(5)."""

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from preprocessing import binarize_otsu, enhance_iwmid, iwmid_components, otsu_threshold, to_grayscale


def test_constant_image_is_fixed_point():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    components = iwmid_components(image)
    for dog in components["differences"]:
        np.testing.assert_array_equal(dog, 0)
    np.testing.assert_array_equal(components["enhancement_difference"], 0)
    np.testing.assert_allclose(enhance_iwmid(image), 128 / 255, atol=1e-7)


def test_step_equations_with_independent_signed_reference():
    image = np.zeros((32, 48), dtype=np.uint8)
    image[:, 24:] = 255
    original = image.copy()
    weights = (0.25, 0.5, 0.75, 1.0)
    components = iwmid_components(image, weights=weights)
    source = image.astype(np.float64) / 255
    blurs = [gaussian_filter(source, sigma=s, mode="reflect", truncate=4) for s in (1, 2, 3, 4)]
    dogs = [source - blurs[0], blurs[0] - blurs[1], blurs[1] - blurs[2], blurs[2] - blurs[3]]
    expected = (1 - weights[0] * np.sign(dogs[0])) * dogs[0]
    expected += sum(weight * dog for weight, dog in zip(weights[1:], dogs[1:]))
    np.testing.assert_allclose(components["enhancement_difference"], expected, atol=1e-7)
    np.testing.assert_allclose(components["enhanced_unclipped"], source + expected, atol=1e-7)
    assert components["differences"][0][:, 23].max() < 0
    assert components["enhanced_unclipped"].min() < 0
    assert components["enhanced_unclipped"].max() > 1
    np.testing.assert_allclose(enhance_iwmid(image, weights=weights), np.clip(source + expected, 0, 1), atol=1e-7)
    np.testing.assert_array_equal(image, original)


def test_sign_weighting_is_asymmetric_and_zero_weights_do_not_disable_d1():
    image = np.zeros((12, 24), dtype=np.float32)
    image[:, 12:] = 1
    parts = iwmid_components(image, weights=(0.5, 0, 0, 0))
    dog = parts["differences"][0]
    np.testing.assert_allclose(parts["enhancement_difference"][dog < 0], 1.5 * dog[dog < 0])
    np.testing.assert_allclose(parts["enhancement_difference"][dog > 0], 0.5 * dog[dog > 0])
    zero_weight_parts = iwmid_components(image, weights=(0, 0, 0, 0))
    np.testing.assert_array_equal(zero_weight_parts["enhancement_difference"], dog)


def test_rgb_channels_are_not_smoothed_together():
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[7:13, 7:13, 0] = 1
    parts = iwmid_components(image)
    for blur in parts["blurred"]:
        np.testing.assert_array_equal(blur[..., 1:], 0)
    assert enhance_iwmid(image).dtype == np.float32


def test_gray_conversion_uint16_and_pil_inputs():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_allclose(to_grayscale(Image.fromarray(rgb)), [[0.299, 0.587, 0.114]], atol=1e-7)
    image16 = np.array([[0, 65535]], dtype=np.uint16)
    np.testing.assert_array_equal(to_grayscale(image16), [[0, 1]])


def test_otsu_separates_two_intensity_populations():
    image = np.full((20, 40), 0.2, dtype=np.float32)
    image[:, 20:] = 0.8
    threshold = otsu_threshold(image)
    assert 0.2 < threshold < 0.8
    expected = np.zeros_like(image, dtype=np.uint8)
    expected[:, 20:] = 1
    actual = binarize_otsu(image)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_otsu_histogram_objective_matches_bruteforce_equation_5():
    image = np.array([[0.0, 0.0, 0.1, 0.2, 0.5, 0.55, 0.8, 1.0]], dtype=np.float32)
    histogram, edges = np.histogram(image, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1].astype(np.float64) + edges[1:].astype(np.float64)) / 2
    mean = np.average(centers, weights=histogram)
    variances = []
    for i in range(256):
        lower, upper = histogram[:i + 1], histogram[i + 1:]
        if lower.sum() == 0 or upper.sum() == 0:
            variances.append(-np.inf)
            continue
        w0, w1 = lower.sum() / image.size, upper.sum() / image.size
        mu0 = np.average(centers[:i + 1], weights=lower)
        mu1 = np.average(centers[i + 1:], weights=upper)
        variances.append(w0 * (mu0 - mean) ** 2 + w1 * (mu1 - mean) ** 2)
    assert otsu_threshold(image) == centers[np.argmax(variances)]


@pytest.mark.parametrize("value", [0, 0.4, 1])
def test_otsu_constant_image_has_documented_empty_bright_class(value):
    image = np.full((8, 8), value, dtype=np.float32)
    assert otsu_threshold(image) == float(image[0, 0])
    np.testing.assert_array_equal(binarize_otsu(image), 0)


@pytest.mark.parametrize("kwargs", [
    {"sigmas": (0, 1, 2, 3)}, {"sigmas": (1, 3, 2, 4)},
    {"sigmas": (1, 2, 2, 4)}, {"sigmas": (1, 2, 3)},
    {"sigmas": (1, 2, 3, np.inf)}, {"weights": (-1, 0, 0, 0)},
    {"weights": (1, 2, 3)}, {"weights": (0, 0, np.nan, 0)},
])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        enhance_iwmid(np.ones((8, 8), dtype=np.float32), **kwargs)


@pytest.mark.parametrize("image,error", [
    (np.zeros((0, 8), dtype=np.float32), ValueError),
    (np.zeros((8, 8, 4), dtype=np.float32), ValueError),
    (np.zeros((8, 8), dtype=np.int64), TypeError),
    (np.full((8, 8), np.nan), ValueError),
    (np.full((8, 8), -0.1), ValueError),
    (np.full((8, 8), 1 + 1e-9), ValueError),
])
def test_invalid_images_are_rejected(image, error):
    with pytest.raises(error):
        enhance_iwmid(image)


def test_otsu_rejects_rgb():
    with pytest.raises(ValueError, match="grayscale"):
        binarize_otsu(np.zeros((8, 8, 3), dtype=np.uint8))


def test_adjacent_float32_levels_are_a_valid_otsu_input():
    low = np.float32(.5)
    high = np.nextafter(low, np.float32(1))
    pixels = np.array([[low, low, high, high]], dtype=np.float32)
    np.testing.assert_array_equal(binarize_otsu(pixels), [[0, 0, 1, 1]])


@pytest.mark.parametrize("counts", [(999, 1, 1), (1, 999, 1), (1, 1, 999)])
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_low_contrast_otsu_matches_exact_rational_objective(counts, offset):
    # Exact rational reference uses only populated histogram bins. This catches
    # empty-tail splits and is independent of floating cumulative moments.
    from fractions import Fraction
    levels = np.float32(offset) + np.arange(3, dtype=np.float32) * np.spacing(np.float32(.5))
    pixels = np.repeat(levels, counts).reshape(13, 77)
    histogram, edges = np.histogram(pixels.astype(np.float64), bins=256,
                                   range=(float(pixels.min()), float(pixels.max())))
    occupied = [(int(i), int(histogram[i])) for i in np.flatnonzero(histogram)]
    scores = []
    for threshold in range(256):
        lower = [(i, n) for i, n in occupied if i <= threshold]
        upper = [(i, n) for i, n in occupied if i > threshold]
        n0, n1 = sum(n for _, n in lower), sum(n for _, n in upper)
        if not n0 or not n1:
            scores.append(Fraction(-1))
            continue
        mean0 = Fraction(sum(n * (2 * i + 1) for i, n in lower), 512 * n0)
        mean1 = Fraction(sum(n * (2 * i + 1) for i, n in upper), 512 * n1)
        scores.append(Fraction(n0 * n1, pixels.size ** 2) * (mean1 - mean0) ** 2)
    best = max(range(256), key=scores.__getitem__)
    expected = float((edges[best] + edges[best + 1]) / 2)
    assert otsu_threshold(pixels) == expected
    np.testing.assert_array_equal(binarize_otsu(pixels), pixels.astype(np.float64) > expected)


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("channels", [1, 3])
def test_uint16_byte_order_preserves_intensity_and_results(byte_order, channels):
    values = np.array([[0, 4096, 32768, 65535], [65535, 32768, 4096, 0]], dtype=np.uint16)
    if channels == 3:
        values = np.stack([values, values[:, ::-1], values], axis=-1)
    image = values.astype(byte_order + "u2")
    original_bytes = image.tobytes()
    image.setflags(write=False)
    np.testing.assert_array_equal(to_grayscale(image), to_grayscale(values))
    np.testing.assert_array_equal(enhance_iwmid(image), enhance_iwmid(values))
    gray = image if channels == 1 else to_grayscale(image)
    native_gray = values if channels == 1 else to_grayscale(values)
    assert otsu_threshold(gray) == otsu_threshold(native_gray)
    np.testing.assert_array_equal(binarize_otsu(gray), binarize_otsu(native_gray))
    assert image.tobytes() == original_bytes
