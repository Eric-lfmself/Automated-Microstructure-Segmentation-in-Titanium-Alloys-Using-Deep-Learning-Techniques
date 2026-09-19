"""Small CPU-only fixtures exercise the dataset contract and orientation safety."""

import csv

import numpy as np
from PIL import Image
import pytest
import torch

from data import DummyTC4Dataset, TC4Dataset, PairedAugment, denormalize_image


def _manifest(tmp_path, mask=None, shape=(16, 16), offset=0):
    if mask is None:
        y, x = np.mgrid[:shape[0], :shape[1]]
        mask = ((x // 4 + y // 4) % 5).astype(np.uint8)
    image = np.repeat((mask[:shape[0], :shape[1]] * 50)[..., None], 3, axis=-1).astype(np.uint8)
    if image.shape[:2] != shape:
        image = np.zeros((*shape, 3), np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray((mask + offset).astype(np.uint8)).save(tmp_path / "mask.png")
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "image", "mask"))
        writer.writerow(("specimen_1", "image.png", "mask.png"))
    return path, image, mask


def test_dummy_contract_determinism_and_classes():
    dataset = DummyTC4Dataset(length=2, image_size=(64, 64), seed=123)
    first = dataset[0]
    repeated = dataset[0]
    recreated = DummyTC4Dataset(length=2, image_size=(64, 64), seed=123)[0]
    assert first["image"].shape == (3, 64, 64)
    assert first["mask"].shape == (64, 64)
    assert first["prior"].shape == (1, 64, 64)
    assert first["image"].dtype == first["prior"].dtype == torch.float32
    assert first["mask"].dtype == torch.int64
    assert first["image"].device == torch.device("cpu")
    assert torch.unique(first["mask"]).tolist() == [0, 1, 2, 3, 4]
    assert set(torch.unique(first["prior"]).tolist()) <= {0.0, 1.0}
    assert first["id"] == "dummy_0000"
    for field in ("image", "mask", "prior"):
        assert torch.equal(first[field], repeated[field])
        assert torch.equal(first[field], recreated[field])
    assert not torch.equal(first["image"], dataset[1]["image"])
    assert not torch.equal(first["image"], DummyTC4Dataset(image_size=(64, 64), seed=124)[0]["image"])


def test_normalization_and_epoch_controls():
    normalized = DummyTC4Dataset(image_size=(64, 64))[0]["image"]
    raw = DummyTC4Dataset(image_size=(64, 64), normalize=False)[0]["image"]
    torch.testing.assert_close(denormalize_image(normalized), raw)
    transform = PairedAugment(crop_size=(32, 32), brightness=.15, contrast=.15)
    dataset = DummyTC4Dataset(image_size=(64, 64), transform=transform)
    epoch0 = dataset[0]
    dataset.set_epoch(1)
    epoch1 = dataset[0]
    assert not torch.equal(epoch0["image"], epoch1["image"])
    dataset.set_epoch(0)
    assert torch.equal(epoch0["image"], dataset[0]["image"])


def test_paired_crop_preserves_all_alignment_and_orientation():
    y, x = np.mgrid[:16, :20]
    image = np.stack((x / 20, y / 16, (x + y) / 36), axis=-1).astype(np.float32)
    mask = ((x // 4 + y // 4) % 5).astype(np.int64)
    prior = (mask % 2).astype(np.uint8)
    transform = PairedAugment(crop_size=(9, 11), brightness=0, contrast=0)
    actual_image, actual_mask, actual_prior = transform(image, mask, prior, rng=np.random.default_rng(8))
    reference_rng = np.random.default_rng(8)
    top, left = reference_rng.integers(0, 8), reference_rng.integers(0, 10)
    np.testing.assert_array_equal(actual_image, image[top:top+9, left:left+11])
    np.testing.assert_array_equal(actual_mask, mask[top:top+9, left:left+11])
    np.testing.assert_array_equal(actual_prior, prior[top:top+9, left:left+11])
    # x increases rightwards and y downwards: neither axis is flipped/swapped.
    assert np.all(np.diff(actual_image[..., 0], axis=1) > 0)
    assert np.all(np.diff(actual_image[..., 1], axis=0) > 0)


def test_photometry_never_changes_targets_or_prior():
    sample = DummyTC4Dataset(image_size=(64, 64), normalize=False)[0]
    image = sample["image"].permute(1, 2, 0).numpy()
    mask, prior = sample["mask"].numpy(), sample["prior"][0].numpy()
    output = PairedAugment(brightness=.2, contrast=.2)(image, mask, prior, rng=np.random.default_rng(3))
    assert not np.array_equal(output[0], image)
    np.testing.assert_array_equal(output[1], mask)
    np.testing.assert_array_equal(output[2], prior)


def test_real_dataset_offsets_and_exact_alignment(tmp_path):
    manifest, image, mask = _manifest(tmp_path, offset=1)
    sample = TC4Dataset(manifest, image_size=None, preprocess=False, normalize=False, label_offset=1)[0]
    np.testing.assert_array_equal(sample["mask"].numpy(), mask)
    np.testing.assert_allclose(sample["image"].permute(1, 2, 0).numpy(), image.astype(np.float32) / 255)
    assert sample["id"] == "specimen_1"
    resized = TC4Dataset(manifest, image_size=(64, 64), preprocess=False, normalize=False, label_offset=1)[0]
    np.testing.assert_array_equal(resized["mask"].numpy(), mask.repeat(4, axis=0).repeat(4, axis=1))


def test_rectangular_input_is_center_cropped_without_angle_distortion(tmp_path):
    mask = np.tile(np.arange(24) // 5, (16, 1)).astype(np.uint8)
    manifest, _, _ = _manifest(tmp_path, mask=mask, shape=(16, 24))
    sample = TC4Dataset(manifest, image_size=(16, 16), preprocess=False, normalize=False)[0]
    np.testing.assert_array_equal(sample["mask"].numpy(), mask[:, 4:20])


def test_rejects_misaligned_images_before_resizing(tmp_path):
    manifest, _, _ = _manifest(tmp_path, mask=np.zeros((17, 16), np.uint8), shape=(16, 16))
    dataset = TC4Dataset(manifest, image_size=(64, 64))
    with pytest.raises(ValueError, match="dimensions differ before resizing"):
        dataset[0]


@pytest.mark.parametrize("value,offset", [(5, 0), (0, 1), (6, 1), (255, 0)])
def test_rejects_invalid_labels(tmp_path, value, offset):
    manifest, _, _ = _manifest(tmp_path, mask=np.full((16, 16), value, np.uint8))
    with pytest.raises(ValueError, match="class IDs"):
        TC4Dataset(manifest, label_offset=offset)[0]


def test_rejects_rgb_masks_and_duplicate_records(tmp_path):
    manifest, _, _ = _manifest(tmp_path)
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="single-band"):
        TC4Dataset(manifest)[0]
    with manifest.open("a") as stream:
        stream.write("specimen_1,image.png,mask.png\n")
    with pytest.raises(ValueError, match="duplicates"):
        TC4Dataset(manifest)


def test_invalid_transform_output_cannot_misalign_prior():
    def bad_transform(image, mask, prior, *, rng):
        return image[:-1], mask[:-1], prior
    with pytest.raises(ValueError, match="spatial dimensions"):
        DummyTC4Dataset(image_size=(64, 64), transform=bad_transform)[0]


def test_invalid_configuration_and_index():
    with pytest.raises(ValueError, match="at least 16"):
        DummyTC4Dataset(image_size=(8, 8))
    with pytest.raises(ValueError, match="positive integers"):
        DummyTC4Dataset(image_size=(64.5, 64))
    with pytest.raises(ValueError, match="brightness"):
        PairedAugment(brightness=float("nan"))
    with pytest.raises(IndexError):
        DummyTC4Dataset(length=1)[1]


def test_rejects_high_bit_depth_images_instead_of_saturating(tmp_path):
    manifest, _, _ = _manifest(tmp_path)
    image = np.linspace(0, 65535, 256).reshape(16, 16).astype(np.uint16)
    Image.fromarray(image).save(tmp_path / 'image16.tiff')
    manifest.write_text('id,image,mask\nspecimen_1,image16.tiff,mask.png\n')
    with pytest.raises(ValueError, match='unsupported image mode'):
        TC4Dataset(manifest, image_size=(64, 64))[0]


def test_minimum_dummy_resolution_retains_all_five_classes():
    sample = DummyTC4Dataset(length=1, image_size=(16, 16))[0]
    assert torch.unique(sample["mask"]).tolist() == [0, 1, 2, 3, 4]
