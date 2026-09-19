# Data interface

We use five semantic IDs: `0` for equiaxed alpha and `1,2,3,4` for the four colony orientation classes. These correspond to Classes 1–5 in our paper. Beta is not a sixth class.

## Local image/mask pairs

Create separate train, validation and test CSV manifests:

```csv
id,image,mask
sample_001,images/sample_001.png,masks/sample_001.png
```

Paths are relative to the CSV directory. We accept 8-bit RGB or grayscale input images and single-band integer masks with labels `0..4`. Set `label_offset=1` for masks labeled `1..5`. Palette PNG masks are read as class indices, not converted to RGB. Convert color-coded annotations through an explicit color-to-class mapping before use. High-bit-depth images require a documented intensity conversion to 8-bit before loading.

```python
from data import TC4Dataset

dataset = TC4Dataset("local_data/train.csv", image_size=(1024, 1024))
sample = dataset[0]
```

| Field | Type and shape | Description |
| --- | --- | --- |
| `image` | CPU float32 `[3,H,W]` | Enhanced RGB with ImageNet normalization |
| `mask` | CPU int64 `[H,W]` | Class IDs 0–4 |
| `prior` | CPU float32 `[1,H,W]` | Binary Otsu prior |
| `id` | string | Sample identifier |

We pass `image` to the network and expose `prior` separately. `preprocess=False` skips IWMID while retaining Otsu; `normalize=False` returns RGB in `[0,1]`.

## Resizing and augmentation

We center-crop each pair to the target aspect ratio, then resize RGB bilinearly and labels by nearest neighbor. This preserves orientation by applying the same scale to both axes; cropping can remove image margins. `image_size=None` retains native size and requires compatible batch collation. Square target sizes are recommended. Large coprime target dimensions require a source image large enough for the corresponding exact-ratio crop.

Paired augmentation applies the same crop to image, mask and prior. Brightness and contrast affect only RGB. We do not apply rotations, flips, elastic deformation or anisotropic resizing, because they can change absolute orientation labels. Custom transforms must use `(image, mask, prior, *, rng) -> (image, mask, prior)` and preserve label semantics. RGBA inputs discard alpha; provide labels and validity information separately.

## Split integrity

Keep all fields of view and patches from one specimen in the same split. We reject duplicate IDs, resolved file paths and identical image bytes across splits. Checkpoint fingerprints also detect changed input files during resume. File hashes do not identify related patches or re-encoded copies, so specimen grouping remains part of manifest preparation.

## Synthetic fixtures

`DummyTC4Dataset` has the same sample interface. Its deterministic textures and labels support unit tests, visualization and smoke checks. The illustrative orientations are fixture settings. We keep their metrics separate from the experimental tables in `results/paper/`.

```sh
python -m scripts.preview_data --size 64 --output runs/dummy_preview.png
python -m scripts.preview_preprocessing --output runs/iwmid_preview
```

We do not search for or download microscopy data. See [data availability](../docs/DATA_AVAILABILITY.md).
