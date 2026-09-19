"""One CPU tensor contract for TC4 training and evaluation (paper §§2.3, 3.2).

Validate before casts, forward, or optimizer changes. Labels are integer IDs
0..4 or an explicit ignore_index; an ID list is required only by evaluation.
"""
from collections.abc import Mapping, Sequence
import torch


def validate_batch(batch, *, ignore_index=255, require_ids=False):
    """Return float32 RGB, int64 labels, and optional sample IDs on CPU.

    Integer widening is safe only after the dtype check; comparing int8 labels
    directly with 255 can overflow the ignore sentinel. Check image finiteness
    after float32 conversion too, since finite float64 values can overflow it.
    """
    if not isinstance(batch, Mapping) or not {"image", "mask"}.issubset(batch):
        raise ValueError("batch must contain image and mask tensors")
    if type(ignore_index) is not int or 0 <= ignore_index < 5:
        raise ValueError("ignore_index must be an integer outside class IDs 0..4")
    images, target = batch["image"], batch["mask"]
    if not isinstance(images, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("image and mask must be CPU tensors")
    if images.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("image and mask must already be on CPU")
    if images.layout != torch.strided or target.layout != torch.strided:
        raise TypeError("image and mask must be dense tensors")
    if images.ndim != 4 or images.shape[1] != 3 or any(d == 0 for d in images.shape):
        raise ValueError("image must be nonempty with shape (N, 3, H, W)")
    if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("mask must contain integer class IDs, excluding bool")
    target = target.to(dtype=torch.long)
    if target.shape != (images.shape[0], *images.shape[-2:]):
        raise ValueError("mask must have shape (N, H, W) aligned with image")
    if torch.any((target != ignore_index) & ((target < 0) | (target >= 5))):
        raise ValueError("mask contains an invalid class ID")
    if not images.is_floating_point():
        raise ValueError("normalized image must contain finite floating-point values")
    images = images.to(dtype=torch.float32)
    if not torch.isfinite(images).all():
        raise ValueError("normalized image must contain finite float32 values")
    ids = None
    if require_ids:
        ids = batch.get("id")
        if (not isinstance(ids, Sequence) or isinstance(ids, (str, bytes))
                or len(ids) != images.shape[0] or any(not isinstance(item, str) for item in ids)):
            raise ValueError("id must be a string sequence with one ID per image")
        ids = list(ids)
    return images, target, ids
