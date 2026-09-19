"""Dataset-aggregated metrics for our paper Section 3.2, Tables 1-3.

Code IDs 0-4 correspond to paper Classes 1-5. Confusion-matrix rows are ground
truth and columns are predictions. We aggregate pixel counts before computing
ratios. Zero-union classes are excluded from macro means and have IoU None;
accuracy is None when every target pixel is ignored.

PAPER_REFERENCE preserves the experimental values in our manuscript.
paper_reference_audit separately computes arithmetic means from the printed
class IoUs, retaining Table 3 exactly. See results/paper for the released tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np
import torch


CLASS_NAMES = ("Equiaxed α-phase", "Colony (Orientation 1)", "Colony (Orientation 2)",
               "Colony (Orientation 3)", "Colony (Orientation 4)")

PAPER_REFERENCE = {
    "unet_vgg": {
        "model_name": "VGG based U-Net", "per_class_table": 1,
        "per_class_iou": [0.8746, 0.5602, 0.6853, 0.7251, 0.6452],
        "pixel_accuracy": 0.9000, "miou": 0.6989,
        "equiaxed_iou": 0.8746, "colony_miou": 0.6538,
    },
    "segformer": {
        "model_name": "SegFormer", "per_class_table": 2,
        "per_class_iou": [0.8410, 0.5803, 0.6955, 0.7397, 0.6704],
        "pixel_accuracy": 0.8449, "miou": 0.6725,
        "equiaxed_iou": 0.8410, "colony_miou": 0.6725,
    },
}


def _numpy_cpu(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on device='cpu'")
        value = value.detach()
        # numpy does not represent bfloat16; this changes storage, not labels.
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        try:
            return value.numpy()
        except (TypeError, RuntimeError) as exc:
            raise TypeError(f"{name} must be a dense, real-valued tensor") from exc
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array or CPU torch.Tensor")
    return value


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


class SegmentationMetrics:
    """Accumulate a confusion matrix for paper Section 3.2 Tables 1–3.

    ``update(prediction, target)`` accepts integer maps shaped H×W or N×H×W;
    alternatively prediction may be float logits shaped C×H×W or N×C×H×W.
    Targets must have an integer dtype, with valid IDs or ``ignore_index``.
    Invalid predictions are rejected even at ignored target locations. Updates
    validate completely before changing the running counts.
    """

    def __init__(self, num_classes: int = 5, ignore_index: int = 255) -> None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise ValueError("ignore_index must be an integer")
        if 0 <= ignore_index < num_classes:
            raise ValueError("ignore_index must not overlap a valid class ID")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear all counts, including ignored-pixel counts."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.ignored_pixels = 0

    def update(self, prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> None:
        """Validate and accumulate integer pixel counts, with no model invocation."""
        truth = _numpy_cpu(target, "target")
        predicted = _numpy_cpu(prediction, "prediction")
        if truth.ndim not in (2, 3) or any(dimension == 0 for dimension in truth.shape):
            raise ValueError("target must be nonempty with shape (H, W) or (N, H, W)")
        if not np.issubdtype(truth.dtype, np.integer):
            raise TypeError("target must contain integer class IDs")
        ignored = truth == self.ignore_index
        if np.any((~ignored) & ((truth < 0) | (truth >= self.num_classes))):
            raise ValueError("target contains an invalid class ID")

        if predicted.shape == truth.shape:
            if not np.issubdtype(predicted.dtype, np.integer):
                raise TypeError("prediction maps must contain integer class IDs")
            if np.any((predicted < 0) | (predicted >= self.num_classes)):
                raise ValueError("prediction contains an invalid class ID")
        else:
            class_axis = 0 if truth.ndim == 2 else 1
            expected_shape = list(truth.shape)
            expected_shape.insert(class_axis, self.num_classes)
            if predicted.shape != tuple(expected_shape):
                raise ValueError(f"prediction shape must be {truth.shape} (labels) or {tuple(expected_shape)} (logits)")
            if not np.issubdtype(predicted.dtype, np.floating):
                raise TypeError("logits must have a floating-point dtype")
            if not np.isfinite(predicted).all():
                raise ValueError("logits must contain only finite values")
            predicted = predicted.argmax(axis=class_axis)

        valid = ~ignored
        encoded = self.num_classes * truth[valid].astype(np.int64) + predicted[valid].astype(np.int64)
        additions = np.bincount(encoded, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += additions
        self.ignored_pixels += int(ignored.sum())

    def compute(self) -> dict:
        """Return JSON-compatible metrics; undefined quantities are ``None``."""
        counts = self.confusion_matrix
        support = counts.sum(axis=1)
        predicted = counts.sum(axis=0)
        diagonal = np.diag(counts)
        union = support + predicted - diagonal
        total = int(counts.sum())
        ious = [float(correct / denominator) if denominator else None
                for correct, denominator in zip(diagonal, union)]
        return {
            "pixel_accuracy": float(diagonal.sum() / total) if total else None,
            "miou": _mean_defined(ious), "per_class_iou": ious,
            "equiaxed_iou": ious[0], "colony_miou": _mean_defined(ious[1:]),
            "confusion_matrix": counts.tolist(), "valid_pixels": total,
            "ignored_pixels": self.ignored_pixels, "per_class_support": support.tolist(),
            "per_class_predicted_pixels": predicted.tolist(), "per_class_union": union.tolist(),
            "num_classes": self.num_classes, "ignore_index": self.ignore_index,
            "absent_class_policy": "zero-union classes excluded from macro means",
        }


def _format(value: float | None) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.4f}"


def _escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_per_class_table(result: Mapping, model_name: str = "Model") -> str:
    """Render per-class IoU in the style of paper Tables 1/2 (not per-class mIoU)."""
    lines = [f"{_escape_cell(model_name)} — per-class IoU (paper Tables 1/2 format)", "",
             "| Paper class | Code ID | Structure | IoU |", "| --- | --- | --- | ---: |"]
    for index, value in enumerate(result["per_class_iou"]):
        name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"Class {index + 1}"
        lines.append(f"| {index + 1} | {index} | {name} | {_format(value)} |")
    return "\n".join(lines)


def render_comparison_table(results: Mapping[str, Mapping]) -> str:
    """Render supplied model results using the four columns of paper Table 3."""
    lines = ["| Model | Pixel accuracy | mIoU | Equiaxed α IoU | Mean colony IoU (2–5) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in results.items():
        cells = [_format(values[key]) for key in ("pixel_accuracy", "miou", "equiaxed_iou", "colony_miou")]
        lines.append(f"| {_escape_cell(name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paper_reference_audit() -> dict:
    """Keep printed Table 3 numbers separate from means recomputed from Tables 1/2.

    We retain our printed tables exactly and expose their arithmetic differences
    without replacing reported results.
    """
    audit = {}
    for key, values in PAPER_REFERENCE.items():
        per_class = values["per_class_iou"]
        calculated_miou = float(np.mean(per_class))
        calculated_colony = float(np.mean(per_class[1:]))
        audit[key] = {
            "paper_reported": deepcopy(values),
            "recomputed_from_printed_class_ious": {"miou": calculated_miou, "colony_miou": calculated_colony},
            "recomputed_minus_table3": {"miou": calculated_miou - values["miou"],
                                         "colony_miou": calculated_colony - values["colony_miou"]},
            "interpretation": "We retain the reported values and list arithmetic means separately.",
        }
    return audit
