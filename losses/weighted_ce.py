"""Shared weighted cross entropy specified in paper §§2.3.1–2.3.2.

The manuscript does not report class weights. Equal positive weights are the
transparent default; real-data weights must be estimated from the training
split only. Labels are zero-based 0..4; 255 is an optional ignore label.
"""

from collections.abc import Sequence
import math

import torch
from torch import nn


def make_weighted_cross_entropy(
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    ignore_index: int = 255,
) -> nn.CrossEntropyLoss:
    """Return a CPU criterion using five finite, strictly positive weights.

    Strict positivity prevents zero total class weight for a valid batch.
    Weights are rescaled by their maximum (mean CE is invariant to this scale).
    A batch containing only ignored pixels must be rejected/skipped by the
    training loop: PyTorch's mean reduction has no denominator in that case.
    """
    if len(weights) != 5:
        raise ValueError("Exactly five class weights are required for TC4")
    try:
        values = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as exc:
        raise ValueError("Class weights must be numeric") from exc
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Every class weight must be finite and strictly positive")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int) or 0 <= ignore_index < 5:
        raise ValueError("ignore_index must be an integer outside class IDs 0..4")
    # Mean CE is invariant to a common scale of class weights. Normalize in
    # Python first so even large, finite user weights cannot overflow sums.
    largest = max(values)
    relative = tuple(value / largest for value in values)
    tensor = torch.tensor(relative, dtype=torch.float32, device="cpu")
    if not torch.isfinite(tensor).all() or not (tensor > 0).all():
        raise ValueError("Class weights must remain finite and positive in float32")
    return nn.CrossEntropyLoss(weight=tensor, ignore_index=ignore_index)
