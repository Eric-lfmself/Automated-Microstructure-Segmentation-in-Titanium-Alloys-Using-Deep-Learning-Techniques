"""Common weighted-CE loop for paper Sections 2.3 and 3.1.

No device discovery or GPU use. Epoch CE is the sum of weighted pixel losses
divided by the sum of valid-pixel class weights, not an average of batch means.
"""

import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from data.batch import validate_batch


def seed_everything(seed):
    """Seed CPU-only sources; deterministic continuation also needs saved RNGs."""
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    # Seed the CPU generator explicitly; do not probe or seed GPU runtimes.
    torch.random.default_generator.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run_epoch(model, loader, criterion, *, optimizer=None, max_batches=None):
    """Train if optimizer is provided; otherwise validate under no_grad.

    All-ignored batches are skipped without a model/optimizer step. Reject an
    epoch with no valid pixels so NaN loss never enters history/checkpoints.
    ``max_batches`` limits retrieved batches, including ignored batches.
    Labels are validated before forward; nonfinite gradients cannot reach an
    optimizer step. This phase helper intentionally sets the model's mode.
    """
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be a positive integer or None")
    if any(p.device.type != "cpu" for p in (*model.parameters(), *model.buffers())):
        raise ValueError("This loop requires an explicitly CPU model")
    if not isinstance(criterion, torch.nn.CrossEntropyLoss) or criterion.label_smoothing != 0:
        raise ValueError("criterion must be CrossEntropyLoss with no label smoothing")
    if criterion.weight is not None and criterion.weight.device.type != "cpu":
        raise ValueError("criterion weights must be on CPU")
    training = optimizer is not None
    model.train(training)
    weighted_sum = weight_mass = 0.0
    correct = valid_pixels = batches = ignored_batches = 0
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        iterator = iter(loader)
        while max_batches is None or batches + ignored_batches < max_batches:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            images, targets, _ = validate_batch(batch, ignore_index=criterion.ignore_index)
            valid = targets != criterion.ignore_index
            if not bool(valid.any()):
                ignored_batches += 1
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (targets.shape[0], 5, *targets.shape[-2:]):
                raise ValueError("Model output must have shape [N,5,H,W] matching labels")
            weights = criterion.weight
            numerator = F.cross_entropy(logits, targets, weight=weights,
                                        ignore_index=criterion.ignore_index, reduction="sum")
            denominator = (weights[targets[valid]].sum() if weights is not None
                           else valid.sum().to(dtype=logits.dtype))
            loss = numerator / denominator
            if not bool(torch.isfinite(loss)) or not bool(denominator > 0):
                raise ValueError("Nonfinite loss or zero valid class-weight mass")
            if optimizer is not None:
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    optimizer.zero_grad(set_to_none=True)
                    raise ValueError("Nonfinite gradients; optimizer step was skipped")
                optimizer.step()
            weighted_sum += float(numerator.detach())
            weight_mass += float(denominator.detach())
            correct += int(((logits.argmax(1) == targets) & valid).sum())
            valid_pixels += int(valid.sum())
            batches += 1
    if valid_pixels == 0 or weight_mass <= 0:
        raise ValueError("Epoch has no valid labeled pixels")
    return {"loss": weighted_sum / weight_mass, "pixel_accuracy": correct / valid_pixels,
            "valid_pixels": valid_pixels, "weight_mass": weight_mass,
            "batches": batches, "ignored_batches": ignored_batches,
            "seconds": time.perf_counter() - start}
