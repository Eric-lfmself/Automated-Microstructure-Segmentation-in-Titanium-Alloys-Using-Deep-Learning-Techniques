"""Shared CPU evaluation, paper Section 3.2 and Tables 1–3.

All pixels are aggregated into one confusion matrix. Dataset/preprocessing time
is separate from the repeated model-only latency, so timings cannot be mistaken
for the historical 36-second full pipeline claim in Section 3.4.
"""
from time import perf_counter
import torch
from eval.metrics import SegmentationMetrics
from data.batch import validate_batch



def evaluate_loader(model, loader, *, max_batches=None):
    """Return global metrics and one small batch for an optional latency probe."""
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be a positive integer or None")
    if any(p.device.type != "cpu" for p in (*model.parameters(), *model.buffers())):
        raise ValueError("Evaluation is explicitly CPU only")
    original_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    accumulator = SegmentationMetrics()
    sample_ids = []
    first_batch = None
    start = perf_counter()
    try:
        with torch.inference_mode():
            iterator = iter(loader)
            batches = 0
            while max_batches is None or batches < max_batches:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                images, target, ids = validate_batch(batch, require_ids=True)
                logits = model(images)
                accumulator.update(logits, target)
                sample_ids.extend(ids)
                if first_batch is None:
                    first_batch = {"image": images, "mask": target,
                                   "prediction": logits.argmax(1), "id": ids}
                batches += 1
    finally:
        for module, mode in original_modes:
            module.training = mode
    if first_batch is None:
        raise ValueError("Evaluation loader is empty")
    result = accumulator.compute()
    result.update(sample_ids=sample_ids, image_count=len(sample_ids),
                  evaluation_wall_seconds=perf_counter()-start,
                  evaluation_wall_scope="Dataset loading, preprocessing, forward, argmax, confusion update; excludes model setup")
    return result, first_batch
