"""Bounded, CPU-only timing helpers for paper Section 3.4.

Model-only timing excludes preprocessing/postprocessing. Pipeline timing calls
each supplied stage exactly once and reports their sum as total elapsed time.
The paper's 36 s/image covers the full pipeline at 1024×1024 on unspecified
hardware; these tiny CPU smoke measurements are not comparable, and this module
does not scale them to larger images or calculate a speedup against the paper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import time

import numpy as np
import torch
from torch import nn


def _require_cpu(value: object, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on device='cpu'")
    elif isinstance(value, nn.Module):
        for tensor in list(value.parameters()) + list(value.buffers()):
            _require_cpu(tensor, name)
    elif isinstance(value, Mapping):
        for item in value.values():
            _require_cpu(item, name)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_cpu(item, name)


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _elapsed(start: float, end: float) -> float:
    value = float(end - start)
    if not math.isfinite(value) or value < 0:
        raise ValueError("timing clock must be finite and monotonic")
    return value


def benchmark_model_cpu(
    model: nn.Module,
    inputs: torch.Tensor,
    warmup: int = 1,
    repetitions: int = 3,
    *,
    enforce_small: bool = True,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Time a model on 1–4 RGB images, normally at most 128×128, without gradients.

    Warmup is restricted to 0–3 calls and measured repetitions to 1–10 calls.
    Uses ``time.perf_counter`` by default. Mode flags are restored even on error;
    this helper does not select a device, download weights, or alter parameters.
    ``enforce_small=False`` explicitly permits larger images for future real-data
    evaluation; it is never selected automatically. Tests still use tiny inputs
    when checking that this option is accepted.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("inputs must be a CPU RGB tensor with shape (N, 3, H, W)")
    if not isinstance(enforce_small, bool):
        raise ValueError("enforce_small must be a bool")
    _bounded_integer(inputs.shape[0], "batch size", 1, 4)
    if any(dimension < 1 for dimension in inputs.shape[2:]):
        raise ValueError("height and width must be positive")
    if enforce_small:
        _bounded_integer(inputs.shape[2], "height", 1, 128)
        _bounded_integer(inputs.shape[3], "width", 1, 128)
    _bounded_integer(warmup, "warmup", 0, 3)
    _bounded_integer(repetitions, "repetitions", 1, 10)
    _require_cpu(model, "model")
    _require_cpu(inputs, "inputs")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("inputs must contain finite floating-point values")
    timer = clock if clock is not None else time.perf_counter
    modes = [(module, module.training) for module in model.modules()]
    durations = []
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                _require_cpu(model(inputs), "model output")
            for _ in range(repetitions):
                start = timer()
                output = model(inputs)
                end = timer()
                durations.append(_elapsed(start, end))
                _require_cpu(output, "model output")
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "scope": "model_only", "device": "cpu", "clock": "time.perf_counter" if clock is None else "injected",
        "enforce_small": enforce_small,
        "warmup": warmup, "repetitions": repetitions, "batch_size": inputs.shape[0], "input_shape": list(inputs.shape),
        "seconds": durations, "seconds_mean": float(np.mean(durations)),
        "seconds_min": min(durations), "seconds_max": max(durations), "seconds_std": float(np.std(durations)),
        "seconds_per_image": float(np.mean(durations) / inputs.shape[0]),
        "comparison_note": "CPU model-only timing; not comparable to paper 36 s full pipeline without matching hardware, resolution and stages.",
    }


def time_pipeline_cpu(
    image: object,
    preprocess: Callable,
    inference: Callable,
    postprocess: Callable,
    *,
    clock: Callable[[], float] | None = None,
) -> tuple[object, dict]:
    """Call each CPU stage once and return ``(output, per_stage_timing)``.

    This helper does not choose an image size, repeat inference, or extrapolate
    performance. The caller controls image size (tiny inputs for smoke tests) and
    supplies an inference callable with its model already in evaluation mode. If an ``nn.Module`` is
    passed directly, it must already be in evaluation mode. No training runs.
    """
    for function, name in ((preprocess, "preprocess"), (inference, "inference"), (postprocess, "postprocess")):
        if not callable(function):
            raise TypeError(f"{name} must be callable")
        _require_cpu(function, name)
        if isinstance(function, nn.Module) and any(module.training for module in function.modules()):
            raise ValueError(f"{name} module must be in evaluation mode")
    _require_cpu(image, "image")
    timer = clock if clock is not None else time.perf_counter
    with torch.inference_mode():
        start = timer()
        prepared = preprocess(image)
        _require_cpu(prepared, "preprocessing output")
        after_preprocess = timer()
        prediction = inference(prepared)
        _require_cpu(prediction, "inference output")
        after_inference = timer()
        output = postprocess(prediction)
        _require_cpu(output, "postprocessing output")
        end = timer()
    return output, {
        "scope": "single_full_pipeline", "device": "cpu", "invocations": 1,
        "clock": "time.perf_counter" if clock is None else "injected",
        "preprocessing_seconds": _elapsed(start, after_preprocess),
        "inference_seconds": _elapsed(after_preprocess, after_inference),
        "postprocessing_seconds": _elapsed(after_inference, end),
        "total_seconds": _elapsed(start, end),
        "comparison_note": "Measured stages only; no hardware or resolution extrapolation to paper Section 3.4.",
    }
