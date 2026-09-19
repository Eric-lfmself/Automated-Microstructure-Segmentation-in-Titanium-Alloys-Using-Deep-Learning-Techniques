"""Small CPU timing tests with a deterministic clock; no large inference."""

import numpy as np
import pytest
import torch
from torch import nn

from eval.timing import benchmark_model_cpu, time_pipeline_cpu


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Dropout()
        self.calls = 0

    def forward(self, image):
        assert image.device.type == "cpu"
        assert not torch.is_grad_enabled()
        assert not self.training and not self.layer.training
        self.calls += 1
        return image


def test_model_timing_excludes_warmup_and_restores_each_mode():
    model = CountingModel()
    model.layer.eval()  # Deliberately mixed module flags, restored exactly.
    ticks = iter([1, 1.1, 2, 2.3])
    result = benchmark_model_cpu(model, torch.zeros((2, 3, 4, 4), device="cpu"),
                                 warmup=1, repetitions=2, clock=lambda: next(ticks))
    assert model.calls == 3
    assert model.training and not model.layer.training
    assert result["seconds"] == pytest.approx([0.1, 0.3])
    assert result["seconds_mean"] == pytest.approx(0.2)
    assert result["seconds_per_image"] == pytest.approx(0.1)
    assert result["seconds_std"] == pytest.approx(0.1)
    assert result["scope"] == "model_only"


def test_model_mode_restored_after_forward_exception():
    class Broken(nn.Module):
        def forward(self, image):
            raise RuntimeError("deliberate test error")
    model = Broken()
    with pytest.raises(RuntimeError, match="deliberate"):
        benchmark_model_cpu(model, torch.zeros((1, 3, 2, 2), device="cpu"), warmup=0, repetitions=1)
    assert model.training


@pytest.mark.parametrize("kwargs", [{"warmup": 4}, {"warmup": -1}, {"repetitions": 0}, {"repetitions": 11}, {"repetitions": True}])
def test_invalid_repetition_limits_rejected_before_forward(kwargs):
    model = CountingModel()
    with pytest.raises(ValueError):
        benchmark_model_cpu(model, torch.zeros((1, 3, 4, 4), device="cpu"), **kwargs)
    assert model.calls == 0


@pytest.mark.parametrize("shape", [(5, 3, 4, 4), (1, 3, 129, 4), (1, 3, 4, 129), (1, 4, 4, 4), (0, 3, 4, 4)])
def test_large_or_invalid_inputs_rejected(shape):
    model = CountingModel()
    with pytest.raises(ValueError):
        benchmark_model_cpu(model, torch.zeros(shape, device="cpu"))
    assert model.calls == 0


def test_non_monotonic_clock_rejected_and_mode_restored():
    model = CountingModel()
    ticks = iter([2, 1])
    with pytest.raises(ValueError, match="monotonic"):
        benchmark_model_cpu(model, torch.zeros((1, 3, 2, 2), device="cpu"),
                            warmup=0, repetitions=1, clock=lambda: next(ticks))
    assert model.training


def test_future_real_data_option_is_explicit_and_tested_only_on_tiny_input():
    model = CountingModel()
    ticks = iter([1, 1.1])
    result = benchmark_model_cpu(model, torch.zeros((1, 3, 2, 2), device="cpu"),
                                 warmup=0, repetitions=1, enforce_small=False, clock=lambda: next(ticks))
    assert result["enforce_small"] is False
    with pytest.raises(ValueError, match="enforce_small"):
        benchmark_model_cpu(model, torch.zeros((1, 3, 2, 2), device="cpu"), enforce_small=0)


def test_pipeline_reports_each_single_invocation_and_total():
    seen = []
    def preprocessing(image):
        seen.append("pre")
        return image + 1
    def inference(image):
        seen.append("model")
        assert not torch.is_grad_enabled()
        return image * 2
    def postprocessing(image):
        seen.append("post")
        return image - 3
    ticks = iter([10, 10.2, 10.7, 10.8])
    output, result = time_pipeline_cpu(np.zeros((4, 4)), preprocessing, inference, postprocessing,
                                      clock=lambda: next(ticks))
    np.testing.assert_array_equal(output, -1)
    assert seen == ["pre", "model", "post"]
    assert result["preprocessing_seconds"] == pytest.approx(0.2)
    assert result["inference_seconds"] == pytest.approx(0.5)
    assert result["postprocessing_seconds"] == pytest.approx(0.1)
    assert result["total_seconds"] == pytest.approx(0.8)
    assert result["invocations"] == 1


def test_pipeline_rejects_training_module_without_invocation():
    model = CountingModel()
    with pytest.raises(ValueError, match="evaluation"):
        time_pipeline_cpu(torch.zeros((1, 3, 4, 4), device="cpu"), lambda x: x, model, lambda x: x)
    assert model.calls == 0
