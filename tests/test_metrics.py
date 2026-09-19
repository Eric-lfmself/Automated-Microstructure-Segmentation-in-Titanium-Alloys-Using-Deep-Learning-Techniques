"""Analytic, CPU-only tests for Table 1/2/3 metrics and source-value preservation."""

import json

import numpy as np
import pytest
import torch

from eval.metrics import PAPER_REFERENCE, SegmentationMetrics, paper_reference_audit, render_comparison_table, render_per_class_table


def test_analytical_confusion_matrix_and_absent_classes():
    metrics = SegmentationMetrics()
    truth = np.array([[0, 0, 1], [1, 2, 255]])
    predicted = np.array([[0, 1, 1], [2, 2, 4]])
    metrics.update(predicted, truth)
    result = metrics.compute()
    assert result["confusion_matrix"] == [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0] * 5, [0] * 5]
    assert result["pixel_accuracy"] == 3 / 5
    assert result["per_class_iou"] == [1 / 2, 1 / 3, 1 / 2, None, None]
    assert result["miou"] == pytest.approx((1 / 2 + 1 / 3 + 1 / 2) / 3)
    assert result["equiaxed_iou"] == 1 / 2
    assert result["colony_miou"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert result["valid_pixels"] == 5 and result["ignored_pixels"] == 1
    assert result["per_class_support"] == [2, 2, 1, 0, 0]
    json.dumps(result, allow_nan=False)


def test_global_aggregation_is_independent_of_batches():
    truth = torch.tensor([[[0, 1], [1, 1]], [[2, 2], [3, 4]], [[0, 255], [0, 4]]], device="cpu")
    prediction = torch.tensor([[[0, 1], [1, 2]], [[2, 3], [4, 4]], [[1, 3], [1, 3]]], device="cpu")
    once, separate = SegmentationMetrics(), SegmentationMetrics()
    once.update(prediction, truth)
    for pred, target in zip(prediction, truth):
        separate.update(pred, target)
    assert once.compute() == separate.compute()


def test_float_logits_equal_integer_predictions_single_and_batched():
    truth = torch.tensor([[[0, 1], [3, 4]]], device="cpu")
    logits = torch.nn.functional.one_hot(truth, 5).permute(0, 3, 1, 2).float().requires_grad_()
    results = []
    for pred, target in ((truth, truth), (logits, truth), (logits[0], truth[0])):
        metrics = SegmentationMetrics()
        metrics.update(pred, target)
        results.append(metrics.compute())
    assert results[0] == results[1] == results[2]
    assert results[0]["miou"] == 1 and results[0]["per_class_iou"][2] is None


def test_false_positive_absent_ground_truth_class_counts_as_zero_iou():
    metrics = SegmentationMetrics()
    metrics.update(np.array([[0, 1]]), np.array([[0, 0]]))
    assert metrics.compute()["per_class_iou"] == [0.5, 0.0, None, None, None]
    assert metrics.compute()["miou"] == 0.25


def test_empty_state_all_ignored_and_reset_have_undefined_scores():
    metrics = SegmentationMetrics()
    for result in (metrics.compute(),):
        assert result["pixel_accuracy"] is None and result["miou"] is None
        assert result["colony_miou"] is None and result["per_class_iou"] == [None] * 5
    metrics.update(np.zeros((2, 2), dtype=int), np.full((2, 2), 255, dtype=int))
    assert metrics.compute()["valid_pixels"] == 0
    assert metrics.compute()["ignored_pixels"] == 4
    assert metrics.compute()["pixel_accuracy"] is None
    metrics.reset()
    assert metrics.compute()["ignored_pixels"] == 0
    assert not np.any(metrics.confusion_matrix)


@pytest.mark.parametrize("prediction,target,error", [
    (np.array([[5]]), np.array([[255]]), ValueError),
    (np.array([[-1]]), np.array([[255]]), ValueError),
    (np.array([[0]]), np.array([[5]]), ValueError),
    (np.array([[0]]), np.array([[-1]]), ValueError),
    (np.array([[0.0]]), np.array([[0]]), TypeError),
    (np.array([[0]]), np.array([[0.0]]), TypeError),
    (np.array([[True]]), np.array([[0]]), TypeError),
    (np.array([[0]]), np.array([[False]]), TypeError),
    (np.zeros((5, 1, 1), dtype=int), np.array([[0]]), TypeError),
    (np.full((5, 1, 1), np.nan), np.array([[0]]), ValueError),
    (np.zeros((4, 1, 1)), np.array([[0]]), ValueError),
    (np.zeros((1, 2), dtype=int), np.array([[0]]), ValueError),
    (np.zeros((0, 1), dtype=int), np.zeros((0, 1), dtype=int), ValueError),
    ([0], np.array([[0]]), TypeError),
])
def test_invalid_updates_are_rejected_atomically(prediction, target, error):
    metrics = SegmentationMetrics()
    original = metrics.compute()
    with pytest.raises(error):
        metrics.update(prediction, target)
    assert metrics.compute() == original


@pytest.mark.parametrize("kwargs", [{"num_classes": 0}, {"num_classes": True}, {"ignore_index": 4}, {"ignore_index": 1.5}])
def test_invalid_metric_configuration(kwargs):
    with pytest.raises(ValueError):
        SegmentationMetrics(**kwargs)


def test_source_reference_values_are_preserved_and_means_are_separate():
    assert PAPER_REFERENCE["unet_vgg"]["per_class_iou"] == [0.8746, 0.5602, 0.6853, 0.7251, 0.6452]
    assert PAPER_REFERENCE["segformer"]["per_class_iou"] == [0.8410, 0.5803, 0.6955, 0.7397, 0.6704]
    audit = paper_reference_audit()
    assert audit["unet_vgg"]["paper_reported"]["miou"] == 0.6989
    assert audit["unet_vgg"]["recomputed_from_printed_class_ious"]["miou"] == pytest.approx(0.69808)
    assert audit["unet_vgg"]["recomputed_from_printed_class_ious"]["colony_miou"] == pytest.approx(0.65395)
    assert audit["segformer"]["paper_reported"]["miou"] == 0.6725
    assert audit["segformer"]["paper_reported"]["colony_miou"] == 0.6725
    assert audit["segformer"]["recomputed_from_printed_class_ious"]["miou"] == pytest.approx(0.70538)
    assert audit["segformer"]["recomputed_from_printed_class_ious"]["colony_miou"] == pytest.approx(0.671475)
    audit["unet_vgg"]["paper_reported"]["per_class_iou"][0] = 0
    assert PAPER_REFERENCE["unet_vgg"]["per_class_iou"][0] == 0.8746


def test_table_renderers_show_undefined_classes_and_paper_values():
    empty = SegmentationMetrics().compute()
    table = render_per_class_table(empty, "Random-init smoke")
    assert "| 1 | 0 | Equiaxed α-phase | N/A |" in table
    comparison = render_comparison_table({"VGG based U-Net": PAPER_REFERENCE["unet_vgg"]})
    assert "| VGG based U-Net | 0.9000 | 0.6989 | 0.8746 | 0.6538 |" in comparison
