"""Integrity of our reported table archive and its public export command."""
import csv
from decimal import Decimal
import json
from pathlib import Path

import pytest

from eval.metrics import PAPER_REFERENCE
from scripts.export_paper_results import export_results


def test_export_preserves_table_values_and_separate_arithmetic(tmp_path):
    out = export_results(tmp_path / "paper")
    data = json.loads((out / "paper_results.json").read_text())
    comparison = list(csv.DictReader((out / "table_3_model_comparison.csv").open()))
    classes = list(csv.DictReader((out / "tables_1_2_per_class.csv").open()))
    assert len(comparison) == 2 and len(classes) == 10
    for model, reference in PAPER_REFERENCE.items():
        rows = [r for r in classes if r['model'] == model]
        assert [int(r['paper_class']) for r in rows] == [1, 2, 3, 4, 5]
        assert [float(r['per_class_iou']) for r in rows] == reference['per_class_iou']
        table3 = next(r for r in comparison if r['model'] == model)
        for csv_key, reference_key in [('pixel_accuracy', 'pixel_accuracy'), ('miou', 'miou'), ('alpha_phase_iou', 'equiaxed_iou'), ('avg_colony_iou_classes_2_5', 'colony_miou')]:
            assert Decimal(table3[csv_key]) == Decimal(str(reference[reference_key]))
        values = [Decimal(r['per_class_iou']) for r in rows]
        for entry in (r for r in data['arithmetic_check'] if r['model'] == model):
            selected = values if entry['metric'] == 'miou' else values[1:]
            mean = sum(selected) / len(selected)
            assert mean == Decimal(entry['arithmetic_mean_of_displayed_class_values'])
            assert mean - Decimal(entry['table_3_reported']) == Decimal(entry['calculated_minus_reported'])
    assert data['value_kind'] == 'paper_reported_experimental_results'
    assert next(r for r in comparison if r['model'] == 'unet_vgg')['pixel_accuracy'] == '0.9000'


def test_export_refuses_to_overwrite_existing_experiment(tmp_path):
    target = tmp_path / 'experiment'; target.mkdir()
    checkpoint = target / 'checkpoint.pt'; checkpoint.write_bytes(b'existing run')
    with pytest.raises(ValueError, match='empty'):
        export_results(target)
    assert checkpoint.read_bytes() == b'existing run'
    assert list(target.iterdir()) == [checkpoint]


def test_export_works_outside_repository_and_matches_archive(tmp_path, monkeypatch):
    archive = Path(__file__).resolve().parents[1] / 'results/paper'
    monkeypatch.chdir(tmp_path)
    exported = export_results('tables')
    assert {p.name for p in exported.iterdir()} == {p.name for p in archive.iterdir()}
    for file in archive.iterdir():
        assert (exported / file.name).read_bytes() == file.read_bytes()
