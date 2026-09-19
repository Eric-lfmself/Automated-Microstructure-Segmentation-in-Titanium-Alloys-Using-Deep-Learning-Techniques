"""Tiny local files verify immutable training identities; no real training."""

import csv
import copy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from configs.config import Config
from data.provenance import collect_provenance, assert_same_training_provenance
from training.checkpoint import read_checkpoint, save_checkpoint, validate_checkpoint_config
from training.runner import run_training, validate_split_manifests


def _splits(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for index, split in enumerate(("train", "val", "test")):
        Image.fromarray(np.full((16, 16, 3), 40 + index * 40, np.uint8)).save(directory / f"{split}_image.png")
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(directory / f"{split}_mask.png")
        path = directory / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "image", "mask"])
            writer.writerow([f"specimen_{split}", f"{split}_image.png", f"{split}_mask.png"])
        manifests[f"{split}_manifest"] = str(path)
    return replace(Config(), dry_run=False, output_dir=str(directory / "output"), **manifests)


def test_cross_split_image_copies_are_rejected_but_identical_masks_allowed(tmp_path):
    config = _splits(tmp_path / "data")
    provenance = collect_provenance(validate_split_manifests(config))
    assert provenance["train"][0]["mask_sha256"] == provenance["val"][0]["mask_sha256"]
    shutil.copyfile(tmp_path / "data/train_image.png", tmp_path / "data/val_image.png")
    with pytest.raises(ValueError, match="identical image content"):
        collect_provenance(validate_split_manifests(config))


@pytest.mark.parametrize("changed_file", ["train_image.png", "train_mask.png"])
def test_same_path_content_change_invalidates_resume_before_model_loading(tmp_path, changed_file):
    config = _splits(tmp_path / "data")
    saved = collect_provenance(validate_split_manifests(config))
    checkpoint = tmp_path / "checkpoint.pt"
    model = torch.nn.Conv2d(3, 5, 1).to(device="cpu")
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), config=config,
                    completed_epochs=1, history=[{"epoch": 1}], data_provenance=saved)
    assert read_checkpoint(checkpoint)["data_provenance"] == saved
    shape = (16, 16, 3) if "image" in changed_file else (16, 16)
    Image.fromarray(np.full(shape, 1, np.uint8)).save(tmp_path / "data" / changed_file)
    def forbidden_factory(*args, **kwargs):
        raise AssertionError("Changed data must be rejected before model creation")
    with pytest.raises(ValueError, match="Training provenance mismatch"):
        run_training(config, run_real=True, resume=checkpoint, model_factory=forbidden_factory)


def test_relocation_keeps_provenance_while_replaced_split_ids_do_not(tmp_path):
    config = _splits(tmp_path / "original")
    saved = collect_provenance(validate_split_manifests(config))
    shutil.copytree(tmp_path / "original", tmp_path / "relocated")
    current_config = replace(config, **{f"{split}_manifest": str(tmp_path / "relocated" / f"{split}.csv")
                                      for split in ("train", "val", "test")})
    current = collect_provenance(validate_split_manifests(current_config))
    assert saved["train"][0]["image_path"] != current["train"][0]["image_path"]
    assert_same_training_provenance(saved, current)
    validate_checkpoint_config({"config": config.to_dict(), "completed_epochs": 1}, current_config)
    current["train"][0]["id"] = "replacement_id"
    with pytest.raises(ValueError, match="Training provenance mismatch"):
        assert_same_training_provenance(saved, current)


def test_test_cannot_reuse_original_training_hash_even_with_new_id(tmp_path):
    config = _splits(tmp_path / "data")
    saved = collect_provenance(validate_split_manifests(config))
    current = collect_provenance(validate_split_manifests(config))
    current["test"][0]["image_sha256"] = saved["train"][0]["image_sha256"]
    with pytest.raises(ValueError, match="checkpoint's original"):
        assert_same_training_provenance(saved, current)
    with pytest.raises(ValueError, match="Missing verified"):
        assert_same_training_provenance(None, current)


def test_row_reordering_is_allowed_for_evaluation_but_rejected_for_resume(tmp_path):
    config = _splits(tmp_path / "data")
    saved = collect_provenance(validate_split_manifests(config))
    extra = dict(saved["train"][0], id="second_training_record", image_sha256="a" * 64)
    saved["train"].append(extra)
    current = copy.deepcopy(saved)
    current["train"].reverse()
    assert_same_training_provenance(saved, current)
    with pytest.raises(ValueError, match="order mismatch"):
        assert_same_training_provenance(saved, current, require_order=True)
