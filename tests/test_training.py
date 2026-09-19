"""Bounded CPU smoke fixtures; no full-model or real-data training."""

import csv
from dataclasses import replace
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from losses import make_weighted_cross_entropy
from training.checkpoint import save_checkpoint, load_checkpoint, read_checkpoint, validate_checkpoint_config
from training.engine import run_epoch, seed_everything
from training.runner import run_training, validate_split_manifests


def _tiny_model():
    return nn.Sequential(nn.Conv2d(3, 5, 1), nn.Dropout2d(.2)).to(device="cpu")


def _samples():
    generator = torch.Generator(device="cpu").manual_seed(5)
    return [{"image": torch.rand((3, 32, 32), generator=generator),
             "mask": torch.full((32, 32), i % 5, dtype=torch.long)} for i in range(4)]


def test_epoch_loss_uses_global_weight_mass_not_batch_mean():
    model = nn.Conv2d(3, 5, 1).to(device="cpu")
    with torch.no_grad():
        model.weight.fill_(.1)
        model.bias.copy_(torch.tensor([2., 1., 0., -1., -2.]))
    first = {"image": torch.zeros((2, 3, 32, 32)), "mask": torch.zeros((2, 32, 32), dtype=torch.long)}
    second = {"image": torch.ones((1, 3, 32, 32)), "mask": torch.full((1, 32, 32), 4, dtype=torch.long)}
    criterion = make_weighted_cross_entropy((1, 2, 3, 4, 20))
    result = run_epoch(model, [first, second], criterion)
    images = torch.cat([first["image"], second["image"]])
    targets = torch.cat([first["mask"], second["mask"]])
    expected = F.cross_entropy(model(images), targets, weight=criterion.weight)
    assert result["loss"] == pytest.approx(float(expected.detach()), rel=1e-6)
    batch_mean = sum(float(criterion(model(b["image"]), b["mask"]).detach()) for b in (first, second)) / 2
    assert abs(result["loss"] - batch_mean) > .1
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)


def test_ignored_batches_do_not_step_optimizer():
    model = _tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    batch = {"image": torch.zeros((2, 3, 32, 32)), "mask": torch.full((2, 32, 32), 255)}
    with pytest.raises(ValueError, match="no valid labeled pixels"):
        run_epoch(model, [batch], make_weighted_cross_entropy(), optimizer=optimizer)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert not optimizer.state


def test_checkpoint_restores_optimizer_rng_and_exact_continuation(tmp_path):
    seed_everything(82)
    generator = torch.Generator(device="cpu").manual_seed(123)
    loader = DataLoader(_samples(), batch_size=2, shuffle=True, generator=generator)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    criterion = make_weighted_cross_entropy()
    run_epoch(model, loader, criterion, optimizer=optimizer, max_batches=1)
    config = Config()
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, config=config, completed_epochs=1,
                    history=[{"epoch": 1}], generators={"train": generator})
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    reference = run_epoch(model, loader, criterion, optimizer=optimizer, max_batches=1)
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    restored = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_generator = torch.Generator(device="cpu").manual_seed(999)
    payload = load_checkpoint(checkpoint, restored, restored_optimizer, config=config,
                              generators={"train": restored_generator})
    assert payload["completed_epochs"] == 1
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(3), expected_random[2])
    restored_loader = DataLoader(_samples(), batch_size=2, shuffle=True, generator=restored_generator)
    actual = run_epoch(restored, restored_loader, criterion, optimizer=restored_optimizer, max_batches=1)
    assert actual["loss"] == reference["loss"]
    for key, value in restored.state_dict().items():
        assert torch.equal(value, expected_state[key])
    for original, recovered in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for key in original:
            torch.testing.assert_close(original[key], recovered[key], rtol=0, atol=0)


def test_checkpoint_protocol_mismatch_is_explicit(tmp_path):
    model = _tiny_model()
    path = tmp_path / "checkpoint.pt"
    config = Config()
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), config=config,
                    completed_epochs=1, history=[{"epoch": 1}])
    payload = read_checkpoint(path)
    with pytest.raises(ValueError, match="class_weights"):
        validate_checkpoint_config(payload, replace(config, class_weights=(1, 2, 3, 4, 5)))
    with pytest.raises(ValueError, match="below"):
        validate_checkpoint_config(payload, replace(config, epochs=0))
    validate_checkpoint_config(payload, replace(config, output_dir="different"))


def test_real_gate_and_output_conflict_fail_before_model_loading(tmp_path):
    def forbidden_factory(*args, **kwargs):
        raise AssertionError("Should not construct a model")
    with pytest.raises(ValueError, match="explicit --run-real"):
        run_training(replace(Config(), dry_run=False), model_factory=forbidden_factory)
    (tmp_path / "existing.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not empty"):
        run_training(replace(Config(), output_dir=str(tmp_path)), model_factory=forbidden_factory)
    assert (tmp_path / "existing.txt").read_text() == "keep me"


def _write_split(tmp_path, split, identifier, image_path=None):
    image_path = image_path or tmp_path / f"{split}_image.png"
    if not image_path.exists():
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(image_path)
    mask_path = tmp_path / f"{split}_mask.png"
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(mask_path)
    manifest = tmp_path / f"{split}.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "image", "mask"])
        writer.writerow([identifier, image_path, mask_path])
    return str(manifest)


def test_split_leakage_ids_and_resolved_paths(tmp_path):
    train = _write_split(tmp_path, "train", "specimen_1")
    val = _write_split(tmp_path, "val", "specimen_1")
    config = replace(Config(), dry_run=False, train_manifest=train, val_manifest=val)
    with pytest.raises(ValueError, match="ID"):
        validate_split_manifests(config)
    alias = tmp_path / "alias.png"
    alias.symlink_to(tmp_path / "train_image.png")
    val = _write_split(tmp_path, "val", "specimen_2", image_path=alias)
    with pytest.raises(ValueError, match="file"):
        validate_split_manifests(replace(config, val_manifest=val))


def test_dry_run_checkpoint_resume_does_not_exceed_one_update(tmp_path):
    calls = []
    def factory(config, *, force_random=False):
        calls.append(force_random)
        return _tiny_model()
    config = replace(Config(), output_dir=str(tmp_path / "run"))
    result = run_training(config, model_factory=factory)
    assert result["completed_epochs"] == 1
    assert len(result["history"]) == 1
    assert calls == [False]
    path = tmp_path / "run" / "checkpoint.pt"
    before = path.read_bytes()
    resumed = run_training(config, resume=path, model_factory=factory)
    assert resumed["status"] == "already_complete"
    assert calls == [False]
    assert path.read_bytes() == before
    for name in ("config.json", "history.json", "history.csv", "curves.png"):
        assert (tmp_path / "run" / name).is_file()


def test_conflicting_resume_history_is_rejected_before_model_loading(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    config = replace(Config(), output_dir=str(output))
    model = _tiny_model()
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), config=config,
                    completed_epochs=1, history=[{"epoch": 1, "train_loss": 1.0}])
    previous = [{"epoch": 1, "train_loss": 99.0}]
    (output / "history.json").write_text(json.dumps(previous))
    def forbidden_factory(*args, **kwargs):
        raise AssertionError("Should reject conflicting history before model loading")
    with pytest.raises(ValueError, match="history does not match"):
        run_training(config, resume=checkpoint, model_factory=forbidden_factory)
    assert json.loads((output / "history.json").read_text()) == previous


def test_invalid_seed_is_rejected_before_model_loading(tmp_path):
    def forbidden_factory(*args, **kwargs):
        raise AssertionError("Should reject invalid seed before model loading")
    with pytest.raises(ValueError, match="seed must"):
        run_training(replace(Config(), seed=-1, output_dir=str(tmp_path)), model_factory=forbidden_factory)

@pytest.mark.parametrize('filename,contents', [
    ('history.csv', 'unrelated CSV: preserve this'),
    ('curves.png', 'unrelated figure: preserve this'),
    ('history.json', '[]'),
])
def test_resume_rejects_orphan_named_outputs_before_model(tmp_path, filename, contents):
    output = tmp_path / 'unrelated'; output.mkdir()
    path = output / filename; path.write_text(contents)
    config = replace(Config(), image_size=(32, 32), dummy_length=2, output_dir=str(output))
    model = _tiny_model()
    checkpoint = tmp_path / 'resume.pt'
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), config=config,
                    completed_epochs=0, history=[])
    def forbidden(*args, **kwargs):
        raise AssertionError('Must fail before constructing a model')
    with pytest.raises(ValueError, match='matching checkpoint.*ownership'):
        run_training(config, resume=checkpoint, model_factory=forbidden)
    assert path.read_text() == contents
    assert list(output.iterdir()) == [path]


@pytest.mark.parametrize('interruption', ['before_export', 'partial_json', 'partial_csv', 'stale_exports'])
def test_completed_resume_repairs_interrupted_exports_without_update(tmp_path, monkeypatch, interruption):
    import training.runner as runner
    output = tmp_path / 'run'
    config = replace(Config(), image_size=(32, 32), dummy_length=2, output_dir=str(output))
    original_save = runner._save_history
    def interrupted(history, destination):
        if interruption == 'partial_json':
            (destination / '.history.json.tmp').write_text('[broken')
        elif interruption == 'partial_csv':
            (destination / 'history.json').write_text(json.dumps(history))
            (destination / '.history.csv.tmp').write_text('partial')
        elif interruption == 'stale_exports':
            (destination / 'history.json').write_text('[]')
            (destination / 'history.csv').write_text('old CSV')
            (destination / 'curves.png').write_text('old figure')
        raise OSError('simulated interruption after checkpoint')
    monkeypatch.setattr(runner, '_save_history', interrupted)
    with pytest.raises(OSError, match='simulated interruption'):
        run_training(config, model_factory=lambda *a, **k: _tiny_model())
    checkpoint = output / 'checkpoint.pt'
    before = checkpoint.read_bytes()
    expected = read_checkpoint(checkpoint)['history']
    monkeypatch.setattr(runner, '_save_history', original_save)
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not construct or train a model')
    result = run_training(config, resume=checkpoint, model_factory=forbidden)
    assert result['status'] == 'already_complete'
    assert result['history'] == expected
    assert checkpoint.read_bytes() == before
    assert json.loads((output / 'history.json').read_text()) == expected
    with (output / 'history.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and int(rows[0]['epoch']) == 1
    with Image.open(output / 'curves.png') as figure:
        figure.verify()
    assert not list(output.glob('.history.*.tmp'))
    fresh = tmp_path / 'fresh'
    result = run_training(replace(config, output_dir=str(fresh)), resume=checkpoint, model_factory=forbidden)
    assert result['status'] == 'already_complete' and not fresh.exists()


@pytest.mark.parametrize('mask,error', [
    (torch.full((1, 2, 2), .9), TypeError),
    (torch.full((1, 2, 2), -.9), TypeError),
    (torch.ones(1, 2, 2, dtype=torch.bool), TypeError),
    (torch.full((1, 2, 2), -1, dtype=torch.int8), ValueError),
    (torch.full((1, 2, 2), 5), ValueError),
    (torch.zeros(1, 3, 2, dtype=torch.long), ValueError),
])
def test_invalid_training_labels_cannot_forward_clear_gradients_or_update(mask, error):
    model = nn.Conv2d(3, 5, 1).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters())
    original = {key: value.clone() for key, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7)
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append('forward'))
    with pytest.raises(error, match='mask'):
        run_epoch(model, [{'image': torch.zeros(1, 3, 2, 2), 'mask': mask}],
                  make_weighted_cross_entropy(), optimizer=optimizer)
    handle.remove()
    assert not calls and not optimizer.state
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert all(torch.all(parameter.grad == 7) for parameter in model.parameters())


def test_nonfinite_gradients_never_reach_optimizer_step():
    model = nn.Conv2d(3, 5, 1).to('cpu')
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters())
    batch = {'image': torch.ones(1, 3, 32, 32), 'mask': torch.zeros(1, 32, 32, dtype=torch.long)}
    with pytest.raises(ValueError, match='Nonfinite gradients'):
        run_epoch(model, [batch], make_weighted_cross_entropy((1e-44, 1, 1, 1, 1)), optimizer=optimizer)
    assert not optimizer.state
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_rejects_non_cpu_buffers_without_using_gpu():
    model = nn.Conv2d(3, 5, 1).to('cpu')
    model.register_buffer('placeholder', torch.empty(0, device='meta'))
    with pytest.raises(ValueError, match='CPU model'):
        run_epoch(model, [], make_weighted_cross_entropy())


def test_training_allows_custom_ignore_without_changing_labels():
    model = nn.Conv2d(3, 5, 1).to('cpu')
    batch = {'image': torch.zeros(1, 3, 2, 2),
             'mask': torch.tensor([[[0, -100], [1, -100]]], dtype=torch.int16)}
    result = run_epoch(model, [batch], make_weighted_cross_entropy(ignore_index=-100))
    assert result['valid_pixels'] == 2


@pytest.mark.parametrize("scalar", [np.float32, np.float64])
@pytest.mark.parametrize("text_scalar", [str, np.str_])
def test_scientific_config_numbers_roundtrip_through_run_and_checkpoint(tmp_path, scalar, text_scalar):
    from configs import load_config
    config = Config(image_size=torch.Size((32, 32)), learning_rate=scalar(.001), dropout=scalar(.1),
                    model=text_scalar("unet_vgg"), device=text_scalar("cpu"),
                    checkpoint=text_scalar("nvidia/mit-b0"),
                    train_manifest=text_scalar("unused/train.csv"), val_manifest=text_scalar("unused/val.csv"),
                    class_weights=tuple(np.arange(1, 6, dtype=scalar)),
                    sigmas=tuple(np.arange(1, 5, dtype=np.int64)),
                    iw_weights=tuple(np.full(4, .5, dtype=scalar)),
                    output_dir=text_scalar(str(tmp_path / "run"))).validate()
    result = run_training(config, model_factory=lambda *args, **kwargs: _tiny_model())
    assert result["status"] == "complete"
    output = tmp_path / "run"
    assert load_config(output / "config.json") == config
    # This strict loader, rather than unsafe pickle loading, must read every
    # accepted field in both Config and the optimizer's parameter groups.
    payload = read_checkpoint(output / "checkpoint.pt")
    assert payload["completed_epochs"] == 1
    assert payload["optimizer"]["param_groups"][0]["lr"] == float(scalar(.001))
    def no_new_model(*args, **kwargs):
        raise AssertionError("Completed resume must not construct or update a model")
    resumed = run_training(config, resume=output / "checkpoint.pt", model_factory=no_new_model)
    assert resumed["status"] == "already_complete"
