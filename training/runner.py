"""Guarded orchestration for the common training template, paper Section 3.1."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import load_config
from data import DummyTC4Dataset, TC4Dataset, PairedAugment
from data.provenance import collect_provenance, assert_same_training_provenance
from losses import make_weighted_cross_entropy

from .checkpoint import read_checkpoint, validate_checkpoint_config, restore_checkpoint, save_checkpoint
from .engine import run_epoch, seed_everything
from .plotting import plot_history


def validate_split_manifests(config):
    """Construct explicit local splits and reject identity/path leakage.

    Requires train and validation manifests, checks test if supplied. Resolved
    paths catch symlink aliases and image/mask reuse across any split. Different
    patches from a shared parent specimen still require user-controlled grouping;
    without specimen metadata that relationship cannot be inferred reliably.
    """
    if not config.train_manifest or not config.val_manifest:
        raise ValueError("Real execution requires explicit train_manifest and val_manifest")
    datasets = {}
    seen_ids, seen_paths = {}, {}
    for split in ("train", "val", "test"):
        manifest = getattr(config, f"{split}_manifest")
        if manifest is None:
            continue
        dataset = TC4Dataset(manifest, image_size=config.image_size, seed=config.seed,
                             sigmas=config.sigmas, weights=config.iw_weights,
                             label_offset=config.label_offset,
                             transform=PairedAugment() if split == "train" else None)
        for identifier, image_path, mask_path in dataset.records:
            if identifier in seen_ids:
                raise ValueError(f"Split leakage: ID {identifier!r} reused in {seen_ids[identifier]} and {split}")
            seen_ids[identifier] = split
            for path in (image_path, mask_path):
                if path in seen_paths:
                    raise ValueError(f"Split leakage: file {path} reused in {seen_paths[path]} and {split}")
                seen_paths[path] = split
        datasets[split] = dataset
    return datasets


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _validate_output(output, resume_payload, resume_path=None):
    """Fail before model creation if new work would overwrite previous output."""
    if not output.exists():
        return False
    if not output.is_dir():
        raise ValueError("output_dir exists and is not a directory")
    if not any(output.iterdir()):
        return False
    if resume_payload is None:
        raise ValueError("output_dir is not empty; choose a new directory or explicitly --resume")
    known = {"config.json", "checkpoint.pt", "history.json", "history.csv", "curves.png",
             "resume_requests.jsonl", ".history.json.tmp", ".history.csv.tmp"}
    if any(item.name not in known for item in output.iterdir()):
        raise ValueError("output_dir contains unrelated files; choose an empty resume destination")
    if any(item.is_symlink() or not item.is_file() for item in output.iterdir()):
        raise ValueError("Resume output must contain regular files, not links or directories")
    config_path = output / "config.json"
    if config_path.exists():
        recorded = dict(resume_payload, config=json.loads(config_path.read_text()))
        validate_checkpoint_config(recorded, resume_payload["config"])
    history_path = output / "history.json"
    if history_path.exists():
        previous = json.loads(history_path.read_text())
        expected = resume_payload["history"]
        if not isinstance(previous, list) or previous != expected[:len(previous)]:
            raise ValueError("Existing output history does not match the resume checkpoint")
    existing_checkpoint = output / "checkpoint.pt"
    if not existing_checkpoint.is_file():
        raise ValueError("Nonempty resume output requires a matching checkpoint to establish ownership")
    same_file = resume_path is not None and existing_checkpoint.resolve() == Path(resume_path).resolve()
    if not same_file and (resume_path is None or
                         _file_digest(existing_checkpoint) != _file_digest(Path(resume_path))):
        raise ValueError("Existing checkpoint differs from requested resume checkpoint")
    return True


def _save_history(history, output):
    temporary = output / ".history.json.tmp"
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "history.json")
    temporary = output / ".history.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "history.csv")


def run_training(config, *, run_real=False, resume=None, model_factory=None):
    """Run the guarded template; default config executes one tiny CPU update.

    ``run_real`` is a separate explicit permission gate, so a config file alone
    cannot silently start private-data training. A completed dry-run resume
    returns immediately, rather than performing another epoch.
    """
    config.validate()
    if not config.dry_run and not run_real:
        raise ValueError("Real-data training requires explicit --run-real authorization")
    if config.dry_run and run_real:
        raise ValueError("run_real cannot be combined with dry_run=True")
    output = Path(config.output_dir).expanduser().resolve()
    payload = read_checkpoint(resume) if resume is not None else None
    if payload is not None:
        validate_checkpoint_config(payload, config)
    owned_output = _validate_output(output, payload, resume)
    if config.dry_run:
        common = dict(length=config.dummy_length, image_size=config.image_size,
                      sigmas=config.sigmas, weights=config.iw_weights)
        datasets = {"train": DummyTC4Dataset(seed=config.seed, transform=PairedAugment(), **common),
                    "val": DummyTC4Dataset(seed=config.seed + 1, **common)}
        data_provenance = None
    else:
        datasets = validate_split_manifests(config)
        data_provenance = collect_provenance(datasets)
        if payload is not None:
            assert_same_training_provenance(payload["data_provenance"], data_provenance, require_order=True)
    completed = payload["completed_epochs"] if payload is not None else 0
    history = payload["history"] if payload is not None else []
    if completed == config.epochs:
        # Checkpoint is authoritative after an interrupted export. Repair only
        # an existing, positively identified run; a fresh destination stays empty.
        if owned_output:
            _save_history(history, output)
            plot_history(history, output / "curves.png", model_name=config.model, synthetic=config.dry_run)
        return {"status": "already_complete", "completed_epochs": completed,
                "dry_run": config.dry_run, "history": history, "output_dir": str(output)}
    seed_everything(config.seed)
    generators = {name: torch.Generator(device="cpu").manual_seed(config.seed + index)
                  for index, name in enumerate(("train", "val"))}
    loaders = {name: DataLoader(datasets[name], batch_size=config.batch_size,
                               shuffle=name == "train", num_workers=0,
                               generator=generators[name], persistent_workers=False)
               for name in ("train", "val")}
    if model_factory is None:
        from models.factory import create_model
        model_factory = create_model
    model = model_factory(config, force_random=resume is not None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    criterion = make_weighted_cross_entropy(config.class_weights)
    if resume is not None:
        restore_checkpoint(payload, model, optimizer, config=config, generators=generators)
        history = list(payload["history"])
        del payload
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if not config_path.exists():
        with config_path.open("x") as stream:
            json.dump(config.to_dict(), stream, indent=2)
            stream.write("\n")
    if resume is not None:
        with (output / "resume_requests.jsonl").open("a") as stream:
            stream.write(json.dumps({"resume": str(Path(resume).resolve()), "config": config.to_dict()}) + "\n")
    for epoch in range(completed, config.epochs):
        datasets["train"].set_epoch(epoch)
        train = run_epoch(model, loaders["train"], criterion, optimizer=optimizer, max_batches=config.max_batches)
        val = run_epoch(model, loaders["val"], criterion, max_batches=config.max_batches)
        history.append({"epoch": epoch + 1, "train_loss": train["loss"], "val_loss": val["loss"],
                        "train_pixel_accuracy": train["pixel_accuracy"], "val_pixel_accuracy": val["pixel_accuracy"],
                        "train_batches": train["batches"], "val_batches": val["batches"],
                        "train_seconds": train["seconds"], "val_seconds": val["seconds"],
                        "synthetic": config.dry_run})
        save_checkpoint(output / "checkpoint.pt", model, optimizer, config=config,
                        completed_epochs=epoch + 1, history=history, generators=generators,
                        data_provenance=data_provenance)
        _save_history(history, output)
        plot_history(history, output / "curves.png", model_name=config.model, synthetic=config.dry_run)
    return {"status": "complete", "completed_epochs": config.epochs, "dry_run": config.dry_run,
            "history": history, "output_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared CPU template; defaults to a bounded dummy smoke test")
    parser.add_argument("--config")
    parser.add_argument("--model", choices=("unet_vgg", "segformer_b0"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-real", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        config = config.as_dry_run()
    elif args.run_real:
        config = replace(config, dry_run=False)
    if args.model is not None:
        config = replace(config, model=args.model)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    result = run_training(config.validate(), run_real=args.run_real, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
