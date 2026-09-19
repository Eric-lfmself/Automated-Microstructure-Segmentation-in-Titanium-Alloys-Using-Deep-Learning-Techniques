"""Safe CPU checkpointing with deterministic epoch-boundary resume.

Only tensors and simple Python values are saved; ``torch.load`` always uses
``weights_only=True``. Resume covers model, optimizer, configuration, completed
epoch count, history, and CPU/Python/NumPy/DataLoader random states.
"""

import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

_FORMAT_VERSION = 1
_MUTABLE_CONFIG = {"epochs", "output_dir", "weights_path", "use_random_init", "allow_download", "checkpoint",
                   "train_manifest", "val_manifest", "test_manifest"}


def _config_dict(config):
    return config.to_dict() if hasattr(config, "to_dict") else dict(config)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def validate_checkpoint_config(payload, config):
    """Require identical data/model/loss/optimizer protocol on continuation.

    Target epoch count, output location, and unused initial-pretraining settings
    may change. Manifest paths may relocate; the runner separately requires
    immutable training/validation content hashes. Epoch count can increase for a real run, never decrease
    below the checkpoint's completed count; dry-run config caps it at one.
    """
    saved, current = _canonical(payload["config"]), _canonical(_config_dict(config))
    changed = sorted(key for key in saved.keys() | current.keys()
                     if key not in _MUTABLE_CONFIG and saved.get(key) != current.get(key))
    if changed:
        raise ValueError("Checkpoint configuration mismatch: " + ", ".join(changed))
    if current["epochs"] < payload["completed_epochs"]:
        raise ValueError("Target epochs cannot be below checkpoint completed_epochs")


def _rng_state(generators):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.random.get_rng_state(),
        "numpy": {"algorithm": numpy_state[0], "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "generators": {name: generator.get_state() for name, generator in (generators or {}).items()},
    }


def _restore_rng(state, generators):
    supplied = generators or {}
    if set(state["generators"]) != set(supplied):
        raise ValueError("DataLoader generator names must match the checkpoint")
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    source = state["numpy"]
    np.random.set_state((source["algorithm"], source["keys"].numpy().astype(np.uint32),
                         source["position"], source["has_gauss"], source["cached_gaussian"]))
    for name, generator in supplied.items():
        generator.set_state(state["generators"][name])


def save_checkpoint(path, model, optimizer, *, config, completed_epochs, history, generators=None,
                    data_provenance=None):
    """Atomically replace one checkpoint after a complete train/validation epoch."""
    if type(completed_epochs) is not int or completed_epochs < 0 or len(history) != completed_epochs:
        raise ValueError("completed_epochs must equal the history length")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Checkpoint model must be on CPU")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": _FORMAT_VERSION, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "config": _config_dict(config),
               "completed_epochs": completed_epochs, "history": history,
               "rng": _rng_state(generators), "data_provenance": data_provenance}
    descriptor, temp_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def read_checkpoint(path):
    """Read safe primitives/tensors on CPU without constructing a model."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "model", "optimizer", "config", "completed_epochs", "history", "rng", "data_provenance"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Not a complete TC4 training checkpoint")
    if payload["format_version"] != _FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format_version")
    count = payload["completed_epochs"]
    if type(count) is not int or count < 0 or not isinstance(payload["history"], list) or len(payload["history"]) != count:
        raise ValueError("Checkpoint epoch/history inconsistency")
    return payload


def load_checkpoint(path, model, optimizer=None, *, config=None, generators=None, restore_rng=True):
    """Restore a checkpoint and return its payload; evaluation can skip RNGs.

    Use ``restore_rng=False`` for inference-only loading, and omit optimizer.
    The caller must construct the model with random initialization first.
    """
    payload = read_checkpoint(path)
    return restore_checkpoint(payload, model, optimizer, config=config,
                              generators=generators, restore_rng=restore_rng)


def restore_checkpoint(payload, model, optimizer=None, *, config=None, generators=None, restore_rng=True):
    """Restore an already-read payload, avoiding duplicate large checkpoint reads."""
    if config is not None:
        validate_checkpoint_config(payload, config)
    if restore_rng and set(payload["rng"]["generators"]) != set(generators or {}):
        raise ValueError("DataLoader generator names must match the checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload["rng"], generators)
    return payload
