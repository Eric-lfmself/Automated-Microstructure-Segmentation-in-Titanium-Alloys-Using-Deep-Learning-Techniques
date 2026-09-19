"""File fingerprints for split integrity and checkpoint provenance.

We record image/mask content hashes and sample IDs with each local-data run.
These checks detect exact duplicates and content changes; specimen-level split
membership must be specified in the input manifests.
"""

import hashlib
from pathlib import Path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_provenance(datasets):
    """Fingerprint each dataset record and reject image copies across splits.

    Return ``{split: [{id, image_path, mask_path, image_sha256, mask_sha256}]}``.
    Paths are resolved for traceability; hashes are streamed over file contents.
    Different files with identical image bytes cannot straddle any two splits.
    Identical masks alone are allowed (e.g. two homogeneous but distinct images).
    """
    provenance = {}
    seen_images = {}
    for split, dataset in datasets.items():
        if not hasattr(dataset, "records"):
            raise ValueError("Provenance requires explicit real-data file records")
        records = []
        for identifier, image_path, mask_path in dataset.records:
            image_path, mask_path = Path(image_path).resolve(), Path(mask_path).resolve()
            image_hash = _sha256_file(image_path)
            previous = seen_images.get(image_hash)
            if previous is not None and previous[0] != split:
                raise ValueError(f"Split leakage: identical image content in {previous[0]} "
                                 f"({previous[1]}) and {split} ({identifier})")
            seen_images[image_hash] = (split, identifier)
            records.append({"id": str(identifier), "image_path": str(image_path),
                            "mask_path": str(mask_path), "image_sha256": image_hash,
                            "mask_sha256": _sha256_file(mask_path)})
        provenance[str(split)] = records
    return provenance


def _identities(provenance, split):
    if not isinstance(provenance, dict) or not isinstance(provenance.get(split), list):
        raise ValueError(f"Missing verified {split} data provenance")
    records = provenance[split]
    if not records:
        raise ValueError(f"Empty {split} data provenance")
    identities = []
    for record in records:
        if not isinstance(record, dict) or not {"id", "image_sha256", "mask_sha256"}.issubset(record):
            raise ValueError(f"Invalid {split} data provenance record")
        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError(f"Invalid {split} provenance ID")
        for key in ("image_sha256", "mask_sha256"):
            value = record[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {split} provenance SHA-256")
        identities.append((record["id"], record["image_sha256"], record["mask_sha256"]))
    if len({item[0] for item in identities}) != len(identities):
        raise ValueError(f"Duplicate {split} provenance IDs")
    return sorted(identities)


def assert_same_training_provenance(saved, current, *, require_order=False):
    """Require original train/validation identities and reject test leakage.

    Compare sample IDs, image bytes and label bytes; local file relocation does
    not matter. Evaluation ignores row ordering by default. Resume must set
    ``require_order=True`` because DataLoader RNG shuffles dataset indices and
    reordered rows would change the next sampled sequence. Missing saved fingerprints are a hard error
    for real-data resume/held-out claims. If current test records are supplied,
    no test ID or image hash may occur in the checkpoint's original train/val.
    """
    original_ids, original_images = set(), set()
    for split in ("train", "val"):
        reference = _identities(saved, split)
        present = _identities(current, split)
        if reference != present:
            raise ValueError(f"Training provenance mismatch: {split} IDs, images, or labels changed")
        if require_order:
            fields = ("id", "image_sha256", "mask_sha256")
            saved_order = [tuple(record[key] for key in fields) for record in saved[split]]
            current_order = [tuple(record[key] for key in fields) for record in current[split]]
            if saved_order != current_order:
                raise ValueError(f"Training provenance order mismatch: {split} record order changed")
        original_ids.update(record[0] for record in reference)
        original_images.update(record[1] for record in reference)
    if isinstance(current, dict) and "test" in current:
        for identifier, image_hash, _ in _identities(current, "test"):
            if identifier in original_ids or image_hash in original_images:
                raise ValueError("Test leakage: sample matches the checkpoint's original train/validation data")
