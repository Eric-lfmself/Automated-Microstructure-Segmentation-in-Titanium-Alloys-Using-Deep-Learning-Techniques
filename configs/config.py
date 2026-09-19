"""Configuration for our preprocessing, models and shared training pipeline.

The source-resolution setting is 1024x1024. Default execution uses a small,
offline CPU validation run; local-data training requires an explicit flag.
Release defaults are documented in docs/METHODS.md.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from numbers import Real
from pathlib import Path


def _native_real(value):
    """Keep accepted scientific scalars safe for JSON and weights-only checkpoints.

    NumPy float64 can look like a Python float to isinstance and JSON while
    retaining an unsafe NumPy pickle type. Canonicalize the actual field values
    before they can reach an optimizer, not only their exported dictionary.
    Leave booleans and unsupported values for the existing strict validation.
    """
    if isinstance(value, Real) and type(value) not in (int, float, bool):
        try:
            return float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Configuration numbers must be finite Python float values") from exc
    return value


def _finite_number(value, name):
    """Reject booleans as numbers and report malformed JSON fields explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain finite numbers, not booleans")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must contain finite numbers")


def _number_sequence(values, count, name, *, allow_zero=False):
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} numbers")
    for value in values:
        _finite_number(value, name)
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must contain {'nonnegative' if allow_zero else 'positive'} values")


@dataclass(frozen=True)
class Config:
    model: str = "unet_vgg"
    num_classes: int = 5
    source_image_size: tuple = (1024, 1024)
    image_size: tuple = (64, 64)
    device: str = "cpu"
    dry_run: bool = True
    use_random_init: bool = True
    allow_download: bool = False
    weights_path: str | None = None
    checkpoint: str = "nvidia/mit-b0"
    sigmas: tuple = (1., 2., 3., 4.)
    iw_weights: tuple = (.5, .5, .5, .5)
    class_weights: tuple = (1., 1., 1., 1., 1.)
    dropout: float = .1
    batch_size: int = 2
    dummy_length: int = 4
    epochs: int = 1
    max_batches: int | None = 1
    learning_rate: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    train_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    label_offset: int = 0
    output_dir: str = "runs/dry_run"

    def __post_init__(self):
        # NumPy strings are JSON-compatible str subclasses but require unsafe
        # pickle globals unless their actual field values become native str.
        for name, value in vars(self).items():
            if isinstance(value, str) and type(value) is not str:
                object.__setattr__(self, name, str.__str__(value))
        for name in ("image_size", "source_image_size"):
            value = getattr(self, name)
            if isinstance(value, (tuple, list)):
                object.__setattr__(self, name, tuple(value))
        # Integer-only counts/seed/dimension entries retain their strict types.
        for name in ("learning_rate", "dropout"):
            object.__setattr__(self, name, _native_real(getattr(self, name)))
        for name in ("sigmas", "iw_weights", "class_weights"):
            values = getattr(self, name)
            if isinstance(values, (tuple, list)):
                object.__setattr__(self, name, tuple(_native_real(value) for value in values))

    def validate(self):
        if self.model not in ("unet_vgg", "segformer_b0"):
            raise ValueError("model must be unet_vgg or segformer_b0")
        if self.device != "cpu":
            raise ValueError("This implementation requires device='cpu'")
        if type(self.num_classes) is not int or self.num_classes != 5:
            raise ValueError("Paper defines exactly five semantic classes, no beta class")
        for name in ("image_size", "source_image_size"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int or v < 32 for v in value):
                raise ValueError(f"{name} must contain two integers >=32")
        for name in ("batch_size", "dummy_length", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValueError("Use num_workers=0 for deterministic local CPU tests")
        if self.max_batches is not None and (type(self.max_batches) is not int or self.max_batches < 1):
            raise ValueError("max_batches must be positive or null")
        _number_sequence(self.sigmas, 4, "sigmas")
        if any(a >= b for a, b in zip(self.sigmas, self.sigmas[1:])):
            raise ValueError("sigmas must contain four strictly increasing positive values")
        _number_sequence(self.iw_weights, 4, "iw_weights", allow_zero=True)
        _number_sequence(self.class_weights, 5, "class_weights")
        _finite_number(self.learning_rate, "learning_rate")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        _finite_number(self.dropout, "dropout")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if type(self.label_offset) is not int or self.label_offset not in (0, 1):
            raise ValueError("label_offset must be the integer 0 or 1")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32 - 2:
            raise ValueError("seed must be an integer in [0, 2**32 - 2)")
        for name in ("output_dir", "checkpoint", "weights_path", "train_manifest", "val_manifest", "test_manifest"):
            value = getattr(self, name)
            if value is None and name not in ("output_dir", "checkpoint"):
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string" +
                                 (" or null" if name not in ("output_dir", "checkpoint") else ""))
        for name in ("dry_run", "use_random_init", "allow_download"):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.dry_run:
            if not self.use_random_init or self.allow_download or self.weights_path:
                raise ValueError("dry-run must use random initialization without downloads or weights_path")
            if self.batch_size > 4 or max(self.image_size) > 128 or self.dummy_length > 8 or self.epochs != 1 or self.max_batches != 1:
                raise ValueError("dry-run capped at batch 4, 128x128, 8 dummy images, 1 epoch, 1 batch")
        return self

    def to_dict(self):
        return asdict(self)

    def as_dry_run(self):
        return replace(self, dry_run=True, device="cpu", use_random_init=True,
                       allow_download=False, weights_path=None, image_size=(64,64),
                       batch_size=2, dummy_length=4, epochs=1, max_batches=1,
                       num_workers=0).validate()


def load_config(path=None):
    """Read a strict JSON config; unknown keys fail instead of being ignored."""
    if path is None:
        return Config().validate()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration JSON must be an object")
    for key in ("image_size", "source_image_size", "sigmas", "iw_weights", "class_weights"):
        if key in values:
            if not isinstance(values[key], (list, tuple)):
                raise ValueError(f"{key} must be a JSON array")
            values[key] = tuple(values[key])
    return Config(**values).validate()
