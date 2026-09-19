"""SegFormer-B0 for our paper Section 2.3.2 and Figure 3.

We use transformers SegformerForSemanticSegmentation with the MiT-B0 encoder
and all-MLP decode head. Tested versions: transformers 4.57.1 and PyTorch 2.8.0.
The optional ImageNet encoder checkpoint is nvidia/mit-b0; the five-class
segmentation head initializes separately. Overlapping patch embeddings and
Mix-FFN include convolutions, as in the standard MiT implementation.

The public implementation follows the MLP design in Section 2.3.2 and Figure 3.
See docs/METHODS.md for architecture and configuration details.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SegFormerB0(nn.Module):
    """Expose the same full-resolution logits interface as ``VGGResUNet``.

    Random initialization builds a local config and makes no Hub request.
    Otherwise, a supplied encoder ``weights_path`` may be a local Hugging Face
    model directory or a raw ``SegformerModel.state_dict()`` file. Without a
    local path, ``nvidia/mit-b0`` is resolved from cache only unless the caller
    explicitly opts into ``allow_download=True``. ``pretrained_loader`` replaces
    ``SegformerModel.from_pretrained`` and receives ``(source,
    local_files_only=...)``, making pretrained flow testable offline. Injected
    loaders own weight completeness; the built-in HF loader checks diagnostics
    and rejects missing encoder parameters rather than random-filling them.
    """

    def __init__(
        self,
        num_classes: int = 5,
        use_random_init: bool = True,
        checkpoint: str = "nvidia/mit-b0",
        allow_download: bool = False,
        weights_path: str | Path | None = None,
        dropout: float = 0.1,
        pretrained_loader: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if use_random_init and (weights_path is not None or pretrained_loader is not None):
            raise ValueError("Set use_random_init=False to load supplied pretrained weights")
        if weights_path is not None and pretrained_loader is not None:
            raise ValueError("Choose weights_path or pretrained_loader, not both")
        if weights_path is not None:
            weights_path = Path(weights_path).expanduser()
        # Lazy import keeps optional-transformer absence from affecting U-Net.
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

        config = SegformerConfig(
            num_channels=3,
            num_labels=num_classes,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            decoder_hidden_size=256,
            classifier_dropout_prob=dropout,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            reshape_last_stage=True,
        )
        self.model = SegformerForSemanticSegmentation(config)
        self.num_classes = num_classes
        self.weights_source = "random"
        if not use_random_init:
            if weights_path is not None and not Path(weights_path).exists():
                raise FileNotFoundError(f"Local encoder checkpoint does not exist: {weights_path}")
            if weights_path is not None and Path(weights_path).is_file():
                state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
                self.model.segformer.load_state_dict(state, strict=True)
                self.weights_source = str(weights_path)
            else:
                source = str(weights_path) if weights_path is not None else checkpoint
                if pretrained_loader is not None:
                    encoder = pretrained_loader(source, local_files_only=not allow_download)
                else:
                    encoder, info = SegformerModel.from_pretrained(
                        source, local_files_only=not allow_download, output_loading_info=True)
                    self._validate_loading_info(info)
                self._validate_encoder(encoder, config)
                encoder.train(self.training)
                self.model.segformer = encoder
                self.weights_source = source

    @staticmethod
    def _validate_loading_info(info) -> None:
        """Allow an unused ImageNet classifier, never incomplete encoder weights."""
        unexpected = set(info.get("unexpected_keys", ())) - {"classifier.weight", "classifier.bias"}
        if (info.get("missing_keys") or info.get("mismatched_keys")
                or info.get("error_msgs") or unexpected):
            raise ValueError(f"Incomplete or incompatible pretrained encoder weights: {info}")

    @staticmethod
    def _validate_encoder(encoder: nn.Module, expected) -> None:
        """Require MiT-B0 shapes and behavior so state_dict resume rebuilds the same network."""
        from transformers import SegformerModel

        if not isinstance(encoder, SegformerModel):
            raise TypeError("pretrained_loader must return a Hugging Face SegformerModel")
        for field in ("num_channels", "num_encoder_blocks", "hidden_sizes", "depths", "num_attention_heads",
                      "sr_ratios", "patch_sizes", "strides", "mlp_ratios", "reshape_last_stage",
                      "hidden_act", "hidden_dropout_prob", "attention_probs_dropout_prob", "drop_path_rate"):
            if getattr(encoder.config, field, None) != getattr(expected, field):
                raise ValueError(f"Pretrained encoder does not match MiT-B0: {field}")

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [N, 3, H, W]")
        if min(images.shape[-2:]) < 32:
            raise ValueError("SegFormerB0 requires image dimensions >= 32")
        logits = self.model(pixel_values=images, return_dict=True).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
