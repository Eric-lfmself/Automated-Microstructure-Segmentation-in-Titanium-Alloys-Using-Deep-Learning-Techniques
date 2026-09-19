"""VGG16-based residual U-Net for our paper Section 2.3.1 and Figure 2.

We retain the 13 VGG16 convolution shapes and use Conv-BN-ReLU residual
blocks, 1x1 projection shortcuts and Dropout2d. Five pooling stages are paired
with transpose-convolution upsampling and encoder skip connections.

ImageNet initialization transfers torchvision VGG16 IMAGENET1K_V1 convolution
weights; normalization, projection and decoder layers initialize separately.
We construct only the convolutional features, without the VGG classifier.
Local state dictionaries and an injected pretrained_loader support offline use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VGG16_FEATURE_INDICES = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
PretrainedLoader = Callable[[], Mapping[str, Tensor]]


class ResidualConvBlock(nn.Module):
    """VGG-shaped convolutions plus BN, a projection shortcut and dropout."""

    def __init__(self, in_channels: int, out_channels: int, depth: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1)
            for i in range(depth)
        )
        self.norms = nn.ModuleList(nn.BatchNorm2d(out_channels) for _ in range(depth))
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                               nn.BatchNorm2d(out_channels))
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x))
            if index + 1 < len(self.convs):
                x = F.relu(x, inplace=False)
        return self.dropout(F.relu(x + residual, inplace=False))


class DecoderStage(nn.Module):
    """Double spatial resolution, concatenate a VGG skip, then refine it."""

    def __init__(self, in_channels: int, skip_channels: int, dropout: float):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, skip_channels, 2, stride=2)
        self.block = ResidualConvBlock(2 * skip_channels, skip_channels, 2, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.block(torch.cat((self.upsample(x), skip), dim=1))


class VGGResUNet(nn.Module):
    """Three-channel images -> full-resolution, unnormalized class logits.

    ``use_random_init=True`` is the safe default. Set it to False and supply a
    local torchvision VGG16 ``weights_path`` or a no-argument callable returning
    its state dict. Explicit ``allow_download=True`` permits torchvision's
    ``VGG16_Weights.IMAGENET1K_V1`` downloader. No automatic device selection is
    performed; callers use CPU explicitly. Input values should be ImageNet
    normalized by the shared data pipeline, especially for pretrained weights.

    Odd image sizes are padded on the bottom/right to a multiple of 32, then
    cropped back after decoding. Minimum input size is 32x32. BN training needs
    more than one value per channel in the usual way.
    """

    def __init__(
        self,
        num_classes: int = 5,
        use_random_init: bool = True,
        dropout: float = 0.1,
        weights_path: str | Path | None = None,
        allow_download: bool = False,
        pretrained_loader: PretrainedLoader | None = None,
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
        self.num_classes = num_classes
        channels = (64, 128, 256, 512, 512)
        depths = (2, 2, 3, 3, 3)
        self.encoder = nn.ModuleList(
            ResidualConvBlock(3 if i == 0 else channels[i - 1], width, depths[i], dropout)
            for i, width in enumerate(channels)
        )
        self.pools = nn.ModuleList(nn.MaxPool2d(2, stride=2) for _ in channels)
        self.decoder = nn.ModuleList(
            DecoderStage(in_width, skip_width, dropout)
            for in_width, skip_width in zip((512, 512, 512, 256, 128), reversed(channels))
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.weights_source = "random"
        if not use_random_init:
            if pretrained_loader is not None:
                state_dict = pretrained_loader()
                self.weights_source = "injected VGG16 loader"
            elif weights_path is not None:
                state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
                self.weights_source = str(weights_path)
            elif allow_download:
                # Lazy import: offline construction does not need torchvision.
                from torchvision.models import VGG16_Weights
                state_dict = VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
                self.weights_source = "torchvision:VGG16_Weights.IMAGENET1K_V1"
            else:
                raise ValueError("Pretrained VGG16 requires local weights, a loader, or allow_download=True")
            self.load_vgg16_convolutions(state_dict)

    def load_vgg16_convolutions(self, state_dict: Mapping[str, Tensor]) -> None:
        """Validate and copy all 13 official VGG16 convolutions atomically.

        Accept full torchvision model keys (``features.0.weight``) or feature
        module keys (``0.weight``). Classification weights, if present, are
        ignored. A trainer checkpoint of this Res-UNet is loaded with its normal
        ``load_state_dict`` method instead.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("VGG16 pretrained loader must return a state dictionary")
        transfers: list[tuple[Tensor, Tensor]] = []
        convs = [conv for block in self.encoder for conv in block.convs]
        for feature_index, conv in zip(VGG16_FEATURE_INDICES, convs):
            for parameter_name in ("weight", "bias"):
                key = f"features.{feature_index}.{parameter_name}"
                feature_key = f"{feature_index}.{parameter_name}"
                source = state_dict.get(key, state_dict.get(feature_key))
                target = getattr(conv, parameter_name)
                if not isinstance(source, Tensor) or source.shape != target.shape:
                    raise ValueError(f"Missing or incompatible VGG16 parameter: {key}")
                transfers.append((target, source))
        with torch.no_grad():
            for target, source in transfers:
                target.copy_(source)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [N, 3, H, W]")
        height, width = images.shape[-2:]
        if min(height, width) < 32:
            raise ValueError("VGGResUNet requires image dimensions >= 32")
        x = F.pad(images, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        skips = []
        for block, pool in zip(self.encoder, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        for stage, skip in zip(self.decoder, reversed(skips)):
            x = stage(x, skip)
        return self.classifier(x)[..., :height, :width]
