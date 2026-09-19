"""Lazy CPU model factory shared by training/evaluation, paper Section 2.3."""


def create_model(config, *, force_random=False):
    """Build a CPU model; resume/evaluation must set ``force_random=True``.

    A trainer checkpoint replaces the full model state, so loading encoder
    pretraining first would be unnecessary and could trigger a download.
    """
    config.validate()
    common = dict(num_classes=config.num_classes, dropout=config.dropout,
                  use_random_init=True if force_random else config.use_random_init,
                  weights_path=None if force_random else config.weights_path,
                  allow_download=False if force_random else config.allow_download)
    if config.model == "unet_vgg":
        from .unet_vgg import VGGResUNet
        model = VGGResUNet(**common)
    elif config.model == "segformer_b0":
        from .segformer import SegFormerB0
        model = SegFormerB0(checkpoint=config.checkpoint, **common)
    else:
        raise ValueError(f"Unsupported model: {config.model}")
    return model.to(device="cpu")
