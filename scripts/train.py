"""CLI: python -m scripts.train --dry-run --model unet_vgg --output-dir runs/smoke."""

from training.runner import main

if __name__ == "__main__":
    main()
