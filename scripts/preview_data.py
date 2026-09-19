"""Save a tiny synthetic data preview; no training, downloads, or inference.

Run from project root: python -m scripts.preview_data --output outputs/dummy.png
"""

import argparse
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from data import CLASS_NAMES, DummyTC4Dataset, PairedAugment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dummy_preview.png"))
    parser.add_argument("--size", type=int, default=128, choices=(64, 128, 256))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = DummyTC4Dataset(length=2, image_size=(args.size, args.size), seed=args.seed,
                          preprocess=False, normalize=False)
    enhanced = DummyTC4Dataset(length=2, image_size=(args.size, args.size), seed=args.seed,
                               normalize=False)
    augmented = DummyTC4Dataset(length=2, image_size=(args.size, args.size), seed=args.seed,
                               normalize=False, transform=PairedAugment(brightness=.15, contrast=.15))
    colormap = ListedColormap(["#f2d277", "#549acc", "#66b394", "#bb80bd", "#e58c68"])
    norm = BoundaryNorm(np.arange(-.5, 5.5), colormap.N)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), constrained_layout=True)
    for index in range(2):
        original, processed, jittered = raw[index], enhanced[index], augmented[index]
        panels = (
            (original["image"].permute(1, 2, 0), "Synthetic RGB", None),
            (processed["image"].permute(1, 2, 0), "IWMID enhanced", None),
            (processed["prior"][0], "Otsu auxiliary prior", "gray"),
            (processed["mask"], "Five-class target", colormap),
            (jittered["image"].permute(1, 2, 0), "Photometric augmentation", None),
        )
        for column, (pixels, title, cmap) in enumerate(panels):
            kwargs = {"cmap": cmap}
            if column == 3:
                kwargs["norm"] = norm
            elif column == 2:
                kwargs.update(vmin=0, vmax=1)
            axes[index, column].imshow(pixels.numpy(), **kwargs)
            axes[index, column].set_title(title, fontsize=10)
            axes[index, column].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colormap(i)) for i in range(5)]
    fig.legend(handles, [f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)],
               loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Synthetic fixtures for software validation", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
