"""Render Section 2.2 Eqs. (1)–(5) on a generated ordinary still-life scene.

Run ``python -m scripts.preview_preprocessing --output artifacts/m2`` from the
project root. The scene depicts a mug, fruit and a book; it is generated locally,
is not a micrograph or a training sample, and requires neither a dataset nor a
network connection. This is a CPU numerical/visual check, not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from preprocessing import binarize_otsu, enhance_iwmid, iwmid_components, otsu_threshold, to_grayscale


def make_still_life(size: int = 256) -> Image.Image:
    """Generate a small ordinary-object illustration without any external assets."""
    y, x = np.mgrid[:256, :256]
    background = np.stack([220 - 0.1 * y, 222 - 0.07 * y + 0 * x, 215 - 0.04 * y + 0 * x], axis=-1)
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 160, 256, 256), fill=(141, 101, 66))
    for start in range(163, 256, 9):
        draw.line((0, start, 256, start + 2), fill=(133, 93, 60), width=1)
    draw.ellipse((41, 186, 212, 223), fill=(94, 74, 56))
    draw.polygon(((150, 166), (219, 156), (241, 182), (168, 195)), fill=(36, 85, 130))
    draw.polygon(((168, 195), (241, 182), (241, 190), (168, 204)), fill=(230, 227, 213))
    draw.ellipse((123, 112, 171, 166), outline=(248, 244, 227), width=10)
    draw.rounded_rectangle((55, 100, 135, 195), radius=15, fill=(231, 235, 218))
    draw.rectangle((55, 101, 135, 167), fill=(231, 235, 218))
    draw.ellipse((55, 91, 135, 116), fill=(248, 246, 232))
    draw.ellipse((63, 97, 127, 110), fill=(80, 55, 35))
    draw.line((66, 122, 66, 171), fill=(252, 251, 241), width=5)
    draw.ellipse((139, 181, 184, 224), fill=(209, 63, 34))
    draw.ellipse((148, 187, 159, 194), fill=(246, 131, 82))
    draw.line((161, 184, 166, 174), fill=(81, 59, 37), width=3)
    draw.ellipse((168, 202, 223, 235), fill=(223, 177, 52))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = make_still_life()
    image.save(args.output / "generated_ordinary_still_life.png")
    parts = iwmid_components(image)
    enhanced = enhance_iwmid(image)
    gray = to_grayscale(enhanced)
    binary = binarize_otsu(gray)
    panels = [("Generated ordinary scene\n(not a micrograph)", parts["input"], False)]
    panels += [(f"B{i}: Gaussian sigma={i}\nEq. (1)", blur, False) for i, blur in enumerate(parts["blurred"], 1)]
    panels += [(f"D{i}: signed difference\nEq. (2)", dog, True) for i, dog in enumerate(parts["differences"], 1)]
    panels += [("D*: weighted difference\nEq. (3)", parts["enhancement_difference"], True),
               ("I*: enhanced, final clip [0, 1]\nEq. (4)", enhanced, False),
               (f"Otsu bright-pixel prior\nEq. (5); T={otsu_threshold(gray):.3f}", binary, False)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, (title, panel, signed) in zip(axes.flat, panels):
        if signed:
            signed_gray = panel @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            limit = max(float(np.abs(signed_gray).max()), 1e-6)
            im = ax.imshow(signed_gray, cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        else:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("IWMID / Otsu numerical preview — generated still life, CPU only", fontsize=15)
    fig.text(0.5, 0.017, "Assumptions: sigmas=(1,2,3,4); weights=(0.5,0.5,0.5,0.5); reflect boundaries. Signed panels use separate color scales.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    fig.savefig(args.output / "iwmid_contact_sheet.png", dpi=150)
    plt.close(fig)
    summary = {"source": "locally generated ordinary still life; not a micrograph", "device": "cpu",
               "sigmas": [1, 2, 3, 4], "weights": [0.5] * 4,
               "enhanced_unclipped_min": float(parts["enhanced_unclipped"].min()),
               "enhanced_unclipped_max": float(parts["enhanced_unclipped"].max()),
               "otsu_threshold": otsu_threshold(gray), "bright_fraction": float(binary.mean())}
    (args.output / "preview_stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output / "iwmid_contact_sheet.png")


if __name__ == "__main__":
    main()
