"""Plot the per-class IoU and timing values reported in our manuscript.

Run from the repository root:
    python -m scripts.plot_paper_results --output-dir runs/paper_figures

The checked-in CSV tables are the only numerical inputs. We neither infer
uncertainty nor reconstruct per-epoch training values from figure images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper_figures"))
    args = parser.parse_args()
    sources = args.results_dir
    per_class = sources / "tables_1_2_per_class.csv"
    timing = sources / "efficiency.csv"
    with per_class.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with timing.open(newline="") as f:
        times = list(csv.DictReader(f))
    lookup = {(r["model"], int(r["paper_class"])): float(r["per_class_iou"]) for r in rows}
    expected = {(m, c) for m in ("unet_vgg", "segformer") for c in range(1, 6)}
    if len(rows) != 10 or set(lookup) != expected or any(not 0 <= v <= 1 for v in lookup.values()):
        raise ValueError("Expected ten valid per-class IoU values from Tables 1 and 2")
    if len(times) != 1:
        raise ValueError("Expected one timing record from Section 3.4")
    t = times[0]
    seconds = [float(t["full_pipeline_seconds_per_image_approx"]), float(t["manual_annotation_seconds_per_image_approx"])]
    if any(not np.isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("Timing values must be positive and finite")
    speedup = float(t["reported_speedup"])
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    blue, violet, teal, ink, muted = "#3977b6", "#8264ac", "#188879", "#1c3042", "#596c7d"
    style = {
        "font.family": "DejaVu Sans", "font.size": 13, "text.color": ink,
        "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
        "axes.edgecolor": "#dce4eb", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "svg.fonttype": "none", "svg.hashsalt": "tc4-paper-results",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 5.8))
        fig.subplots_adjust(left=.235, right=.955, top=.76, bottom=.13)
        y = np.arange(5)
        for model, label, offset, color, hatch in (
            ("unet_vgg", "VGG-based U-Net", -.18, blue, None),
            ("segformer", "SegFormer", .18, violet, "///"),
        ):
            values = [lookup[model, c] for c in range(1, 6)]
            bars = ax.barh(y + offset, values, height=.29, color=color, label=label,
                           hatch=hatch, edgecolor="white", linewidth=.65, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(value + .012, bar.get_y() + bar.get_height()/2, f"{value:.4f}",
                        va="center", ha="left", fontsize=12, color=ink)
        ax.set_yticks(y, ["Equiaxed α\nClass 1"] + [f"Colony orientation {c}\nClass {c+1}" for c in range(1,5)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.03)
        ax.set_xticks(np.arange(0, 1.01, .2))
        ax.set_xlabel("Intersection over union (IoU)", labelpad=9)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", color="#e7edf2", linewidth=.8, zorder=0)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.04), ncol=2, frameon=False, fontsize=12)
        fig.text(.035,.94,"Segmentation quality by microstructure class",fontsize=21,weight="bold")
        fig.text(.035,.875,"Paper Tables 1–2 · original reported values",fontsize=12.5,color=muted)
        fig.text(.035,.018,"Class IDs follow the paper (1–5); the code uses IDs 0–4.",fontsize=10.5,color=muted)
        fig.savefig(output / "per_class_iou.png", dpi=220)
        fig.savefig(output / "per_class_iou.svg", metadata={"Date": None})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        fig.subplots_adjust(left=.23, right=.95, top=.73, bottom=.24)
        bars = ax.barh([1,0], seconds, height=.44, color=[teal,"#a0adba"], zorder=3)
        for bar, value in zip(bars, seconds):
            ax.text(value + 14, bar.get_y()+bar.get_height()/2, f"≈ {value:g} s", va="center", fontsize=14, weight="bold")
        ax.set_yticks([1,0], ["Automated pipeline", "Manual annotation"])
        ax.set_xlim(0, 1020)
        ax.set_xticks([0,180,360,540,720,900])
        ax.set_xlabel("Seconds per image", labelpad=10)
        ax.set_ylim(-.5,1.65)
        ax.tick_params(axis="both",length=0)
        ax.tick_params(axis="y",pad=12)
        ax.grid(axis="x",color="#e7edf2",linewidth=.8,zorder=0)
        ax.text(.96,.83,f"≈ {speedup:g}× faster",transform=ax.transAxes,ha="right",fontsize=25,weight="bold",color=teal)
        fig.text(.035,.92,"Processing time per 1024 × 1024 image",fontsize=21,weight="bold")
        fig.text(.035,.835,"Paper Section 3.4 · approximate reported measurements",fontsize=12.5,color=muted)
        fig.text(.035,.075,"Full pipeline: preprocessing, model inference and post-processing.",fontsize=11,color=muted)
        fig.text(.035,.025,"The paper describes standard hardware; exact device specifications are not reported.",fontsize=10.5,color=muted)
        fig.savefig(output / "processing_time.png", dpi=220)
        fig.savefig(output / "processing_time.svg", metadata={"Date": None})
        plt.close(fig)

    manifest = {
        "description": "We plot the reported per-class IoU and full-pipeline timing without recomputing aggregates.",
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (per_class,timing)],
        "operations": ["Read CSV values", "Draw zero-baseline horizontal bars", "Print original four-decimal IoU values", "Label timing values as approximate"],
        "uncertainty": "No error bars are added; the source tables do not report repeated-run variability.",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "figure_sources.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
