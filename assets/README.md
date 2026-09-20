# Framework and result figures

We provide the method overview as an editable SVG and a PNG preview. The SVG uses native text, shapes and connectors; its microscopy insets reproduce Figure 6 panels (a), (c) and (d). The inset images are clipped from the archived figure without intensity or color adjustments. See [framework_sources.json](framework_sources.json) for source coordinates and checksums. The [Mermaid file](method_overview.mmd) records the compact pipeline topology; the SVG contains the expanded architecture layout.

The method diagram shows our public implementation: shared IWMID preprocessing, two independently trained architectures and a separate Otsu prior. The network feature blocks are schematic. The result panels embedded in the diagram come from our paper.

## Experimental graphics

| Figure | Source |
| --- | --- |
| [Per-class IoU](results/per_class_iou.svg) | [Tables 1–2](../results/paper/tables_1_2_per_class.csv) |
| [Processing time](results/processing_time.svg) | [Section 3.4](../results/paper/efficiency.csv) |
| [Training curves and qualitative comparison](paper/README.md) | Original Figures 4–6 |

We generate the two numerical graphics directly from the archived CSV files:

```sh
python -m scripts.plot_paper_results --output-dir runs/paper_figures
```

The output directory must be empty. We export PNG and SVG figures together with source checksums and plotting operations in `figure_sources.json`. The charts use zero-baseline axes and retain every reported class value. We add no error bars or reconstructed training-epoch values. [Result-figure provenance](results/figure_sources.json) lists the inputs and exports.

The manuscript images, including the insets inside the method overview, retain their manuscript copyright. See [RIGHTS.md](../RIGHTS.md).
