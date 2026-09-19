# Experimental results from our paper

We release the experimental values reported in *Automated Microstructure Segmentation in Titanium Alloys Using Deep Learning Techniques*. Each record identifies its source table, section or manuscript page. Decimal strings preserve the printed precision, including trailing zeros.

## Model comparison: Table 3

| Model | Pixel accuracy | mIoU | Equiaxed alpha IoU | Mean colony IoU |
| --- | ---: | ---: | ---: | ---: |
| VGG based U-Net | 0.9000 | 0.6989 | 0.8746 | 0.6538 |
| SegFormer | 0.8449 | 0.6725 | 0.8410 | 0.6725 |

## Per-class scores: Tables 1 and 2

| Model | Paper class | Structure | IoU |
| --- | ---: | --- | ---: |
| VGG based U-Net | 1 | Equiaxed alpha-phase | 0.8746 |
| VGG based U-Net | 2 | Colony (Orientation 1) | 0.5602 |
| VGG based U-Net | 3 | Colony (Orientation 2) | 0.6853 |
| VGG based U-Net | 4 | Colony (Orientation 3) | 0.7251 |
| VGG based U-Net | 5 | Colony (Orientation 4) | 0.6452 |
| SegFormer | 1 | Equiaxed alpha-phase | 0.8410 |
| SegFormer | 2 | Colony (Orientation 1) | 0.5803 |
| SegFormer | 3 | Colony (Orientation 2) | 0.6955 |
| SegFormer | 4 | Colony (Orientation 3) | 0.7397 |
| SegFormer | 5 | Colony (Orientation 4) | 0.6704 |

The tables label their score column `mIoU`; each row describes one class. We name this field `per_class_iou` and retain the original column heading as metadata. Code IDs are paper class IDs minus one. The archive's `segformer` identifier corresponds to the command-line model `segformer_b0`.

## Training summary and efficiency

In Section 3.1 we describe VGG U-Net validation accuracy as plateauing around 0.83 with training loss around 0.45; for SegFormer we describe validation accuracy fluctuating around 0.77 and training loss around 0.53. These approximate narrative values are separate from Table 3 and are not exact final-epoch measurements extracted from the plots.

Section 3.4 reports approximately 36 seconds per 1024 x 1024 image for preprocessing, model inference and post-processing, compared with approximately 900 seconds (15 minutes) for manual annotation, giving a 25x speedup. We preserve the hardware description as `standard hardware`; this archive does not add a device specification.

Section 3.2 describes eight representative held-out test micrographs evaluated by both models. We retain that statement without inferring the complete 112-image split membership.

## Files

| File | Content |
| --- | --- |
| [paper_results.json](paper_results.json) | All reported values, source locations and evaluation context |
| [paper_metadata.json](paper_metadata.json) | Title, author order, affiliation and source PDF checksum |
| [tables_1_2_per_class.csv](tables_1_2_per_class.csv) | Ten per-class IoU rows |
| [table_3_model_comparison.csv](table_3_model_comparison.csv) | Two model-level comparison rows |
| [validation_summary.csv](validation_summary.csv) | Approximate Section 3.1 training descriptions |
| [efficiency.csv](efficiency.csv) | Section 3.4 timing comparison |
| [arithmetic_check.csv](arithmetic_check.csv) | Arithmetic means of printed class scores, kept separately |
| [AGGREGATION.md](AGGREGATION.md) | Explanation of the table arithmetic differences |

We preserve reported aggregate results here. Per-image predictions, full-resolution annotations, trained weights and raw epoch logs are not included. Synthetic software checks write to their own run directories and do not alter this archive.
