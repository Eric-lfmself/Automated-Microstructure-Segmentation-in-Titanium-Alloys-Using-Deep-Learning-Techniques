"""Export the experimental tables in our paper without running a model.

Source values and manuscript locations are packaged with eval. We preserve
reported decimal precision and keep arithmetic comparisons in separate files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_source(name):
    return json.loads((Path(__file__).resolve().parents[1] / "eval" / name).read_text(encoding="utf-8"))


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_results(output):
    """Write the reported tables to an empty directory; never replace a run."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty; choose a new path")
    data = _read_source("paper_results.json")
    metadata = _read_source("paper_metadata.json")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("paper_results.json", data), ("paper_metadata.json", metadata)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, key in (("tables_1_2_per_class.csv", "tables_1_2"),
                      ("table_3_model_comparison.csv", "table_3"),
                      ("arithmetic_check.csv", "arithmetic_check"),
                      ("validation_summary.csv", "validation_summary")):
        _csv(output / name, data[key])
    efficiency = dict(data["efficiency"])
    efficiency["image_size"] = "x".join(map(str, efficiency["image_size"]))
    efficiency["pipeline_scope"] = "; ".join(efficiency["pipeline_scope"])
    _csv(output / "efficiency.csv", [efficiency])
    comparison = ["| Model | Pixel accuracy | mIoU | Equiaxed alpha IoU | Mean colony IoU |",
                  "| --- | ---: | ---: | ---: | ---: |"]
    for row in data["table_3"]:
        comparison.append("| " + " | ".join(row[k] for k in ("model_as_printed", "pixel_accuracy", "miou", "alpha_phase_iou", "avg_colony_iou_classes_2_5")) + " |")
    per_class = ["| Model | Paper class | Structure | IoU |", "| --- | ---: | --- | ---: |"]
    for row in data["tables_1_2"]:
        per_class.append(f"| {row['model_as_printed']} | {row['paper_class']} | {row['structure']} | {row['per_class_iou']} |")
    readme = """# Experimental results from our paper

We release the experimental values reported in *Automated Microstructure Segmentation in Titanium Alloys Using Deep Learning Techniques*. Each record identifies its source table, section or manuscript page. Decimal strings preserve the printed precision, including trailing zeros.

## Model comparison: Table 3

""" + "\n".join(comparison) + """

## Per-class scores: Tables 1 and 2

""" + "\n".join(per_class) + """

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
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    lines = ["# Aggregation notes", "", "We retain Tables 1-3 exactly as printed in our manuscript. The arithmetic means of the displayed class scores differ from some Table 3 entries. We show both values rather than replacing either table.", "", "| Model | Metric | Table 3 | Mean of displayed class values | Mean minus Table 3 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in data["arithmetic_check"]:
        lines.append("| " + " | ".join(str(row[k]) for k in ("model", "metric", "table_3_reported", "arithmetic_mean_of_displayed_class_values", "calculated_minus_reported")) + " |")
    lines += ["", "These calculations use the printed per-class decimals. They do not recover the original per-image predictions or establish the cause of the differences. We keep this arithmetic record separate from the reported experiments. Our evaluator computes new-run metrics directly from an accumulated confusion matrix.", ""]
    (output / "AGGREGATION.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper_tables"))
    args = parser.parse_args(argv)
    try:
        result = export_results(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(result)


if __name__ == "__main__":
    main()
