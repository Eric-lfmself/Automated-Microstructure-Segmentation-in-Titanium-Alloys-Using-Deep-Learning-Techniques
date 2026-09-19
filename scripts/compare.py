"""Compare measured result JSON files, while preserving paper source conflicts."""
import argparse
import json
from pathlib import Path
from eval.metrics import render_comparison_table, paper_reference_audit


def comparison(results):
    if len(results) < 2:
        raise ValueError("Provide at least two model results")
    first = results[0]
    for row in results[1:]:
        for key in ("mode","dataset_signature","sample_ids"):
            if row[key] != first[key]:
                raise ValueError(f"Comparison requires the same {key}")
    models = [r["model"] for r in results]
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model results")
    return "# Measured model comparison\n\nMode: "+first["mode"]+"\n\n"+render_comparison_table(dict(zip(models,results)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,default=Path("runs/comparison.md"))
    args=parser.parse_args(argv)
    if args.output.resolve() in {path.resolve() for path in args.results}:
        parser.error("Output cannot overwrite an input metrics JSON")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path")
    report=comparison([json.loads(p.read_text()) for p in args.results])
    report += "\n\n## Results in our paper and aggregation notes\n\n```json\n"+json.dumps(paper_reference_audit(),indent=2)+"\n```\n"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report);print(args.output.resolve())

if __name__ == "__main__":
    main()
