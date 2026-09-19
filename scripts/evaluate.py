"""Evaluate CPU dummy fixtures or explicitly supplied held-out local data.

No model weights or dataset are downloaded. A real evaluation requires both
--run-real and a trained checkpoint; random predictions are always labeled.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from configs import load_config
from data import DummyTC4Dataset
from data.provenance import collect_provenance, assert_same_training_provenance
from eval.metrics import render_per_class_table, paper_reference_audit
from eval.runner import evaluate_loader
from eval.timing import benchmark_model_cpu
from models.factory import create_model
from training.checkpoint import read_checkpoint, restore_checkpoint
from training.runner import validate_split_manifests
from training.engine import seed_everything


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON configuration; defaults to the built-in CPU dry-run")
    parser.add_argument("--model", choices=("unet_vgg", "segformer_b0"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-real", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.model:
        config = replace(config, model=args.model)
    if args.dry_run:
        config = config.as_dry_run()
    if args.run_real:
        if config.dry_run:
            parser.error("--run-real requires a real configuration with dry_run=false")
        if args.checkpoint is None or not config.test_manifest:
            parser.error("Real evaluation requires --checkpoint and an explicit test_manifest")
    elif not config.dry_run:
        parser.error("Use --dry-run or explicitly --run-real for this configuration")
    if config.allow_download:
        parser.error("Evaluation loads local trained checkpoints only; allow_download must be false")
    config.validate()
    output = args.output_dir or Path("runs") / ("eval_dummy_"+config.model if config.dry_run else "eval_"+config.model)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty: {output}; choose a new directory")
    return args, config, output


def preview(batch, output, *, synthetic):
    """Small qualitative preview, Fig. 6 layout; never presented as paper data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ListedColormap(["#f2d277", "#549acc", "#66b394", "#bb80bd", "#e58c68"])
    norm = BoundaryNorm(np.arange(-.5,5.5),5)
    image = batch["image"][0].permute(1,2,0).numpy()
    image = np.clip(image*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
    fig, axes = plt.subplots(1,3,figsize=(10,3.6),layout="constrained")
    for ax, pixels, title in zip(axes,[image,batch["mask"][0].numpy(),batch["prediction"][0].numpy()],
                                  ["Enhanced RGB", "Ground truth", "Prediction"]):
        ax.imshow(pixels, **({} if pixels.ndim==3 else {"cmap":colors,"norm":norm}))
        ax.set_title(title);ax.axis("off")
    label = "Synthetic smoke; not paper results" if synthetic else "Held-out local prediction"
    fig.suptitle(label+" | "+str(batch["id"][0])+" | labels 0–4")
    fig.savefig(output,dpi=140);plt.close(fig)



def _dataset_signature(identity):
    """Hash sample identity while treating equivalent JSON numbers identically.

    IWMID accepts integer/float parameters equivalently; JSON spells 1 and 1.0
    differently, and -0.0 is the same nonnegative coefficient as 0.0. Preserve
    record order, content hashes, sizes and all other identity fields.
    """
    canonical = dict(identity)
    for key in ("sigmas", "weights"):
        canonical[key] = [0.0 if value == 0 else float(value) for value in identity[key]]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main(argv=None):
    args, config, output = parse_config(argv)
    torch.set_num_threads(2)
    seed_everything(config.seed)
    kwargs = dict(image_size=config.image_size,seed=config.seed,sigmas=config.sigmas,weights=config.iw_weights)
    if config.dry_run:
        # Match the trainer's independent validation fixture stream.
        kwargs["seed"] = config.seed + 1
        dataset = DummyTC4Dataset(length=config.dummy_length,**kwargs)
        identity = dict(kind="dummy",seed=dataset.seed,length=config.dummy_length,image_size=config.image_size,
                        sigmas=config.sigmas,weights=config.iw_weights)
    else:
        datasets = validate_split_manifests(config)
        dataset = datasets["test"]
        provenance = collect_provenance(datasets)
        test_identities = [{key:record[key] for key in ("id","image_sha256","mask_sha256")}
                           for record in provenance["test"]]
        identity = dict(kind="real",records=test_identities,label_offset=config.label_offset,
                        image_size=config.image_size,sigmas=config.sigmas,weights=config.iw_weights)
    payload = None
    checkpoint_info = None
    if args.checkpoint:
        payload = read_checkpoint(args.checkpoint)
        if not config.dry_run and payload["completed_epochs"] < 1:
            raise ValueError("Real evaluation requires a trained checkpoint with at least one completed epoch")
        saved = payload["config"]
        for key in ("model","num_classes","image_size","dropout","sigmas","iw_weights","label_offset"):
            actual = getattr(config,key)
            expected = saved[key]
            if isinstance(actual,tuple): expected = tuple(expected)
            if actual != expected:
                raise ValueError(f"Checkpoint evaluation configuration mismatch: {key}")
        if bool(saved["dry_run"]) != config.dry_run:
            raise ValueError("Cannot use a dummy-trained checkpoint for real evaluation or vice versa")
        if not config.dry_run:
            assert_same_training_provenance(payload.get("data_provenance"), provenance)
        checkpoint_info = {"path":str(args.checkpoint),"completed_epochs":payload["completed_epochs"]}
    model = create_model(config,force_random=True)
    if payload is not None:
        restore_checkpoint(payload,model,restore_rng=False)
        del payload
    result, batch = evaluate_loader(model,DataLoader(dataset,batch_size=config.batch_size,num_workers=0,shuffle=False),
                                     max_batches=config.max_batches if config.dry_run else None)
    # Only a small 64–128 sample is benchmarked during a dry-run. Real benchmarking
    # uses one image, explicitly opted into with --run-real above.
    timing = benchmark_model_cpu(model,batch["image"][:1],warmup=1,repetitions=3,enforce_small=config.dry_run)
    result.update(model=config.model,mode="dummy_validation_only" if config.dry_run else "held_out_local_evaluation",
                  checkpoint=checkpoint_info,weights_source="checkpoint" if args.checkpoint else "random",
                  config=config.to_dict(),model_only_timing=timing,
                  dataset_signature=_dataset_signature(identity))
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (output/"paper_reference_audit.json").write_text(json.dumps(paper_reference_audit(),indent=2,allow_nan=False)+"\n")
    report = f"# {config.model}: {result['mode']}\n\n"
    if config.dry_run:
        report += "Synthetic data / random initialization or one-step smoke checkpoint. These are pipeline checks, not reproduced paper scores.\n\n"
    report += render_per_class_table(result,model_name=config.model)
    report += "\n\nModel-only latency is separately recorded in metrics.json. It excludes IWMID, loading and postprocessing, and is not comparable to the paper's full pipeline 36 s/image.\n"
    (output/"metrics.md").write_text(report)
    preview(batch,output/"predictions.png",synthetic=config.dry_run)
    print(json.dumps({k:result[k] for k in ("model","mode","pixel_accuracy","miou","image_count")},indent=2))
    print(output.resolve())

if __name__ == "__main__":
    main()
