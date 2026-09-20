# Training, evaluation and result export

## Environment and software checks

Install the project with `python -m pip install -e .` in a Python 3.11–3.12 environment. Run `python -m pytest -q` for tests and `python -m scripts.smoke` for a complete small workflow. The command-line implementation uses CPU. Smoke mode is offline, randomly initialized and limited to small synthetic batches.

## Pretrained encoder weights

We support local pretrained weights without network access:

- **VGG16:** a torchvision state dictionary with `features.*` keys, or a feature-only state dictionary. We load the 13 convolution weights and biases; a classifier is not constructed.
- **SegFormer-B0:** a local Hugging Face encoder directory or a raw `SegformerModel.state_dict()` file. The ImageNet checkpoint identifier is `nvidia/mit-b0`.

Set `use_random_init=false`, `allow_download=false` and `weights_path` to the local file or directory. Both wrappers expand `~`; relative paths resolve from the invoking working directory. A new five-class head initializes separately. For a new training run, `allow_download=true` explicitly permits encoder-weight retrieval; datasets are never downloaded. Evaluation and resume load their trained checkpoint locally.

## Local-data configuration

Copy `configs/real_template.json` to `configs/my_experiment.json`, then set:

1. `model` to `unet_vgg` or `segformer_b0`.
2. `train_manifest`, `val_manifest` and `test_manifest` to separate local CSV files.
3. Encoder initialization: set `weights_path` to local pretrained weights, or set `use_random_init=true` to train from scratch. Then choose image size, class weights, learning rate, batch size and total epochs.
4. `output_dir` to a new directory, such as `runs/my_experiment`.

The template uses 1024 × 1024 inputs, 100 epochs and CPU execution. These are editable settings, not an archived experimental run. Compute any data-derived class weights from training labels only. Keep `allow_download=false` in configurations used for evaluation.

```sh
python -m scripts.check_config --config configs/my_experiment.json
python -m scripts.train --config configs/my_experiment.json --run-real
```

We save `config.json`, `checkpoint.pt`, `history.json`, `history.csv` and `curves.png`. A new run refuses a nonempty output directory.

## Resume

```sh
python -m scripts.train --config configs/my_experiment.json --run-real \
  --resume runs/my_experiment/checkpoint.pt
```

Increase the total `epochs` to continue beyond a completed run. We restore model and optimizer parameters, epoch, metric history and Python/NumPy/Torch/data-loader random states. Resume initializes the architecture without retrieving encoder weights, then restores the complete checkpoint. Loading uses `weights_only=True`.

Training and validation sample IDs, file hashes, order and protocol must match. Paths can relocate if contents remain identical. Resume to an existing directory verifies checkpoint ownership and history; an already-completed resume does not perform another optimizer update. Use the same software versions for deterministic continuation.

## Held-out evaluation

```sh
python -m scripts.evaluate --config configs/my_experiment.json --run-real \
  --checkpoint runs/my_experiment/checkpoint.pt --output-dir runs/my_evaluation
```

We require a trained checkpoint and explicit test manifest, and verify separation from the original training/validation records. Outputs include metrics JSON/Markdown, a prediction preview, inference timing and the reported-paper comparison notes. Metrics use a dataset-wide confusion matrix. We exclude zero-union classes from macro means and label undefined values `N/A`.

Compare two local evaluations that use the same held-out samples and preprocessing:

```sh
python -m scripts.compare runs/eval_unet/metrics.json runs/eval_segformer/metrics.json \
  --output runs/model_comparison.md
```

We reject comparisons with different dataset signatures or sample IDs. Timing records specify their measurement scope: model-only inference timing is distinct from the full pipeline measurement in Section 3.4.

## Paper result archive

```sh
python -m scripts.export_paper_results --output-dir runs/paper_tables
```

This command exports the experimental values already reported in our manuscript. It does not run a model. We keep the reported tables, arithmetic aggregation notes and generated local evaluation outputs separate. The output directory must be empty to prevent overwriting an existing experiment.

## Troubleshooting

**The Python version is unsupported.** We support Python 3.11 and 3.12. Create the environment with the matching executable, for example `python3.12 -m venv .venv`, then activate it and install the project. Use `python -m pip` to install into that environment.

**Training requests encoder weights.** The local-data template enables pretrained initialization and disables downloads. Set `weights_path` to a local encoder file or directory in a [supported format](#pretrained-encoder-weights), or set `use_random_init=true` for a new run from scratch. `scripts.check_config` validates configuration values; training also checks the data files and weights.

**The output directory is not empty.** Choose a new `output_dir` for a new training run, smoke check, evaluation or table export. To continue an existing training run, use `--resume` with its checkpoint as described in [Resume](#resume).

**Masks or splits are rejected.** Use single-band class-ID masks with values `0..4`, or `1..5` with `label_offset=1`. Keep train, validation and test manifests separate, with unique IDs and no repeated images across splits. The [data guide](../data/README.md) covers mask conversion, relative paths and specimen grouping.
