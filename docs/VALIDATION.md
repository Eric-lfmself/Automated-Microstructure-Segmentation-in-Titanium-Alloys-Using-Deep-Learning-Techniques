# Software validation

We validated this release with Python 3.12.14, PyTorch 2.8.0, torchvision 0.23.0 and Transformers 4.57.1 on macOS ARM64.

- **204 tests passed**, covering preprocessing, data/labels, model structure, weighted loss, metrics, checkpoints, resume, command defaults and manuscript-result export.
- Both complete model pipelines passed a CPU smoke check with two 64 × 64 synthetic images, one optimizer update, validation, checkpoint saving/loading, evaluation and comparison.
- We built a wheel and source distribution, installed the wheel locally, and ran configuration inspection, exact paper-table export and both model pipelines from outside the source directory.
- The archived CSV/JSON values were checked against the manuscript tables. The three figure files retain their recorded extraction checksums.

The suite reports 14 dependency deprecation warnings from Matplotlib/Pyparsing. These did not cause test failures. These checks validate software behavior; our experimental results are recorded separately in `results/paper/`.

We provide the same test and smoke commands in `.github/workflows/tests.yml`. The workflow runs when this repository is pushed to GitHub; a local test result does not stand in for a hosted workflow run.
