# Contributing

We welcome reproducible bug reports and focused improvements. Please include the relevant configuration, software versions and a minimal example. Use synthetic images in public reports; do not attach private microscopy images, checkpoints or credentials.

Before proposing a change, run:

```sh
python -m pytest -q
python -m scripts.smoke
```

Keep experimental results in `results/paper/` faithful to the cited manuscript. Explain any proposed correction with its table or section reference; do not replace reported values with smoke-test output. Follow the distribution terms in `RIGHTS.md`.
