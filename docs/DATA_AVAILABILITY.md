# Data and result availability

Our study uses 112 annotated optical micrographs of TC4 titanium alloy, approximately 1024 × 1024 pixels each. Section 2.1 describes the dataset and attributes the source images and annotations to Chen et al. (2024), *Instance segmentation from small dataset by a dual-layer semantics-based deep learning framework*, Science China Technological Sciences 67, 2817–2833, DOI: 10.1007/s11431-023-2646-3.

We release the experimental values reported in Tables 1–3 and the timing comparison in Section 3.4 under `results/paper/`. We include the original training-curve and qualitative-result figures under `assets/paper/`, with page and figure references. These files record our reported experiments; they are separate from synthetic software-validation outputs.

The repository does not contain the 112 microscopy images, pixel masks, specimen-level split manifests, trained checkpoints, per-image predictions or per-epoch numerical logs. Figure images preserve the curves as published; we do not convert them into purported original logs. The CSV interface supports local image/mask collections without downloading data.
