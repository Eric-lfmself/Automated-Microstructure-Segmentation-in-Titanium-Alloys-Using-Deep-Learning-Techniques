# Methods and implementation

We organize the implementation around Sections 2 and 3 of our paper.

## Task and labels

We predict five semantic classes: equiaxed alpha (code ID 0, paper Class 1) and four colony orientation classes (code IDs 1–4, paper Classes 2–5). Beta is not an additional semantic class. Both networks receive RGB tensors and return five-channel logits at the input resolution.

## IWMID and Otsu

For four Gaussian kernels, we compute each smoothed image from the original input:

```text
Bi = Gi * I                               (i = 1, 2, 3, 4)
D1 = I - B1; D2 = B1 - B2
D3 = B2 - B3; D4 = B3 - B4
D* = (1 - w1 sign(D1)) D1 + w2 D2 + w3 D3 + w4 D4
I* = I + D*
```

These correspond to Equations 1–4. We use signed floating-point differences and clip only the final enhanced image. Equation 5 selects the Otsu threshold by maximizing between-class variance. We return the binary alpha/beta prior separately from the RGB input and semantic target; the public network interface uses three input channels.

## VGG16 residual U-Net

We retain VGG16 convolution depths 2/2/3/3/3 and five pooling stages. Residual convolution blocks use batch normalization, ReLU, projection shortcuts where needed, and spatial dropout. Five transpose-convolution stages recover the input resolution, with encoder skip features at each stage. ImageNet initialization transfers only the 13 VGG16 convolution weights and biases. We initialize the added normalization, projection and decoder components separately.

## SegFormer-B0

We use Hugging Face `SegformerForSemanticSegmentation` with the MiT-B0 encoder and MLP decode head. The four encoder widths are 32/64/160/256, attention heads 1/2/5/8, and spatial-reduction ratios 8/4/2/1. The five-class head initializes separately from the optional `nvidia/mit-b0` ImageNet encoder.

Our public implementation follows the MLP design in Section 2.3.2 and Figure 3. The interpolation-only description in Section 3.2 differs from that design; we preserve this distinction in the documentation. Standard MiT uses convolutions in overlapping patch embeddings and Mix-FFN, without positional embeddings.

## Shared training and metrics

We use the same preprocessing, orientation-preserving augmentation, weighted cross-entropy and training loop for both models. Epoch loss uses the total weighted loss divided by the accumulated class-weight denominator. Pixel accuracy uses valid pixel counts. Evaluation accumulates a global confusion matrix before calculating per-class IoU and macro means. Classes with zero union are excluded from macro means and displayed as `N/A`.

The input pipeline does not apply rotations or flips: absolute orientation labels require an explicit label permutation under those transformations. Image and mask resizing share the same crop, and labels use nearest-neighbor interpolation.

## Release configuration

We expose all settings in `configs/config.py`. The following are configurable release defaults; the result archive preserves the experimental values reported in our paper independently of these settings.

| Setting | Default |
| --- | --- |
| Source image size | 1024 × 1024 |
| Smoke image size / batch | 64 × 64 / 2 |
| Gaussian sigmas | 1, 2, 3, 4 |
| IWMID weights | 0.5, 0.5, 0.5, 0.5 |
| Class weights | 1, 1, 1, 1, 1 |
| Optimizer / learning rate | AdamW / 0.0001 |
| AdamW weight decay | 0.01 |
| Dropout | 0.1 |
| Random seed | 42 |
| Image normalization | ImageNet RGB mean and standard deviation |
| Grayscale / histogram | BT.601 / 256 Otsu bins |
| Execution backend | CPU |

We record the resolved configuration, split fingerprints and random states with each checkpoint. This makes subsequent local runs traceable without treating configuration defaults as recovered experimental logs.
