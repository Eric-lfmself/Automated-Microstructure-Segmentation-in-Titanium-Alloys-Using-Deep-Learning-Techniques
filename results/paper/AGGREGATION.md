# Aggregation notes

We retain Tables 1-3 exactly as printed in our manuscript. The arithmetic means of the displayed class scores differ from some Table 3 entries. We show both values rather than replacing either table.

| Model | Metric | Table 3 | Mean of displayed class values | Mean minus Table 3 |
| --- | --- | ---: | ---: | ---: |
| unet_vgg | miou | 0.6989 | 0.69808 | -0.00082 |
| unet_vgg | avg_colony_iou_classes_2_5 | 0.6538 | 0.65395 | 0.00015 |
| segformer | miou | 0.6725 | 0.70538 | 0.03288 |
| segformer | avg_colony_iou_classes_2_5 | 0.6725 | 0.671475 | -0.001025 |

These calculations use the printed per-class decimals. They do not recover the original per-image predictions or establish the cause of the differences. We keep this arithmetic record separate from the reported experiments. Our evaluator computes new-run metrics directly from an accumulated confusion matrix.
