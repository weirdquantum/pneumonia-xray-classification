| Variant | Description | AUC | Accuracy | Sensitivity | Specificity | Acc @0.5 | Δ AUC | Δ Accuracy |
|---|---|---|---|---|---|---|---|---|
| full (reference) | v2 CNN: shuffle, augment, class weight, BatchNorm, 224px, 30 epochs | 98.0 | 91.2 | 99.5 | 77.4 | 91.3 | – | – |
| epochs_5 | CNN trained for 5 epochs | 95.1 | 84.6 | 99.5 | 59.8 | 88.8 | -2.9 | -6.6 |
| low_res_112 | CNN at 112x112 input | 98.4 | 91.0 | 99.2 | 77.4 | 90.4 | +0.4 | -0.2 |
| no_augment | CNN without data augmentation | 93.7 | 85.1 | 98.5 | 62.8 | 81.9 | -4.3 | -6.1 |
| no_batchnorm | CNN without BatchNorm | 93.0 | 80.1 | 99.0 | 48.7 | 83.7 | -5.1 | -11.1 |
| no_class_weight | CNN without class-weighted loss | 97.6 | 91.3 | 99.7 | 77.4 | 83.5 | -0.5 | +0.2 |
| no_shuffle | CNN with class-sorted (unshuffled) batches, the v1 bug | 61.0 | 57.5 | 57.2 | 58.1 | 62.5 | -37.0 | -33.7 |
| resnet50_linear_probe | ResNet50 with frozen ImageNet features, head only (v1-style transfer) | 93.2 | 83.8 | 92.8 | 68.8 | 79.5 | -4.8 | -7.4 |
| v1_like | v1 recipe: v1 CNN, 100px, no augment/class weight/shuffle, 5 epochs, batch 10 | 22.5 | 37.5 | 0.0 | 100.0 | 62.5 | -75.5 | -53.7 |
