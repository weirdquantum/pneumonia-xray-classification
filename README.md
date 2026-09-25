# Pneumonia X-ray Classification

Original chest X-ray classification experiments implemented in 2025 under instructor guidance. This repository starts by preserving three historical notebooks for learning, future comparisons, and Git version control.

**Status: original baseline archive (`v0.1.0-baseline`).** No model has been retrained for this release. Saved outputs are historical observations, not newly reproduced benchmark results or clinical validation.

## Baseline notebooks

| Notebook | Actual method | Historical accuracy |
| --- | --- | --- |
| [cnn.ipynb](notebooks/baselines/cnn.ipynb) | Keras CNN trained from scratch; four convolution blocks, then a dense classifier | 78.04% |
| [resnet.ipynb](notebooks/baselines/resnet.ipynb) | Frozen ImageNet ResNet50 feature extraction + logistic regression / random forest | 80.77% / 78.69% |
| [VIT.ipynb](notebooks/baselines/VIT.ipynb) | ViT image preprocessing + flattened pixels + logistic regression / random forest | 75.32% / 75.96% |

`VIT.ipynb` retains its original filename, but **does not load or run a Vision Transformer network**. `ViTFeatureExtractor` produces preprocessed pixel values; these are flattened directly for the classifiers. ResNet50 is used as a feature extractor and is not fine-tuned.

The recorded CNN output lists 5,216 training images and 624 test images. Dataset contents, duplicates, and patient overlap have not been independently audited in this release. The ViT-preprocessing notebook catches image-processing exceptions and may skip files, so equal evaluation populations must not be assumed. These numbers are not a controlled model comparison.

## Architecture at a glance

```text
cnn.ipynb
100x100 RGB -> rescale 1/255
-> [Conv3x3 + ReLU -> MaxPool2x2 -> Dropout(0.2)] x 4
   convolution channels: 32, 32, 64, 64
-> Flatten -> Dense(128, ReLU) -> Dropout(0.1)
-> Dense(2, Softmax)
Training: Adam, categorical cross-entropy, 5 epochs, batch size 10

resnet.ipynb
100x100 image -> ResNet50 preprocessing
-> ImageNet ResNet50(include_top=False) -> flatten features
-> Logistic Regression / Random Forest

VIT.ipynb
Image -> resize -> ViTFeatureExtractor -> flatten pixel_values
-> Logistic Regression / Random Forest
```

## Reading and running the archive

You can read the notebooks directly on GitHub without installing anything. To experiment, open a copy in Google Colab or Jupyter and use a new branch; keep the tagged baseline intact.

1. Obtain an authorized copy of the original NORMAL/PNEUMONIA chest X-ray data. Images are not distributed here. The original Drive folder contains dataset shortcuts; the upstream dataset source, version and redistribution terms still need verification, so this archive does not prescribe an unverified download.
2. Arrange the data as follows:

   ```text
   train/
     NORMAL/
     PNEUMONIA/
   test/
     NORMAL/
     PNEUMONIA/
   ```

3. For the original Colab paths, mount your own Google Drive at `/content/drive` and place `train` and `test` in `MyDrive`. For a different layout, update the path cells in your working copy, including the sample image path in `VIT.ipynb`.
4. Provide the libraries used by the selected notebook: TensorFlow/Keras, NumPy and scikit-learn; additionally seaborn for CNN plotting, OpenCV for ResNet50, and Transformers/Pillow for the pixel-preprocessing experiment. Some notebooks contain unpinned `pip install` cells. Pretrained weights/preprocessor downloads require network access.
5. Run cells from top to bottom in a fresh kernel. Rerunning feature extraction selectively in `resnet.ipynb` requires recreating the ResNet50 object: later cells reuse the name `model` for scikit-learn classifiers.

The CNN installation log records Python 3.11 paths, TensorFlow 2.18.0 and Keras 3.8.0. This is partial historical evidence, **not a verified environment lock**. Compatibility with current packages and exact result reproduction are not guaranteed. No new requirements lock, pretrained artifact or reproduction claim is included in this release.

## Known limitations and future comparison rules

- There is no explicit independent validation split or controlled multi-seed experiment. The CNN training generator uses `shuffle=False`.
- The ResNet notebook reads images with OpenCV; color-channel handling should be audited before reproduction.
- The ViT-preprocessing notebook resizes training images to 50x50 and test images to 100x100 before further preprocessing, skips exceptions, and records a logistic-regression convergence warning.
- Accuracy alone is insufficient for imbalanced classification. Future experiments should include per-class precision/recall, F1, confusion matrices and score-based metrics.
- Future comparisons need the same audited data split, preprocessing policy and evaluation population. If the existing test set informs development, label its results as development evaluation rather than an untouched holdout.
- This is an educational image-classification project, not a diagnostic system.

## Archive integrity and publication cleanup

Cell source text, execution counts and all saved cell outputs are preserved. Publication removes notebook-level Colab metadata and cell-level `executionInfo`, `colab` and `outputId` metadata (including account identifiers and session information). The ViT notebook's saved widget metadata is wrapped in the standard widget-state structure for renderer compatibility; widget data and cell outputs are retained. JSON formatting is normalized.

SHA-256 of the original Drive exports, before cleanup:

```text
cnn.ipynb     ff44b7dab99e494cac5d08fa6fab67b7a4c405b5b90a588a2b9aca0961bf82f0
resnet.ipynb  6b7cf9baf4663cead9e6ed64950368e09afd1368de7967929555b8afbd40ed02
VIT.ipynb     ce75e57c20f456746822160f27a90ff44c8c5699ea7d5281a3b31e359e91b0fe
```

Code and data provenance/third-party terms require further documentation before adding a blanket open-source license. No license grant for the dataset is made here.

## Git learning workflow / 版本控制

The initial history contains two commits: archive the notebooks, then add repository documentation. The annotated tag `v0.1.0-baseline` identifies this starting point.

```bash
git status                     # 查看工作区与暂存区
git log --oneline --decorate    # 查看提交历史和标签
git show --stat HEAD            # 查看最近一次提交涉及的文件
git show v0.1.0-baseline:README.md  # 读取基线版本的说明
git switch -c codex/reproduce-cnn  # 后续复现时创建分支，不必现在执行
```

修改文件后，使用 `git add <具体文件>` 选择本次内容，再用 `git commit -m "说明这次变化"` 保存本地快照。`git push` 才会把提交上传到 GitHub。基线文件保留在 `notebooks/baselines/`；改进实现放入新的目录，通过分支和 PR 记录变化。

本阶段仅归档导师指导下完成的原始实验；后续新增的个人改进及其验证结果会单独记录。

## CNN subproject

[Stage 1: data audit and CNN architecture explanation](projects/cnn/README.md). The original baseline archive is unchanged. New training and model comparisons have not started.
