# Pneumonia X-ray Classification

A chest X-ray classification learning project, originally implemented under instructor guidance in 2025 and rebuilt in PyTorch. The final model is a **four-block CNN trained from scratch**, with 197,026 trainable parameters.

**Final release: `v0.2.0-cnn` — PyTorch baseline CNN.**

## Final model and recorded result

| Model | Seed | Training / validation images | Best epoch | Validation accuracy | Validation balanced accuracy |
| --- | ---: | --- | ---: | ---: | ---: |
| BaselineCNN | 42 | 3,790 / 947 | 26 of 30 | 96.30% | 96.41% |

These are **validation metrics used for checkpoint selection**, not independent test results. Only one seed was run for the selected model. The development-test subset has not been evaluated. No improvement over the original notebook's historical test score is claimed.

- [Implementation, setup and prediction commands / 中文说明](projects/cnn/README.md)
- [Model architecture](projects/cnn/models.py) · [Training entry point](projects/cnn/train.py)
- [Training history](projects/cnn/results/baseline_seed42/history.csv) · [Configuration](projects/cnn/results/baseline_seed42/config.json)
- [CNN explanation / 逐层讲解](projects/cnn/docs/ARCHITECTURE.md)
- [Data audit and limitations](projects/cnn/docs/DATA_AUDIT.md)

Download the trained checkpoint and checksum from this repository's **Releases → v0.2.0-cnn**. The checkpoint supports CPU loading and includes preprocessing and class metadata. Raw images are not published.

## Original 2025 notebook archive

| Notebook | Actual method |
| --- | --- |
| [cnn.ipynb](notebooks/baselines/cnn.ipynb) | Keras CNN trained from scratch |
| [resnet.ipynb](notebooks/baselines/resnet.ipynb) | Frozen ResNet50 features + logistic regression / random forest |
| [VIT.ipynb](notebooks/baselines/VIT.ipynb) | ViT image preprocessing + flattened pixels + classical classifiers; no ViT network |

Original code and saved outputs remain unchanged from the archived baseline. Colab account/session metadata was cleaned before publication; original export hashes and the cleanup record are retained in the README at tag `v0.1.0-baseline`.

The experimental alternative CNN remains in Git history at commit `5af2761`. The final working tree uses the baseline model; this is a scope decision, not a claim that the baseline wins every metric.

## Data and reproducibility boundaries

The fixed manifests record image hashes and conservative filename/content grouping. Patient-level independence is not verified. The upstream data source and applicable redistribution terms still need definitive linkage to the supplied archives; candidate source details are documented in the audit. No raw-image redistribution or blanket dataset license is granted here.

The model is an educational artifact, not a diagnostic system. The repository preserves the original baseline tag, development commits and the final release so the project history can be inspected.
