# 胸部 X 光肺炎二分类 (Pneumonia Chest X-Ray Classification)

2025 年暑期科研项目：基于儿童胸部 X 光片，区分 **NORMAL（正常）** 与 **PNEUMONIA（肺炎）**，对比三类方法：

1. 从零训练的卷积神经网络（CNN）
2. ImageNet 预训练 ResNet50 作为冻结特征提取器 + 传统分类器
3. ViT 图像预处理 + 像素特征 + 传统分类器（基线）

## 数据集

使用 Kaggle 公开数据集 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)（Kermany et al., *Cell* 2018）。数据未包含在本仓库中，请下载后放在项目根目录：

```
.
├── train/
│   ├── NORMAL/      # 1,341 张
│   └── PNEUMONIA/   # 3,875 张
└── test/
    ├── NORMAL/      # 234 张
    └── PNEUMONIA/   # 390 张
```

训练集类别比例约为 1 : 2.9（正常 : 肺炎），存在明显的类别不平衡。

## 方法

| Notebook | 方法 | 输入 | 分类器 |
|---|---|---|---|
| [`cnn.ipynb`](cnn.ipynb) | 4 层 Conv + MaxPool + Dropout，从零训练 5 个 epoch | 100×100 RGB，归一化到 [0,1] | Dense(128) → Softmax(2) |
| [`resnet.ipynb`](resnet.ipynb) | ResNet50 (ImageNet 权重，`include_top=False`，冻结) 提取 4×4×2048 特征并展平 | 100×100，`resnet50.preprocess_input` | Logistic Regression / Random Forest |
| [`VIT.ipynb`](VIT.ipynb) | `ViTFeatureExtractor` (google/vit-base-patch16-224) 将图片缩放并标准化为 224×224×3，展平后作为特征 | 150,528 维像素向量 | Logistic Regression / Random Forest |

## 结果（测试集 624 张，Accuracy）

| 方法 | Accuracy |
|---|---|
| CNN（从零训练） | 78.0% |
| **ResNet50 特征 + Logistic Regression** | **80.8%** |
| ResNet50 特征 + Random Forest | 78.7% |
| ViT 预处理像素 + Logistic Regression | 75.3% |
| ViT 预处理像素 + Random Forest | 76.0% |

结论：在小数据集上，ImageNet 预训练特征的迁移效果优于从零训练的小型 CNN 和原始像素特征。

## 运行

Notebook 最初在 Google Colab 上运行，数据路径为 `/content/drive/MyDrive/train` 与 `/content/drive/MyDrive/test`。本地运行时请修改各 notebook 开头的 `train_dir` / `test_dir`。

```bash
pip install -r requirements.txt
```

## 已知局限与后续工作

- 仅报告了 Accuracy；在类别不平衡的医学场景中，还应报告敏感度（召回率）、特异度、F1 和 ROC-AUC。
- 未划分验证集，未使用数据增强和类别权重。
- `VIT.ipynb` 只使用了 ViT 的预处理器，并未使用 ViT 模型本身提取特征；下一步应接入 `ViTModel` 的 [CLS] 嵌入或直接微调。
