# CNN：基础版与改进版

两个模型均已用 PyTorch 实现，共用数据加载和训练入口。已用少量真实图像验证训练、权重保存加载和预测一致性；尚未进行完整训练，不报告模型性能。

## 核心文件

| 文件 | 作用 |
| --- | --- |
| [models.py](models.py) | `BaselineCNN`、`ImprovedCNN` 两个模型 |
| [dataset.py](dataset.py) | 从既有划分清单加载图像，统一预处理和训练增强 |
| [train.py](train.py) | 共用训练、验证、早停和最佳权重保存 |

## 两个模型有什么区别

| | 基础版 | 改进版 |
| --- | --- | --- |
| 输入 | 100×100 RGB，像素除以 255 | 相同 |
| 卷积通道 | 32 → 32 → 64 → 64 | 相同 |
| 卷积块 | Conv → ReLU → MaxPool → Dropout(0.2) | Conv → BatchNorm → ReLU → MaxPool |
| 分类头 | Flatten → Linear(128) → ReLU → Dropout(0.1) → Linear(2) | 全局平均池化 → Flatten → Dropout(0.3) → Linear(2) |
| 可学习参数 | 197,026 | 66,082 |
| 训练增强 | 无 | 旋转 ±7°，平移最多 5% |
| 损失 | 普通交叉熵 | 由训练集类别数量计算权重的交叉熵 |
| 优化器 | Adam | AdamW，weight decay=1e-4 |

两者都输出 logits，训练时直接送入交叉熵；预测时才做 Softmax。标签固定为 NORMAL=0、PNEUMONIA=1，肺炎概率 ≥0.5 判为肺炎。训练改动同时使用，不做消融实验，也不预先宣称改进模型更好。

## 运行

在仓库根目录执行。建议 Python 3.11；当前本地 `.venv-cnn` 已安装依赖。新环境先创建虚拟环境并安装 `projects/cnn/requirements.txt`。

```bash
source .venv-cnn/bin/activate

# 基础模型
python projects/cnn/train.py --model baseline --seed 42

# 改进模型
python projects/cnn/train.py --model improved --seed 42
```

默认 batch size=32、学习率=1e-3、最多 30 epoch，连续 7 个 epoch 验证 Balanced Accuracy 未提升则早停。自动选择 CUDA → MPS → CPU，也可以指定 `--device cpu` 或 `--device cuda`。同结构重建并非去年 Keras 结果的精确复现。

默认读取本地 `data/` 和现有 `reports/data_v1/manifest.csv`，只加载训练集和验证集。命令不会评估开发测试集。Colab 上按同样目录放置数据，或使用 `--data-root` 指定路径。

每次运行输出到 `runs/cnn/<model>/seed<seed>/`：

- `best.pt`：验证 Balanced Accuracy 最高的模型权重、模型名称、epoch 和配置。
- `history.csv`：各 epoch 的训练/验证损失、准确率、Balanced Accuracy。
- `config.json`：参数量、训练配置、类别权重、环境版本和数据清单哈希。

已有输出目录不会被覆盖；重跑使用新的 `--output`。后续正式实验分别使用 seed 42、43、44。显存不足时两个模型统一改为 `--batch-size 16` 并重跑，不自动改变一方的条件。

## 已做的验证

CPU 上使用真实训练图像 8 张、验证图像 4 张，各完成一个 epoch，仅检验流程。两个模型的输出尺寸、参数量、反向传播、最佳权重保存/加载、评估模式下重复预测以及单张/批量预测一致性均通过；增强仅用于改进版训练集。这些小样本检查不是可用于简历的实验成绩。

## 前一阶段资料

- [原始 CNN 逐层讲解](docs/ARCHITECTURE.md)
- [数据审计报告与划分局限](docs/DATA_AUDIT.md)
- [固定数据清单](reports/data_v1/manifest.csv)

正式训练沿用既有划分，不在此阶段重新调整数据。开发评估子集为 301 张，不能直接与旧 Notebook 的 624 张历史测试结果比较。
