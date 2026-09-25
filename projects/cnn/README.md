# PyTorch CNN：最终模型

最终采用原始结构的 **BaselineCNN**，从零训练。候选改进方案不再保留为当前入口，其实现与实验过程仍可通过 Git 历史查看。此决定不代表基础模型在所有指标上都更优。

## 模型结构

```text
100×100 RGB / 255
→ Conv(3→32) → ReLU → MaxPool → Dropout(0.2)
→ Conv(32→32) → ReLU → MaxPool → Dropout(0.2)
→ Conv(32→64) → ReLU → MaxPool → Dropout(0.2)
→ Conv(64→64) → ReLU → MaxPool → Dropout(0.2)
→ Flatten(1024) → Linear(128) → ReLU → Dropout(0.1)
→ Linear(2)
```

参数量 **197,026**。训练时直接用 logits 计算交叉熵；预测时使用 Softmax。NORMAL=0，PNEUMONIA=1，阈值 0.5。优化器为 Adam，无数据增强。

## 已完成训练

| 项目 | 结果 |
| --- | --- |
| 训练 / 验证图像 | 3,790 / 947 |
| seed | 42 |
| batch size / 学习率 | 32 / 0.001 |
| 完成 epoch / 最佳 epoch | 30 / 26 |
| 最佳权重的验证准确率 | **96.30%** |
| 最佳权重的验证 Balanced Accuracy | **96.41%** |
| 训练设备 / 用时 | Apple MPS / 约 11 分 29 秒 |

**这些是用于选取权重的验证集成绩，不是独立测试成绩。**当前仅运行一个随机种子，开发评估集尚未评估。不能与原 Notebook 的 624 张历史测试结果直接比较，也不声称临床有效性。

[训练记录](results/baseline_seed42/history.csv) · [训练配置](results/baseline_seed42/config.json) · [机器可读结果与权重哈希](results/baseline_seed42/metrics.json)

## 安装和运行

在仓库根目录执行，建议 Python 3.11：

```bash
python3.11 -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install -r projects/cnn/requirements.txt
```

### 使用已训练权重

从仓库的 **Releases → v0.2.0-cnn** 下载 `cnn_baseline_seed42.pt` 和 `SHA256SUMS`，放入本地 `weights/`。该目录不进入 Git。下载后可用 `shasum -a 256 -c SHA256SUMS` 在权重目录核对。

```bash
python projects/cnn/predict.py --checkpoint weights/cnn_baseline_seed42.pt --image /path/to/image.jpeg
```

输出预测类别、肺炎模型概率分数与阈值。该分数没有经过概率校准，不是临床患病概率。这里只提供离线命令，不提供展示界面。

### 重新训练

把获得授权的原始数据放入 `data/train/{NORMAL,PNEUMONIA}`、`data/test/{NORMAL,PNEUMONIA}`，使用仓库内固定清单。数据源的具体版本与使用边界见审计报告；影像不随仓库分发。

```bash
python projects/cnn/train.py --seed 42 --output runs/cnn/baseline/reproduction
```

默认最多 30 epoch，按验证 Balanced Accuracy 保存最佳权重，连续 7 轮不提升则早停；自动选择 CUDA/MPS/CPU。原来的 `--model baseline` 参数仍可使用。不同设备不保证逐位一致。

输出 `best.pt`、`history.csv` 和 `config.json`，拒绝覆盖非空输出目录。训练代码不读取开发评估集。

## 代码与资料

- [models.py](models.py)：最终基础 CNN。
- [dataset.py](dataset.py)：固定清单读取、RGB 转换、100×100 缩放和除以 255。
- [train.py](train.py)：训练、验证、早停和权重保存。
- [predict.py](predict.py)：加载权重并预测单张图像。
- [CNN 逐层讲解](docs/ARCHITECTURE.md) · [数据审计报告](docs/DATA_AUDIT.md)

数据分组使用文件名与内容代理，不是已验证患者级划分。隔离后开发评估集为 301 张，其中肺炎 70 张，存在选择偏差风险。本次不以该集合的结果选模型。
