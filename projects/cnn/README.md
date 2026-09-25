# CNN pneumonia classification / CNN 肺炎图像分类

Stage 1 establishes an audited dataset manifest and explains the original CNN. No training, final-model implementation, interactive demo or deployment is included at this stage. Baseline notebooks and `v0.1.0-baseline` remain unchanged.

## 先读什么

1. [原始 CNN 逐层讲解](docs/ARCHITECTURE.md)：从图像到分类、尺寸、参数量、损失和训练。
2. [数据审计报告](docs/DATA_AUDIT.md)：来源证据、隔离规则、固定划分和局限。
3. [机器可读摘要](reports/data_v1/summary.json)、[完整清单](reports/data_v1/manifest.csv)、[隔离清单](reports/data_v1/quarantine.csv)。

## 本地重新执行审计

在仓库根目录执行；数据应位于 `data/train/{NORMAL,PNEUMONIA}` 和 `data/test/{NORMAL,PNEUMONIA}`。ZIP 原包和影像均被 Git 忽略。

```bash
python3 -m venv .venv-audit
source .venv-audit/bin/activate
python -m pip install -r projects/cnn/requirements-audit.txt
python -m unittest discover -s projects/cnn/tests -v
python projects/cnn/audit_data.py --data-root data --output data/audit_recheck
```

报告目录必须不存在或为空，防止误覆盖冻结清单。比较重跑与已保存报告的 `dataset_sha256`、`split_sha256` 和 `manifest_sha256`；Python 版本等环境字段可能因机器不同而变化。解码依赖固定为 Pillow 11.3.0，环境变化需重新核对像素哈希。

`audit_data.py` 只读取数据并写报告，不移动、删除或修改影像。`quarantine.csv` 是逻辑隔离清单；后续训练仅加载 `status=eligible` 且 `split=train` 的记录，验证仅使用 `split=val`，开发评估仅使用 `split=development_test`。

## 分组和哈希

文件名分组加原文件/解码 RGB 像素哈希形成关联组件。跨原始集合、标签冲突或含损坏图的组件全部隔离。同集合像素相同的副本保留路径排序后的第一份。未知文件名以自身路径作为临时组并在摘要报告，不能据此声称患者级无泄漏。

每类以完整组为单位选择最接近 20% 图像数的验证集；使用固定种子 42，不根据模型成绩挑选划分。未执行感知近重复检索，也没有验证不同文件名一定属于不同患者。

## 下一阶段（尚未实施）

在固定划分上实现同结构 PyTorch CNN 基线与一个组合改进 CNN，两个模型均运行三个种子。完成阶段讲解和验收后才进入训练。当前报告中不存在可用于简历的新增模型成绩。
