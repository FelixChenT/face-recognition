# 轻量级人脸识别项目入门指南

本指南将引导您完成准备数据、配置环境和训练轻量级人脸识别模型所需的每一步。本说明假定您不熟悉 PyTorch 或机器学习工具。

## 1. 先决条件
- **操作系统：** Windows 10/11、macOS 13+ 或最新的 Linux 发行版。
- **Python：** 3.13 版本（该项目已通过 CPython 3.13.7 验证）。更早的 Python 版本将无法获得预构建的 Torch wheels。
- **命令行工具：**
  - 在 Windows 上，使用 *PowerShell*（已安装）。
  - 在 macOS/Linux 上，使用默认的终端。
- **磁盘空间：** 10 GB 可用空间，用于虚拟环境、数据集和训练产物。
- **互联网连接：** 需要下载 Python 包和公共数据集。

### 可选的硬件加速
- **GPU：** 具有至少 6 GB VRAM 的 NVIDIA GPU（例如 RTX 3060），可加快训练速度。安装 CUDA 12.4+ 驱动程序和匹配的 cuDNN。
- **仅 CPU 设置：** 现代 8 核 CPU（例如 AMD Ryzen 7 或 Intel Core i7）可以训练模型，但每个 epoch 需要更长的时间。
- **内存：** 最低 16 GB 系统 RAM，以轻松处理数据增强管道。

## 2. 克隆存储库
```bash
# 将 <path> 替换为您要存储项目的目录
cd <path>
git clone https://github.com/<your-account>/face-recognition.git
cd face-recognition
```

## 3. 创建并激活虚拟环境
```bash
# Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

## 4. 安装 Python 依赖项
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
需求文件指定了 CPU 友好的 wheels。如果您有 CUDA GPU，请从 https://download.pytorch.org 将 torch 和 torchvision 条目替换为特定于 CUDA 的 wheels。

## 5. 准备数据集
1. 查看 `docs/datasets.md` 以获取推荐的公共数据集（MS1M-ArcFace、CelebA、LFW）。
2. 将存档下载到存储库之外（例如 `D:\datasets\ms1m`）。
3. 提取图像，使每个身份都有自己的文件夹或保持原始文件夹布局。
4. （可选）运行人脸对齐以标准化为 112x112 的裁剪。

### 5.1 生成清单文件
训练管道需要 JSON Lines 清单文件，每行一个样本：
```json
{"path": "D:/datasets/ms1m/id0001/img0001.jpg", "label": 0}
{"path": "D:/datasets/ms1m/id0001/img0002.jpg", "label": 0}
{"path": "D:/datasets/ms1m/id0002/img0001.jpg", "label": 1}
```
- `path` 可以是相对于清单位置的相对路径或绝对路径。
- `label` 是一个整数类 ID。在训练和验证清单中使用一致的 ID。

创建两个清单：
- `data/manifests/train.jsonl`
- `data/manifests/val.jsonl`
将它们放在存储库中（创建 `data/manifests` 文件夹）。确保验证文件涵盖也出现在训练拆分中的身份。

## 6. 更新训练配置
打开 `configs/mobile.yaml` 并验证以下字段：
- `raining.dataset.train_manifest`：训练清单的路径（绝对路径或相对于配置文件的相对路径）。
- `raining.dataset.val_manifest`：验证清单的路径。
- `raining.dataset.image_size`：人脸裁剪的分辨率（默认为 112）。
- `raining.batch_size`：根据您的 GPU/CPU 内存进行调整。在 GPU 上从 64 开始，在 CPU 上从 32 开始。
- `raining.precision`：对于 CUDA GPU 设置为 `p16`，对于 CPU 训练设置为 `p32`。

如果更改任何设置，请保持 YAML 结构和缩进一致。

## 7. 启动训练
```bash
python scripts/train.py --config configs/mobile.yaml
```
主要行为：
- 检查点保存到 `rtifacts/checkpoints/`。脚本会保留最佳 epoch（基于验证准确性）并修剪超出配置限制的旧文件。
- 每个 epoch 的训练和验证指标都会打印到控制台。
- 要从命令行覆盖设备，请添加 `--device cuda:0` 或 `--device cpu`。

### 训练故障排除
- **内存不足 (GPU)：** 降低 `raining.batch_size` 或减小配置中的 `model.width_multiplier`。
- **数据加载缓慢：** 增加 `raining.num_workers`（Windows 用户可能需要保持在 ` ` 或 2）。
- **缺少清单：** 确认 YAML 中指定的路径正确且可访问。

## 8. 评估和导出嵌入
训练脚本已经报告了验证准确性。对于部署：
```bash
python scripts/export.py --config configs/mobile.yaml --weights artifacts/checkpoints/epoch_050.pt --output artifacts/exports/mobileface.onnx --device cpu --quantize
```
- `--weights` 应指向您要导出的检查点。对于最佳模型，请使用验证准确性提高时保存的文件。
- `--quantize` 会生成一个额外的 INT8 ONNX 文件，适用于边缘推理。

## 9. 验证导出的模型
使用 ONNX Runtime 运行快速健全性检查：
```bash
python -m pip install onnxruntime
python - <<'PY'
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('artifacts/exports/mobileface.int8.onnx', providers=['CPUExecutionProvider'])
dummy = np.random.randn(1, 3, 112, 112).astype('float32')
emb = session.run(None, {'images': dummy})[0]
print('Embedding shape:', emb.shape)
PY
```

## 10. 可选：运行测试套件
```bash
python -m pytest
```
通过测试可以确认模型、数据管道和训练器是否按预期运行。

## 11. 推荐的硬件配置文件
| 场景 | CPU | GPU | RAM | 预期的训练吞吐量 |
|----------|-----|-----|-----|------------------------------|
| 最小（仅 CPU） | 8 核台式机 CPU | 无 | 16 GB | 在 MS1M 子集上约 1-2 epochs/小时 |
| 平衡 | 12 核 CPU | NVIDIA RTX 3060 (12 GB) | 32 GB | 约 6-8 epochs/小时 |
| 性能 | 16 核 CPU | NVIDIA RTX 4090 (24 GB) | 64 GB | 约 20+ epochs/小时 |

对于边缘部署，导出的 ONNX 图可以在具有至少 2 GB RAM 和 128 MB 二进制文件存储空间的 ARM64 SoC 上舒适地运行。

## 12. 后续步骤
- 在基线准确性可接受后，集成其他增强或蒸馏策略。
- 通过向训练循环添加回调，使用 TensorBoard 或 Weights & Biases 跟踪实验指标。
- 在发布到生产环境之前，准备一个模型卡，总结评估结果。

通过遵循本指南，您可以在没有机器学习经验的情况下导入数据集、配置环境和启动训练。