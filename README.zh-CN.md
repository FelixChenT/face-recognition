# 轻量级人脸识别

一个面向移动端和边缘部署的轻量级、实时人脸识别技术栈。

## 项目布局
- src/ – 生产模块 (模型、管道、实用程序)。
- scripts/ – 用于训练、评估和导出工作流的 CLI 入口点。
- configs/ – 用于实验和硬件假设的 YAML 配置。
- 	ests/ – 与源树镜像的 pytest 套件。
-  ssets/ – 用于单元测试的小样本图像和嵌入。
- 
- otebooks/ – 在升级到 src/ 之前的探索性实验。
- docs/ – 附加文档 (数据集、研究笔记)。

## 快速开始
1. 创建一个面向 Python 3.10+ 的虚拟环境。
2. 安装依赖项：`python -m pip install -r requirements.txt`。
3. 运行单元测试：`pytest`。
4. 使用默认配置启动训练：`python scripts/train.py --config configs/mobile.yaml`。

## 贡献
请遵循 [`AGENTS.md`](AGENTS.md) 中的存储库指南，了解代码风格、测试和文档的期望。