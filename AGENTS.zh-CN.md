# 仓库指南

## 项目结构和模块组织
- `project.md` 记录了轻量级人脸识别的需求；请随着规格的演进而更新。
- 将生产代码放在 `src/` 下，并使用子包，例如 `src/models/` 用于存放模型架构，`src/pipelines/` 用于存放预处理/后处理流程，`src/utils/` 用于存放共享的辅助函数。
- 实验性的代码存放在 `notebooks/` (Jupyter) 和 `scripts/` (命令行工作流)；在发布前将稳定的逻辑迁移到 `src/` 中。
- 测试代码的目录结构与 `src/` 保持一致，存放在 `tests/` 中。小的参考图片和嵌入向量存放在 `assets/` 中。不要对模型权重文件进行版本控制；将它们存放在被 Git 忽略的 `artifacts/` 目录下。

## 构建、测试和开发命令
- `python -m pip install -r requirements.txt` - 安装指定的运行时和工具依赖。
- `pre-commit run --all-files` - 在推送前应用格式化工具 (`black`, `ruff`, `isort`) 和静态检查。
- `pytest` - 运行默认的单元测试套件；添加 `-m slow` 以运行性能测试。
- `python scripts/train.py --config configs/mobile.yaml` - 训练适用于移动端的骨干网络；为实验复用配置文件。
- `python scripts/export.py --weights artifacts/latest.pt --device cpu` - 创建可导出的推理图，用于部署测试。

## 编码风格和命名约定
- 使用 Python 3.10+，4个空格缩进，为公共函数添加类型提示，并为与张量交互的模块使用 NumPy 风格的文档字符串。
- 模块、函数和文件名使用 `snake_case` (下划线命名法)；类名使用 `PascalCase` (驼峰命名法)；常量使用 `UPPER_SNAKE_CASE` (大写下划线命名法)。
- 优先使用 TorchScript 友好的结构；避免使用会破坏 ONNX 导出的动态控制流。

## 测试指南
- 在 `tests/<module>/test_<feature>.py` 下编写 `pytest` 测试用例，镜像 `src` 的目录结构。
- 使用参数化测试和合成张量覆盖输入的边缘情况 (如光照、遮挡)。
- 保持快速单元测试的运行时间在 1 秒以下；使用 `@pytest.mark.slow` 标记较慢的硬件基准测试。
- 通过 `pytest --cov=src --cov-report=term-missing` 维持 >=85% 的语句覆盖率。

## 提交和拉取请求指南
- 遵循约定式提交规范 (`feat:`, `fix:`, `refactor:`)，主题行 <=72 个字符，正文使用简洁的项目符号提供上下文。
- 在页脚引用问题 ID (`Refs #123`)，并在相关时注明模型大小和延迟的变化。
- 拉取请求必须包含简短的叙述、前/后指标、测试结果以及更新的文档/配置。
- 当 UI、训练曲线或性能仪表板发生变化时，附上截图或 TensorBoard 导出结果。

## 模型和数据处理
- 不要将个人或敏感数据集放入仓库；在 `docs/datasets.md` 中记录所需的公共数据集。
- 在 `configs/hardware.yaml` 中记录硬件/软件假设，并在内核或量化设置更改时更新。
- 引入新的模型权重时，在发布说明中发布 SHA256 哈希值，并将二进制文件上传到共享的模型注册表。