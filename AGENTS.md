# AGENTS.md

## 1) 目的与适用范围
本文件定义在本仓库中运行的代码代理（含人类协作者）的执行规范，目标是：
- 代码可维护、可测试、可文档化
- 修改可追踪、可复现
- 对 `torch.compile` 兼容性优化任务友好

适用范围：仓库根目录及其子目录。

---

## 2) 快速开始

### 2.1 环境
```bash
# Python >= 3.11
uv venv --python 3.11
source .venv/bin/activate

# 按你的环境安装 PyTorch
# 例如：uv pip install torch torchvision torchaudio

# 开发安装
uv pip install --editable . --group dev

# 可选依赖
uv pip install --editable ".[triton]"
uv pip install --editable ".[cupy12]"
uv pip install --editable ".[nir]"
```

### 2.2 常用命令
```bash
# 格式化
uv format

# 全量测试
pytest

# 单文件测试
pytest test/activation_based/test_roofline.py

# 单测函数
pytest test/activation_based/test_roofline.py::test_example

# 详细输出
pytest -v
```

### 2.3 文档构建
```bash
cd docs
make html
# docs/build/html/index.html
```

---

## 3) 代码组织约定
- 神经元实现：`spikingjelly/activation_based/neuron/`
- 层实现：`spikingjelly/activation_based/layer/`
- 数据集实现：`spikingjelly/datasets/`
- CUDA/Triton kernel：独立模块，不与高层逻辑混杂
- 公共 API：通过 `__init__.py` + `__all__` 暴露

---

## 4) 风格与命名
- 导入顺序：标准库 → 第三方 → 本地模块
- 本地模块优先使用相对导入
- 类名：`PascalCase`
- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有成员：前导下划线 `_name`
- 行宽：88
- 字符串：双引号
- `src` 目录下不放 `.ipynb`

---

## 5) 类型与异常处理

### 5.1 类型标注
- 公共函数参数与返回值应完整标注
- 可空浮点用 `Optional[float]`
- 常用类型来自 `typing`（如 `Optional`、`Callable`、`Tuple`、`List`）

### 5.2 可选依赖
- 对 `cupy`、`triton`、`lava` 等使用 `try-except`
- 失败时使用 `logging.info` 给出提示
- 必要时抛出带清晰信息的 `ImportError`
- 与现有风格一致时可用 `BaseException` 做宽捕获

---

## 6) 文档规范（公共 API 必须）
公共 API 使用中英双语 docstring，包含：
1. API Language 引用标签（中文/English ref）
2. 中文与英文分节
3. 参数 `:param:` + 类型 `:type:`
4. 返回 `:return:` + `:rtype:`
5. 必要时数学公式 `.. math::`
6. 必要时图示 `.. image::`
7. 可选可运行示例 `.. code-block:: python`
8. 重要注意事项和参考文献

参考实现：
- `spikingjelly/activation_based/functional/loss.py`

---

## 7) 代理执行约束（兼容 Codex/CLI）
- 先读需求，再改代码，最后用最小必要测试验证
- 优先做“最小正确改动”，避免无关重构
- 不主动改动无关文件
- 若发现仓库中已有非本次任务改动，不回滚、不覆盖
- 输出中明确：改了什么、为什么、如何验证、剩余风险

建议执行顺序：
1. 阅读相关文件与测试
2. 实施改动
3. 运行格式化/静态检查/相关测试
4. 汇报结果与后续建议

---

## 8) 与当前主线任务的关系
当前仓库主线任务见 `plan.md`（`torch.compile` 兼容性升级）。
代理在涉及神经元、Triton/CuPy 后端时，优先保证：
- 减少/消除 graph break
- 保持数值一致性
- 补充最小但有效的回归测试
