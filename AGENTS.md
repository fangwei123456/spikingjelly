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
本节适用于所有“对外可见”的类、函数、方法、模块常量与工厂函数。目标是让用户在阅读源码、IDE 提示与 Sphinx 文档时获得一致信息。

### 6.1 总体要求
- 公共 API 必须提供**中英双语 docstring**，并保持语义一致，不得中英文内容冲突。
- docstring 使用 **Sphinx/RST 风格字段**，统一采用 `:param:`、`:type:`、`:return:`、`:rtype:`、`:raises:`。
- 文档内容必须与实现同步：参数默认值、张量形状、dtype、设备约束、后端限制、异常条件需真实可验证。
- 任何影响行为的改动（参数、返回、状态、副作用）必须同步更新 docstring 与相关文档页面。

### 6.2 推荐结构（按顺序）
1. `.. admonition:: API 文档` 中包含中文/English 引用标签（如已有项目约定，沿用既有格式）。
2. 中文小节：功能概述、关键行为、状态语义（如神经元 `reset()` 影响）。
3. English 小节：与中文等价的英文描述。
4. 参数与类型：逐项写 `:param:` + `:type:`，单位、形状、取值范围要明确。
5. 返回与类型：写 `:return:` + `:rtype:`，说明返回张量/对象的结构与语义。
6. 异常：用 `:raises:` 标明触发条件（如依赖缺失、shape 不合法、dtype 不支持）。
7. 必要时补充数学公式、示意图、最小可运行示例、注意事项与参考文献。

### 6.3 类型标注与 docstring 一致性
- 公共函数签名必须完整类型标注，docstring 中的 `:type:`/`:rtype:` 必须与签名一致。
- 可空参数显式写 `Optional[...]`，并在文档中写明 `None` 的行为分支。
- 对张量参数至少说明：
  - 维度约定（如 `[..., T, N]` 或 `[T, N, *]`）
  - dtype 约束（如 `float32/float16`）
  - device/后端约束（CPU/CUDA、torch/cupy/triton）
- 如果返回值依赖 `training/eval`、`detach_reset`、后端选择等开关，必须明确写出条件差异。

### 6.4 Sphinx/RST 写作细则
- 使用 RST 指令：
  - 数学公式：`.. math::`
  - 代码示例：`.. code-block:: python`
  - 图示：`.. image:: path`
  - 交叉引用：优先使用 `:class:`、`:func:`、`:mod:`。
- 避免口语化描述，优先使用可验证陈述（输入条件 -> 行为 -> 输出结果）。
- 示例代码要求可复制运行，尽量覆盖常见路径与边界行为（如状态重置、后端切换）。

### 6.5 最小质量门槛（提交前）
- 新增/修改公共 API 时，至少检查：
  1. docstring 中英双语齐全，字段完整。
  2. 类型标注与 `:type:`/`:rtype:` 一致。
  3. 关键约束（shape/dtype/device/backend/state）已写明。
  4. 文档可构建且无新增严重告警（建议执行 `cd docs && make html`）。

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

---

## 9) SpikingJelly 开发与测试补充约束
适用于神经元状态逻辑与 CuPy/CUDA 后端的最小必守规则：

1. 测试入口
- 统一使用 `pytest`；避免以 `python test_xxx.py` 直接运行测试文件。

2. 有状态神经元
- `IFNode`、`LIFNode`、`ParametricLIFNode` 等包含内部状态；若两次 forward 的 batch/shape 变化，先 `reset()` 再执行下一次 forward。

3. CuPy + FP16 速查
- half2 对齐：除时间维 `T` 外的神经元总数需为偶数。
- 数值比较：`float16` 下对 torch/cupy 结果比对应使用较宽容差（建议 `atol=1e-2, rtol=1e-2`）。

---

## 10) 远程开发工作流（本地代理 + SSH）
目标：**代理运行在本地开发机**，通过 SSH 在远端服务器执行测试与验证；不在远端安装 Codex/Cursor 等代理程序。

### 10.1 原则
- 控制面本地化：代码编辑、上下文管理、指令执行由本地代理完成。
- 执行面远端化：耗算力任务（GPU 测试、性能基准）通过 SSH 下发到远端。
- 环境最小侵入：远端仅需 Python/依赖/仓库副本，不要求安装任何本地代理插件。

### 10.2 推荐前置配置
1. 本地 `~/.ssh/config` 配置 Host 别名、端口、用户与私钥。
2. 远端准备独立工作目录（如 `~/work/spikingjelly-dev`）。
3. 远端创建虚拟环境并安装依赖（建议与本地版本矩阵对齐）。
4. 需要 GPU 时，先在远端确认 `nvidia-smi`、CUDA、PyTorch/CuPy 版本兼容。

### 10.3 标准执行流程
1. 本地修改代码并自测最小用例（快速失败优先）。
2. 同步代码到远端（`git push` + 远端 `git pull`，或 `rsync`）。
3. 通过 SSH 在远端执行测试命令（优先精确到模块/用例）：
   - `pytest test/xxx.py`
   - `pytest test/xxx.py::test_case -v`
4. 回收结果：保存日志、失败栈、环境信息（Python/torch/cupy 版本）。
5. 本地修复后再次同步并复测，直至通过。

### 10.4 命令约定（建议）
- 远端命令应显式包含：
  - 工作目录切换：`cd ~/work/spikingjelly-dev`
  - 环境激活：`source .venv/bin/activate`
  - 测试命令：`pytest ...`
- 对关键任务建议固定随机种子、固定设备可见性（如 `CUDA_VISIBLE_DEVICES`），保证复现性。

### 10.5 安全与审计
- 禁止在命令中明文硬编码密码/密钥。
- 优先使用 SSH key 与最小权限账号。
- 长任务建议使用 `tmux`/`screen` 或 CI 任务承载，避免会话中断导致结果丢失。
- 每次远端验证在提交说明中记录：执行主机、commit、命令、结果摘要。

### 10.6 常见问题处理
- 远端通过、本地失败：优先比对依赖版本、CUDA/驱动、默认 dtype 与随机性设置。
- 本地通过、远端失败：先检查 GPU 内存、CuPy/PyTorch ABI 兼容、测试输入是否满足 FP16 对齐约束。
- 结果波动：增加重复次数并记录统计；必要时拆分为稳定性测试与功能正确性测试。

---

## 11) 代理指令入口与优先级
为确保 Codex / Cursor / 其他代理在本仓库内自动读取正确指令，约定如下：

- 主规范文件：`AGENTS.md`（本文件，仓库级单一事实来源）。
- 本地技能文件：`.hermes/skills/spikingjelly-development/SKILL.md`（经验性补充，内容已合并到第 9 节）。
- 机器可读入口：`.cursorrules/codex.yaml`（指向本仓库应加载的指令文件）。

若多份指令存在冲突，优先级为：
1. 用户当次明确任务要求
2. `AGENTS.md`
3. `.cursorrules/codex.yaml` 中列出的补充文件
