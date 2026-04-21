## 概述

为 `FlexSN` 新增 `backend="inductor"` 选项，将用户自定义的单步动力学函数 `core` 编译为高效的 Triton GPU 内核。该路径无需设置 `PYTORCH_JIT=0`，支持 `torch.compile` 跨层融合，推理和训练性能均优于现有 `backend="triton"`。

---

## 背景与动机

现有 `backend="triton"` 路径依赖自研 FX→Triton 映射表，存在以下局限：
- 需要 `PYTORCH_JIT=0`，与 `torch.compile` / Inductor 生态不兼容
- 映射表外的算子直接报错，`core` 函数受限
- `FlexSN` 作为 autograd 黑盒，`Linear→FlexSN→Linear` 无法跨层融合

`backend="inductor"` 使用 PyTorch 原生工具（`make_fx`、`aot_function`）追踪 `core`，复用现有 FlexSN Triton 模板基础设施生成内核，在保持 API 兼容的前提下解决上述问题。

---

## 新增文件

```
spikingjelly/activation_based/triton_kernel/flex_sn_inductor/
├── __init__.py          # HOP 定义 + eager_scan（M1/M2，已有）
├── hop.py               # FlexSNScan HigherOrderOp（已有）
└── kernel.py            # 新增：推理核 + 训练核构建

benchmark/flexsn/
├── flex_sn_inductor.py       # 新增：单层 + Linear→FlexSN→Linear benchmark
└── benchmark_vgg_train.py    # 新增：VGG16-BN 训练耗时对比

docs/source/tutorials/cn/flexsn_inductor.rst   # 新增：中文教程
docs/source/tutorials/en/flexsn_inductor.rst   # 新增：英文教程
docs/source/APIs/spikingjelly.activation_based.triton_kernel.flex_sn_inductor.rst
```

---

## 核心实现

### 推理路径（`kernel.py: build_inference_kernel`）

用 `make_fx` 追踪 `core`（无需 `PYTORCH_JIT=0`），生成 aten 级 FX 图后，
通过现有 `generate_triton_code_str` + `get_flexsn_inference_kernel`
编译出带 `tl.static_range(T)` 时间循环的单个 Triton 扫描内核。

### 训练路径（`kernel.py: build_training_kernels`）

用 `aot_function` 同时追踪正向和反向计算图（无需 `PYTORCH_JIT=0`）：
- 正向核：保存 backward 所需的中间值
- 反向核：时间逆序扫描，计算 BPTT 梯度

对于非可微输出（如硬阈值脉冲信号），AOT backward 会自动去掉 `grad_s`，
代码自动检测并生成 shim wrapper 适配模板的调用签名。

### `flexsn.py` 调度策略

| 条件 | 路径 |
|------|------|
| 推理（no grad）+ CUDA | Triton 单核 scan（1 次 launch） |
| 训练 + CUDA（含 `torch.compile` 内外） | `@torch._dynamo.disable` 包装的 Triton FlexSNFunction |
| CPU / kernel 不可用 | `eager_scan` / HOP fallback |

训练路径在 `__init__` 时将 `FlexSNFunction.apply` 预包装为 `@torch._dynamo.disable`。
在 `torch.compile()`（无 `fullgraph=True`）下，Dynamo 在此处产生 graph break，
FlexSN 继续用 Triton 单核扫描；周围 Conv/Linear 由 Inductor 编译，整体比不加 compile 快约 28%。
`torch.compile(fullgraph=True)` 会报错，移除该选项即可。

### bug fix（`torch2triton/torch2graph.py`）

`generate_forward_and_backward_graph` 原来取第一个输出张量调 `.backward()`，
但脉冲信号（比较运算结果）没有 `grad_fn`，导致报错。
修复：跳过无梯度的输出，取第一个有 `requires_grad` 或 `grad_fn` 的张量。

---

## 测试结果

### 单元测试

```
test/activation_based/test_flex_sn_inductor.py  36/36 passed
```

### GPU 验证（RTX 4090，CUDA 11.8，PyTorch 2.7.1）

**推理 benchmark（`gpu_verify.sh`）**

| T | B | N | triton (ms) | inductor (ms) | ratio |
|---|---|---|-------------|---------------|-------|
| 8 | 128 | 1024 | 0.16 | 0.09 | **0.52×** |
| 32 | 128 | 1024 | 0.19 | 0.05 | **0.28×** |
| 8 | 128 | 4096 | 0.08 | 0.07 | **0.89×** |

**训练 benchmark（SpikingVGG-16-BN，T=4，B=64，CIFAR-10，前向+反向）**

| 后端 | ms/iter | img/s | vs torch |
|------|---------|-------|---------|
| LIFNode torch | 36.20 | 1768 | 1.00× |
| LIFNode triton | 28.03 | 2283 | 0.77× |
| **FlexSN inductor** | **24.99** | **2561** | **0.69×** |

### 数值精度

- 推理：与 `backend="triton"` 最大绝对误差 = 0.0
- 训练梯度：与 `backend="torch"` 最大绝对误差 < 1e-6

---

## API 变更

`FlexSN.__init__` 新增 `backend="inductor"` 分支（默认不变，仍为 `"triton"`）：

```python
neuron = FlexSN(
    core=lif_core_sg,       # 使用 surrogate gradient 的 core 函数
    num_inputs=1,
    num_states=1,
    num_outputs=1,
    step_mode="m",
    backend="inductor",     # 新增选项
)

# 推理（无需 torch.compile）
with torch.no_grad():
    out = neuron(x)

# 训练
out = neuron(x)
out.sum().backward()

# 推荐：torch.compile() 使周围层由 Inductor 编译，FlexSN 仍走 Triton 核
# 训练整体比不加 compile 快约 28%（注意：fullgraph=True 会报错，不要加）
model = torch.compile(nn.Sequential(linear1, neuron, linear2))
```

`core` 中的算子需在 `FX_TO_TRITON` 映射表内（add/sub/mul/div/比较/sigmoid 等常见逐元素算子），不支持的算子自动回退 `eager_scan` 并输出警告日志。

---

## 已知限制

| 功能 | 状态 |
|------|------|
| `torch.compile()` + 训练（无 fullgraph） | ✅ 支持，Triton 核 + graph break，比不加 compile 快 ~28% |
| `torch.compile(fullgraph=True)` + 训练 | ❌ 报错（移除 fullgraph=True 即可） |
| float16 | ✅ 支持 |
| 任意 num_inputs / num_states / num_outputs | ✅ 支持 |
| Gradient checkpointing | ❌ 未实现 |
| DDP / FSDP | ❌ 未验证 |
| ROCm / XPU | ❌ 仅 CUDA |
| `core` 算子覆盖（exp/log/sqrt/tanh/sin/cos/erf/relu/clamp/pow 等） | ✅ 已支持 |

---

## 相关文件

- 详细设计文档：`workspace/design_flexsn_inductor.md`
- GPU 验证脚本：`workspace/gpu_verify.sh`
