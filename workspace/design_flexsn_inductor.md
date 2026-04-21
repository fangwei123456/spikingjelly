# FlexSN Inductor Backend — 设计文档（v0.2）

**作者**: wei.fang
**日期**: 2026-04-20
**状态**: 评审已通过第 7 节开放问题，待开工
**分支**: `feature/flexsn-inductor-backend`
**目标平台**: NVIDIA GPU（CUDA）。不考虑 CPU / 非 CUDA 后端

---

## 1. 背景与动机

当前 [FlexSN](spikingjelly/activation_based/neuron/flexsn.py) 通过 "torch.fx 追踪用户 `core` → 自研 FX→Triton 映射表 → 拼接 Triton 内核模板" 的方式，把用户自定义的单步神经元函数编译成多步 Triton 内核。

该路径的局限：

| 问题 | 现状 |
|------|------|
| 算子覆盖 | 需要手动维护 [FX_TO_TRITON 映射表](spikingjelly/activation_based/triton_kernel/torch2triton/graph2triton.py)；用户 `core` 里一旦出现表外算子即失败 |
| 跨层融合 | FlexSN 是 autograd.Function 黑盒，`Linear→FlexSN→Linear` 无法被编译器看穿 |
| 生态隔离 | 必须设置 `PYTORCH_JIT=0`；与 torch.compile / AOTInductor / CUDA Graphs 不协同 |
| 后端扩展 | 只支持 NVIDIA GPU；ROCm/XPU/Metal 需各自重写 |
| autograd 维护 | 需要手写 [FlexSNFunction.backward](spikingjelly/activation_based/triton_kernel/flexsn/wrapper.py) |

**本提案**：把 FlexSN 重构为一个注册到 PyTorch Inductor 的 HigherOrderOp（HOP），由 Inductor 负责 `core` 内部算子的 lowering，SpikingJelly 只贡献"时间步 scan"这一语义。参考 `FlexAttention` 的实现路径。

## 2. 目标与非目标

### 2.1 目标（POC 阶段）

- **G1**：LIF 神经元在 Inductor 后端下跑通 forward + backward，数值对齐现有路径（相对误差 < 1e-5）
- **G2**：生成的 Triton 内核在 T=8/32, batch=128, hidden=1024 配置下性能不劣于现有 [FlexSN Triton 路径](spikingjelly/activation_based/triton_kernel/flexsn/wrapper.py) 的 0.9×
- **G3**：`Linear → FlexSN(LIF) → Linear` 在 `torch.compile` 下能被 Inductor 融合或至少消除中间 HBM 往返
- **G4**：保留现有 `backend="triton"` 作为基线，在 [FlexSN](spikingjelly/activation_based/neuron/flexsn.py) 新增 `backend="inductor"`；默认不改

### 2.2 非目标

- **NG1**：不追求替代现有 [torch2triton](spikingjelly/activation_based/triton_kernel/torch2triton/) 体系
- **NG2**：POC 只支持 LIF 一种神经元；不处理任意 `num_states/num_inputs` 组合
- **NG3**：不支持动态时间步长 T
- **NG4**：不处理 DDP/FSDP/checkpoint 交互（留作后续）
- **NG5**：不支持 CPU / ROCm / XPU / Metal；仅 CUDA

## 3. 整体架构

```
                ┌─────────────────────────────────────────┐
                │  用户代码                                │
                │  neuron = FlexSN(core=lif_core,         │
                │                   backend="inductor")   │
                │  model = torch.compile(model)  ← 必须   │
                └────────────────┬────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │  flex_sn_scan (HigherOrderOp)           │  ← 新增
                │  - eager fallback (Python for 循环)     │
                │  - AOTAutograd registration             │
                └────────────────┬────────────────────────┘
                                 │ torch.compile
                                 ▼
                ┌─────────────────────────────────────────┐
                │  Inductor Lowering                      │  ← 新增
                │  lower_flex_sn_scan(core_subgraph, ...) │
                │  - 生成外层时间循环                       │
                │  - 内层复用 Inductor 对 core 的 codegen │
                └────────────────┬────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │  Triton kernel (autotuned)              │
                └─────────────────────────────────────────┘
```

## 4. 详细设计

### 4.0 新增文件布局

```
spikingjelly/activation_based/triton_kernel/flex_sn_inductor/
├── __init__.py
├── hop.py           # HigherOrderOp 定义 + eager 实现
├── autograd.py      # AOTAutograd 注册
├── lowering.py      # Inductor lowering (TimeLoopScanKernel)
└── ir.py            # 自定义 Inductor IR 节点
```

与现有 [triton_kernel/flexsn/](spikingjelly/activation_based/triton_kernel/flexsn/) 并列，互不干扰。

### 4.1 HigherOrderOp 定义

新文件 `spikingjelly/activation_based/triton_kernel/flex_sn_inductor/hop.py`：

```python
from torch._higher_order_ops.utils import HopInstance
from torch._ops import HigherOrderOperator

class FlexSNScan(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_sn_scan")

    def __call__(self, core_fn, inputs_seq, init_states, T):
        # inputs_seq: tuple of [T, ...] tensors
        # init_states: tuple of [...] tensors
        # 返回: (outputs_seq, final_states) —— 语义与现 FlexSN 多步一致
        return super().__call__(core_fn, inputs_seq, init_states, T)

flex_sn_scan = FlexSNScan()
```

通过 `flex_sn_scan.py_impl(DispatchKey.CompositeExplicitAutograd)` 注册 eager 实现：纯 Python for 循环。

### 4.2 AOTAutograd 接入

- `flex_sn_scan.py_autograd_impl(...)`：用 `torch.autograd.Function` 包装，forward 调 HOP，backward 反向展开时间循环。
- AOTAutograd 会将 `core_fn` 的前向图和反向图分别追踪，产物供 Inductor lowering 使用。
- 初期可参考 `torch._higher_order_ops.scan` 的 autograd 实现（若 API 可用）。

### 4.3 Inductor Lowering

新文件 `spikingjelly/activation_based/triton_kernel/flex_sn_inductor/lowering.py`：

```python
from torch._inductor.lowering import register_lowering

@register_lowering(flex_sn_scan)
def lower_flex_sn_scan(core_subgraph, inputs_seq, init_states, T):
    # 1. 分配 outputs_seq / state 缓冲
    # 2. 发射一个自定义 InductorIR 节点（TimeLoopScan），该节点在
    #    codegen 时包裹外层 `for t in tl.static_range(T)` 时间循环，
    #    内部调用 Inductor 对 core_subgraph 生成的 pointwise kernel body
    # 3. 依赖 Inductor scheduler 处理后续融合
    ...
```

**关键技术点**：
- Inductor IR 没有现成的"带状态的 scan"节点。需要定义一个 `TimeLoopScanKernel`（继承 `torch._inductor.ir.Loops` 或 `ComputedBuffer`），在其 `codegen` 方法里手写时间循环外壳，`inner_fn` 使用 Inductor 已 lower 好的 `core_subgraph`。
- 状态变量通过 Triton 寄存器保持（`v = tl.load(...)` 一次，循环内更新，最后 `tl.store(...)`），对应 Inductor 里 `StorageBox` 的 in-place 语义。

### 4.4 对外 API 扩展

在 [FlexSN.__init__](spikingjelly/activation_based/neuron/flexsn.py) 中新增 `backend="inductor"` 分支：

```python
if self.backend == "inductor":
    from ..triton_kernel.flex_sn_inductor import flex_sn_scan
    self._scan_fn = flex_sn_scan
elif self.backend == "triton":
    self.kernel = FlexSNKernel(...)  # 现有路径，保留作为基线
```

`multi_step_forward` 分支：

```python
if self.backend == "inductor":
    outs, new_states = self._scan_fn(self.core, inputs_seq, self.states, T)
    # 模块内部不调用 torch.compile；由用户在外层显式套 compile
    ...
```

**用户使用契约（强制）**：

```python
neuron = FlexSN(core=lif_core, backend="inductor", ...)
model = nn.Sequential(Linear(...), neuron, Linear(...))
model = torch.compile(model, fullgraph=True)  # 必须；否则退化为 eager HOP，性能无保证
```

理由：FlexSN 内部自动 `torch.compile` 会切断与外层模型的联合编译，失去跨层融合（G3）的关键价值。

### 4.5 编译失败策略

Inductor lowering 抛错时**直接上抛**，不静默回落 eager。原因：

- 静默回落会掩盖用户 `core` 里的不兼容写法，后期调试困难
- 性能退化不易察觉

文档中需明确告知用户：遇到 `FlexSNInductorCompileError` 时，检查 `core` 是否使用了 Inductor 不支持的算子，或临时切回 `backend="triton"`。

## 5. 版本锚定

| 依赖 | 锚定版本 | 原因 |
|------|---------|------|
| PyTorch | **2.11.0 + CUDA 构建** | HOP + Inductor 私有 API 跨版本易 break；锚定后观察 |
| Triton | 随 PyTorch 携带 | 不单独锚 |
| Python | ≥ 3.10 | HOP 使用的 typing 特性 |
| CUDA | ≥ 12.0 | Triton 后端的标准要求 |

在 [pyproject.toml](pyproject.toml) 增加可选依赖组：

```toml
[project.optional-dependencies]
flexsn-inductor = ["torch==2.11.0"]
```

**开发与 CI 环境**：使用带 NVIDIA GPU 的机器（本地 macOS 开发机只能做非 GPU 部分的代码审阅 / 单元测试骨架；Inductor lowering 的实测必须在 CUDA 环境进行）。

## 6. 里程碑

| # | 目标 | 预计工作量 | 验证 |
|---|------|----------|------|
| M1 | HOP 定义 + eager 实现（Python for 循环） | 2–3 天 | 单元测试：与现 LIF 数值一致 |
| M2 | AOTAutograd 注册；eager 下 backward 打通 | 2–3 天 | 梯度数值对齐 |
| M3 | Inductor lowering（TimeLoopScanKernel） | 1–2 周 | `torch.compile` 产物可运行，输出生成的 Triton 代码快照 |
| M4 | 性能对比与融合验证 | 3–5 天 | benchmark：纯 FlexSN、`Linear→FlexSN→Linear`、混合网络 |
| M5 | API 集成 + 测试 + 文档 | 2–3 天 | `backend="inductor"` 开关，tutorial 示例 |

**总计**：约 3–4 周（单人）。M1–M2 完成后 go/no-go 决策点。

## 7. 风险与开放问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| Inductor IR 不支持寄存器级 scan | 高：整个路线受阻 | 调研 `torch._higher_order_ops.scan` 是否已有 Inductor 支持；若有直接复用；若无，参考 FlexAttention 的 `TritonTemplate` 机制 |
| torch 2.11 私有 API 变动 | 中 | POC 期间锁死版本；正式发布前做 2–3 个版本兼容性测试 |
| `PYTORCH_JIT=0` 约束能否解除 | 中 | Inductor 路径不再依赖自研 FX 追踪，理论上可解除；待 M1 验证 |
| backward 时间循环反向展开的内存开销 | 中 | 先不做 gradient checkpointing；M4 benchmark 观察 |
| Triton autotune 配置迁移 | 低 | Inductor 有自己的 autotune，POC 不手工调优 |

### 开放问题（评审已决议）

1. ~~HOP 放置位置~~ → **`spikingjelly/activation_based/triton_kernel/flex_sn_inductor/`**（与现有 `triton_kernel/flexsn/` 平级）
2. ~~是否要求用户显式 `torch.compile`~~ → **要求显式**。模块内部不做自动 compile
3. ~~失败降级策略~~ → **直接报错**，不静默回落
4. ~~命名~~ → **`FlexSN(backend="inductor")`**（沿用现有 backend 开关）

## 8. 测试计划

- **单元测试**：`test/activation_based/neuron/test_flex_sn_inductor.py`
  - LIF 前向输出与 `FlexSN(backend="triton")` 对齐
  - LIF 梯度与 PyTorch eager 对齐
  - T=1/8/32，batch=1/128，hidden=32/1024 组合
- **Benchmark**：`benchmark/flex_sn_inductor.py`
  - 纯 FlexSN 单层
  - `Linear → FlexSN → Linear` 融合验证（对比生成 Triton 代码是否出现融合）
  - 与 `backend="triton"` 基线对比
- **回归**：现有 [test](test/) 全部通过，`backend="triton"` 不受影响

### 8.1 GPU 验证脚本 `gpu_verify.sh`

由于本地 macOS 无法跑 CUDA，提供了 [gpu_verify.sh](gpu_verify.sh)（untracked，不入 git），在 NVIDIA GPU 主机上一键验证本分支。

**用法**：

```bash
bash gpu_verify.sh                 # 全部 5 个步骤
bash gpu_verify.sh --skip-bench    # 跳过 benchmark（仅正确性验证）
bash gpu_verify.sh --only parity   # 只跑某一步
```

可选步骤名：`env` / `tests` / `triton-emit` / `parity` / `bench`

**五个步骤**：

| # | 步骤 | 目的 | 失败即退出 |
|---|------|------|----------|
| 1 | `env` | 检查 `torch` / CUDA / `triton` 是否可用 | ✓ |
| 2 | `tests` | 跑 `test/activation_based/test_flex_sn_inductor.py` 全部 36 个测试 | ✓ |
| 3 | `triton-emit` | `TORCH_LOGS=output_code` 抓 Inductor 生成的内核，断言出现 `@triton.jit` | ✓ |
| 4 | `parity` | `backend="triton"` vs `backend="inductor"` 数值对齐（`atol=1e-5`）| ✓ |
| 5 | `bench` | 3 个配置下耗时对比，按 G2 判定标准自动输出 `OK / CLOSE / CONSIDER M3.b` | 否 |

**实现要点**：
- `set -euo pipefail` + 每步 `[OK]/[FAIL]` 标记
- 所有输出落盘到 `.gpu_verify/`，事后可翻查
- benchmark 3 次预热后再计时，避开 Inductor 首次编译开销
- G2 判定标准（`ratio ≤ 1.1× = OK` / `≤ 1.5× = CLOSE` / `> 1.5× = CONSIDER M3.b`）内置在输出 flag 列里

**决策流**：

```
bench.log ratio ≤ 1.1   → M3.b 不必做；直接进 M5
         1.1 < ratio ≤ 1.5 → 记录待观察；进 M5，M3.b 作为可选优化
         ratio > 1.5       → 触发 M3.b 单节点 Inductor lowering 立项
```

## 9. 后续工作（POC 之后，不在本次范围）

- 扩展到任意 `num_states/num_inputs/num_outputs`
- 支持 PLIF/IF/QIF 等神经元
- DDP/FSDP 兼容性
- AOTInductor 部署（导出 `.so`）
- 跨后端（ROCm/XPU）测试

---

## 评审结论（2026-04-20）

- [x] 整体方向认可
- [x] 里程碑拆分合理
- [x] 开放问题 1–4 已决议
- [x] 目标平台：NVIDIA GPU only，锚定 torch 2.11.0

**下一步**：开 `feature/flexsn-inductor-backend` 分支，从 M1 开工。

---

## M3.b 详细设计（2026-04-20 立项）

### 背景

M1–M3.a + M5 验收后，GPU benchmark 显示（RTX 4090）：

| 配置 | inductor / triton 比值 | 结论 |
|------|----------------------|------|
| T=8, B=128, N=1024 | 0.87× | 更快 |
| T=32, B=128, N=1024 | **2.04×** | 不可接受 |
| T=8, B=128, N=4096 | 1.15× | 可接受 |
| Linear→FlexSN→Linear, T=32 | **2.00×** | 不可接受 |

根本原因：当前 M3.a 路径在 `torch.compile` 下把 `eager_scan` 的 Python `for t in range(T)` 循环展开成 T 份 `core_fn` aten 算子序列，Inductor 为每步生成独立 Triton kernel，T=32 时产生 32 × (kernel launch + HBM 往返)。而 `backend="triton"` 的单核方案只有 1 次 launch，时间循环在寄存器内推进。

### 目标

用单个 Triton 内核替代展开的多内核方案，使 T=32 的性能比值回到 ≤ 1.1×（G2 标准）。

### 技术方案

#### 方案 A：TritonTemplate（推荐）

参考 `FlexAttention`（`torch/_inductor/kernel/flex_attention.py`）的实现路径：

1. 用 `torch.library.Library` 把 `flex_sn_scan` 注册为正式 custom op，脱离 HigherOrderOperator。
2. 用 `@torch.library.register_fake` 注册 meta kernel（shape/dtype 推断）。
3. 用 `torch._inductor.lowering.register_lowering` 注册 Inductor lowering，在里面：
   a. 用 `make_fx` 追踪 `core_fn` 得到 FX 图；
   b. 用 Inductor 的 `pointwise_lowering` 把 FX 图里的每个节点 lower 成 `OpsHandler` 调用；
   c. 拼接成一个 `TritonTemplate`，外层套 `for t in tl.static_range(T)` 时间循环，内层嵌入 lower 后的 `core_fn` body；
   d. 状态张量用 `tl.load` 一次读入寄存器，循环内原地更新，最后 `tl.store`。
4. `TritonTemplate` 向 Inductor scheduler 注册，参与正常的 autotune 和 kernel fusion。

**关键参考**：
- `torch/_inductor/kernel/flex_attention.py`：FlexAttention 的完整 TritonTemplate 实现
- `torch/_inductor/lowering.py`：`register_lowering` 用法
- `torch/_inductor/ops_handler.py`：`OpsHandler` / pointwise IR

#### 方案 B：自定义 Inductor IR 节点（备选）

继承 `torch._inductor.ir.Loops` 或 `ExternKernel`，在 `codegen` 方法里直接 emit 时间循环 Triton 代码。比方案 A 更底层，API 变动风险更高，优先级低。

#### 方案 C：`torch.vmap` 改写（放弃）

`vmap` 无法自然表达带状态的 scan（状态需跨步传递），放弃。

### 新增文件

```
spikingjelly/activation_based/triton_kernel/flex_sn_inductor/
├── hop.py          # 已有；改为 custom op 注册（torch.library）
├── lowering.py     # 新增：Inductor lowering + TritonTemplate
└── template.py     # 新增：scan kernel 模板字符串 + tl.static_range 外壳
```

### 实现步骤

| 步 | 内容 | 验证 |
|----|------|------|
| 1 | 将 `FlexSNScan` 从 HOP 改为 `torch.library` custom op；保持 eager fallback | 现有 36 个单元测试仍全通过 |
| 2 | 实现 `template.py`：写死 LIF 版 scan 内核，用 `tl.static_range(T)` | 手动 `triton.compile` 验证输出正确 |
| 3 | 实现 `lowering.py`：`make_fx` 追踪 `core_fn` + Inductor pointwise lower + 拼 template | `TORCH_LOGS=output_code` 验证生成单个 `@triton.jit` |
| 4 | 参数化 template：支持任意 `core_fn`（不止 LIF） | `test_flex_sn_inductor.py` 扩展测试 |
| 5 | 性能验证 | `gpu_verify.sh --only bench`：T=32 ratio ≤ 1.1× |

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| `TritonTemplate` API 在 PyTorch 2.7 后变动 | 中 | 锁定 2.7.1；接口层加 version guard |
| `make_fx` 追踪 `core_fn` 遇到不可追踪算子 | 中 | 沿用 torch2triton 的算子映射表作为 fallback |
| 状态变量 in-place 语义与 Inductor alias analysis 冲突 | 中 | 参考 FlexAttention 的 output buffer 管理方式 |
| 自动 autotune block_size 选择不佳 | 低 | 初期固定 block_size=128；M4 阶段再调优 |

### 预计工作量

约 2 周（单人）。步骤 1–3 完成后有中间验收点。
