# plan.md

## 项目目标
将 SpikingJelly 神经元后端从 `torch.autograd.Function` 逐步迁移到注册算子路径，
其中 Triton 主路径采用 `torch.library.triton_op`（保留 `custom_op` 回退），
以降低 `torch.compile` 的 graph break，提升融合与执行效率，并保持数值行为一致。

---

## 背景与问题
当前 CUDA/Triton 神经元后端大量依赖 `autograd.Function`。在 `torch.compile`（Dynamo/Inductor）场景中，
追踪进入函数内部时会遇到 CuPy 调用、`data_ptr()`、kernel 对象等不可追踪行为，导致 graph break。

主要影响：
1. 神经元 kernel 与 Conv/BN 等算子跨边界融合能力下降
2. kernel launch 难以被 CUDA Graph 稳定捕获
3. compile 优化收益受限

核心策略：
- Triton 路径采用 `triton_op`（配合 `wrap_triton`）作为默认实现
- 保留 `custom_op` 作为运行时 fallback 路径
- 先做 Triton（收益最高），再考虑 CuPy（可选）

---

## 决策记录（Decision Log）
- `accepted`: 基于 `torch>=2.6.0`，Triton 主线路径采用 `torch.library.triton_op`
- `accepted`: 保留现有 Triton `custom_op` 作为运行时回退，避免一次性切换风险
- `accepted`: Triton surrogate 在本阶段限制为 `Sigmoid` / `ATan`

影响范围：
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py`
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py`
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/plif.py`
- 对应 neuron 调用侧 LIF/IF/PLIF 的 Triton 分发逻辑

---

## 优先级
1. Triton 后端（高）
2. CuPy 后端（中，可选）
3. surrogate 梯度函数替代（低，当前不需要）

---

## 当前状态总览
- `surrogate.py` 重构：已完成
- Triton `triton_op` 主路径 + `custom_op` 回退：已完成并通过 compile 兼容测试
- 神经元模块分发切换（LIF/IF/PLIF）：已完成
- Triton wrapper 参数规则：已重构为接收 `SurrogateFunctionBase` 并在内部解析 `sg_triton_id + sg_alpha`
- compile 兼容测试：已新增并通过（`test/activation_based/test_compile_compat.py`）
- CI/GPU 运行时验证：核心用例已覆盖并通过（含 `test_triton_neuron.py`）
- 本地静态检查（全仓）：已通过（`uvx ruff check .`）

---

## 已完成（Phase 1）

### A. surrogate.py 清理与重构
- 删除所有 `@torch.jit.script`（含 `heaviside`）
- 删除 `MultiArgsSurrogateFunctionBase`，统一为 `SurrogateFunctionBase`
- 重设计参数管理：`spiking` + 动态参数注册表（`_sg_param_names`）
- 删除 `sg_type_name()`，调用侧改为 `type(surrogate).__name__`

### B. autograd.Function 新式 API 迁移
- 全部 18 个 surrogate Function 迁移到 `forward + setup_context + backward`
- 包含单参数、多参数和 pass-through 类型

### C. 历史缺陷修复
- 修复 `nonzero_sign_log_abs.backward` 参数错误打包问题

验证状态：
- surrogate 相关测试 `18/18` 通过（历史记录）
- 全仓 `ruff` 当前通过（2026-04-20，`uvx ruff check .`）

---

## 已完成（Phase 2-4）

### Phase 2-3：Triton `triton_op` 注册与回退路径
目标文件：
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py`
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py`
- `spikingjelly/activation_based/triton_kernel/neuron_kernel/plif.py`

实现约束：
1. 默认注册为 `torch.library.triton_op`，并通过 `torch.library.wrap_triton` 组织 kernel 调用
2. 保留 `custom_op` fallback，但不是默认性能路径
3. 每个 op 都提供：
   - `register_fake`（形状推断）
   - autograd 上下文与 `register_autograd`
4. 对外提供薄封装函数，仅暴露业务需要输出

surrogate 参数传递规范：
- neuron / wrapper 调用层按对象语义传递 `SurrogateFunctionBase`
- wrapper 内部解析为稳定参数：
  - `sg_triton_id: int`
  - `sg_alpha: float`
- 对非支持 surrogate 类型显式报错（`NotImplementedError`）

PLIF 特殊点：
- `r_tau` 为可学习 Tensor
- backward 需返回 `grad_r_tau`
- impl 返回 4 个张量：`(s_seq, v_seq, h_seq, v_init_v_seq)`

### Phase 4：神经元模块分发切换
目标文件：
- `spikingjelly/activation_based/neuron/lif.py`
- `spikingjelly/activation_based/neuron/integrate_and_fire.py`
- `spikingjelly/activation_based/neuron/plif.py`

迁移规则：
- 保持 neuron 层对外接口与现有行为兼容
- 调用侧继续负责拆分：
  - `v_reset: float` 与 `soft_reset: bool`
- surrogate 元信息由调用侧映射为内部 `sg_type` 与 `sg_alpha`
- 非 `Sigmoid/ATan` 时在 Triton 路径快速失败并给出清晰提示

---

## 待做（Phase 5-6）

### Phase 5（可选）：CuPy custom_op
- 目标对象：`IFNodeATGF`、`LIFNodeATGF`、`ParametricLIFNodeATGF`
- 目标：减少 graph break，提升 compile 图连通性

### Phase 6：测试与回归
已新增文件：
- `[done]` `test/activation_based/test_compile_compat.py`

已覆盖：
1. 本地（M 系列 Mac）仅做静态检查，不运行 Triton 内核
2. CI/GPU 环境执行 `torch._dynamo.explain(compiled_model)(x)` 断言 0 graph breaks
3. CI/GPU 环境执行 `torch.compile(..., backend="inductor")` 稳定执行（按版本调整配置）
4. CI/GPU 环境执行与 torch backend 前向/反向一致性对比
5. CI/GPU 环境验证 Triton 主/回退路径切换一致性（`triton_op` 与 `custom_op`）
6. 非支持 surrogate 报错路径验证

---

## 文件级任务清单

### 已完成
- `[done]` `spikingjelly/activation_based/surrogate.py`

### 已完成（本轮代码迁移）
- `[done]` `spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py`
- `[done]` `spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py`
- `[done]` `spikingjelly/activation_based/triton_kernel/neuron_kernel/plif.py`
- `[done]` `spikingjelly/activation_based/neuron/lif.py`
- `[done]` `spikingjelly/activation_based/neuron/integrate_and_fire.py`
- `[done]` `spikingjelly/activation_based/neuron/plif.py`

### 已完成（回归）
- `[done]` `test/activation_based/test_triton_neuron.py`

---

## 不在本次范围
- 不创建 `spikingjelly/activation_based/custom_ops/` 子包
- 不改 `spikingjelly/activation_based/cuda_kernel/spike_op.py`（独立优化路径）

---

## 验收标准（Definition of Done）
1. Triton 路径核心神经元默认走 `triton_op`，compile 无新增 graph break
2. `custom_op` fallback 路径可用且行为一致
3. 关键模型前向/反向数值与旧实现一致（容差内，CI/GPU 环境）
4. 新增 compile 兼容测试通过（CI/GPU 环境）
5. 本地静态检查通过（`uvx ruff check .`）
6. 文档与代码注释可解释 surrogate 参数与 reset 语义拆分
7. wrapper 对外只暴露 `SurrogateFunctionBase`，内部统一解析 `sg_triton_id + sg_alpha`

---

## 执行建议（每次提交最小闭环）
1. 先完成一个神经元类型（如 LIF）端到端迁移 + 测试
2. 再复制模式到 IF 与 PLIF
3. 本地每阶段结束后执行 `source .venv/bin/activate` + `uvx ruff check .`
