# Phase 5 详细执行手册（CuPy custom_op）

## 1. 目标与非目标

### 1.1 目标
- 将 `plan.md` 中 Phase 5（CuPy custom_op）从方向描述展开为可直接执行的实施手册。
- 在不改变公共 API 的前提下，规划 CuPy 路径迁移方案，减少 `torch.compile` 场景下 graph break 风险。
- 给出可操作的分阶段任务（M1-M5）、测试矩阵、回滚策略与观察指标，确保迁移可追踪、可验证、可回退。

### 1.2 非目标
- 不重排 `plan.md` 全局结构，不改动已完成的 Triton 主线规划。
- 不在本手册中实施代码变更；本文件仅定义 Phase 5 的执行方案。
- 不扩展到 `step_mode="s"` 的完整迁移闭环；单步模式仅标注后续扩展建议。
- 不修改无关模块（例如 `spikingjelly/activation_based/cuda_kernel/spike_op.py`）。

---

## 2. 当前现状与 graph break 来源

### 2.1 当前现状（基于仓库代码）
- Triton 路径已完成 `triton_op` 主路径 + `custom_op` 回退，并已有 compile 兼容测试。
- CuPy 相关逻辑仍以现有 `autograd.Function` 链路为主，核心对象集中在：
  - `spikingjelly/activation_based/cuda_kernel/auto_cuda/neuron_kernel.py`
  - `spikingjelly/activation_based/cuda_kernel/auto_cuda/ss_neuron_kernel.py`
- 神经元调用侧涉及：
  - `spikingjelly/activation_based/neuron/integrate_and_fire.py`
  - `spikingjelly/activation_based/neuron/lif.py`
  - `spikingjelly/activation_based/neuron/plif.py`

### 2.2 主要 graph break 风险点
- `autograd.Function` 内部执行包含不可追踪的运行时行为（如 kernel 对象、设备指针/底层调用路径）。
- CuPy kernel launch 及相关上下文切换行为对 Dynamo/Inductor 不透明，易形成图边界。
- surrogate 语义在运行时以对象行为驱动，若缺乏稳定标量参数桥接，编译路径稳定性不足。

### 2.3 迁移收益预期
- 降低 CuPy 后端在 `torch.compile` 下的图断裂概率。
- 提升 CuPy 与现有 compile 流程的协同性，减少后端切换时行为不一致风险。
- 为后续扩展（例如 `step_mode="s"`）保留一致的接口与参数传递范式。

---

## 3. 方案设计（custom_op 包装层 + 旧路径 fallback + 分发兼容）

### 3.1 总体设计
- 在 CuPy 路径引入 `custom_op` 包装层作为新执行路径。
- 保留现有 `autograd.Function` 实现作为运行时 fallback，避免一次性切换风险。
- 神经元调用层保持现有后端分发语义不变，对外仍以 `backend="cupy"` 使用。

### 3.2 目标对象与优先级
- 本阶段改造对象限定为：
  - `IFNodeATGF`
  - `LIFNodeATGF`
  - `ParametricLIFNodeATGF`
- 优先闭环 `step_mode="m"`（多步模式）；`step_mode="s"` 记录为后续增量任务。

### 3.3 surrogate 参数桥接策略
- 调用层继续接收 `SurrogateFunctionBase` 对象。
- 包装层内部将 surrogate 对象解析为稳定标量参数（例如 `sg_id` + `sg_alpha` 等固定字段）。
- 对不支持 surrogate 类型执行快速失败（`NotImplementedError`），并提供明确错误信息。

### 3.4 fallback 策略
- 默认尝试新 `custom_op` 路径。
- 在以下情形自动回退到旧 `autograd.Function` 路径：
  - 运行环境缺失关键能力（如 CuPy 不可用/注册失败）
  - surrogate 不在支持集合内
  - 显式环境开关要求回退（建议新增与 Triton 对齐的调试开关约定）
- fallback 行为需可观测（日志或调试标记），便于 CI 与问题定位。

---

## 4. 接口与行为约束

### 4.1 公共接口约束（必须保持）
- `IFNode` / `LIFNode` / `ParametricLIFNode` 的构造参数与调用方式不变。
- 神经元模块对外返回值语义不变（包括前向输出张量形状、训练反向路径可用性）。
- 不新增破坏性 API，不改变已有默认后端选择逻辑。

### 4.2 内部实现约束
- CuPy 改造仅作用于既定目标对象，不外溢至无关后端或模块。
- 新旧路径在数值语义上应保持一致（允许容差内差异），尤其是：
  - 输出张量
  - 输入梯度
  - `ParametricLIFNode` 的 `w.grad`
- 保留旧 `autograd.Function` 代码，不做破坏性删除，直至迁移稳定完成。

### 4.3 兼容性约束
- 与现有 Triton 迁移范式保持一致的 surrogate 参数风格（对象输入、内部标量化）。
- 保持对 `torch.compile`/Dynamo/Inductor 流程友好，不引入新的 compile 回归。

---

## 5. 分阶段实施清单（M1-M5）

### M1：现状基线与接口冻结
- 输入：
  - 当前 `backend="cupy"` 执行路径代码
  - 现有 compile 兼容测试结构（参考 Triton 侧）
- 输出：
  - CuPy 迁移基线说明（目标对象、当前调用链、待保留接口）
  - surrogate 支持白名单（首期建议：`Sigmoid` / `ATan`）
- 完成判据：
  - 公共 API 与行为冻结清单确认
  - fallback 触发条件清单确认

### M2：custom_op 包装层最小可用实现
- 输入：
  - M1 的接口冻结与 surrogate 白名单
- 输出：
  - `IFNodeATGF` / `LIFNodeATGF` / `ParametricLIFNodeATGF` 对应 `custom_op` 包装接口
  - 内部 surrogate 标量参数解析逻辑
  - fake/meta 与 autograd 注册框架（按现有工程约定）
- 完成判据：
  - 三类神经元在 `step_mode="m"` 下可走新路径执行
  - 运行时可显式识别当前是新路径还是 fallback 路径

### M3：调用层接线与回退闭环
- 输入：
  - M2 新增包装层
- 输出：
  - neuron 调用层完成 CuPy 新路径接线（保持 `backend="cupy"` 外部语义不变）
  - 回退开关与异常分支闭环（环境缺失、注册失败、不支持 surrogate）
- 完成判据：
  - 默认路径可执行，异常路径可稳定回退
  - 不支持 surrogate 场景报错清晰且可测试

### M4：一致性验证与 compile 验证
- 输入：
  - M3 功能闭环
- 输出：
  - 数值一致性结果（新路径 vs torch，必要时对比旧 CuPy 路径）
  - compile 关键用例结果（graph break、forward/backward 可执行性）
- 完成判据：
  - 关键测试全部通过
  - 无新增高优先级行为回归

### M5：稳定化与发布准备
- 输入：
  - M4 验证报告
- 输出：
  - 风险清单、回滚指引、观察指标阈值
  - 面向后续 `step_mode="s"` 的扩展建议
- 完成判据：
  - 迁移文档与测试矩阵可复用
  - 回滚路径在 CI/本地均可复现

---

## 6. 测试与验收矩阵

### 6.1 新增/补充测试场景
1. `torch._dynamo.explain` 下 CuPy 后端关键用例 graph break 数量对比（迁移前后）。
2. `torch.compile(..., backend="inductor")` 前向/反向可执行性（GPU 环境）。
3. CuPy 新路径 vs torch 后端数值一致性（输出与输入梯度，PLIF 额外校验 `w.grad`）。
4. fallback 路径一致性（强制新旧路径切换后输出/梯度一致）。
5. 非支持 surrogate 的快速失败与报错信息校验。

### 6.2 建议测试组织
- 复用 `test/activation_based/test_compile_compat.py` 的结构模式，按 CuPy 增加对应测试分组。
- 对需隔离环境变量/路径切换的用例，采用子进程执行避免状态污染。
- 对 GPU 依赖测试添加显式 skip 条件，避免非 CUDA 环境误报。

### 6.3 验收标准（Phase 5）
- CuPy 路径新增 `custom_op` 方案在目标对象上可执行并可回退。
- compile 关键用例无新增不可接受 graph break。
- 与 torch 参考路径在容差内保持输出与梯度一致。
- 非支持 surrogate 有清晰错误提示且覆盖测试。

---

## 7. 风险、回滚与观察指标

### 7.1 主要风险
- CuPy 运行时依赖差异导致环境相关不稳定（驱动、版本、设备上下文）。
- compile 行为在不同 torch 版本上的差异导致测试波动。
- surrogate 参数映射不一致引发隐性数值偏移。

### 7.2 回滚策略
- 保留旧 `autograd.Function` 路径，确保一键回退可用。
- 提供显式开关强制回退，作为故障隔离手段。
- 回滚触发条件：
  - 关键一致性测试失败
  - compile 用例出现持续性高频 graph break 回归
  - 线上/CI 出现不可接受稳定性问题

### 7.3 观察指标
- graph break 数量（按模型与后端维度统计）。
- 编译成功率与首轮执行成功率。
- 输出/梯度一致性误差分布（max/mean）。
- fallback 触发频次及原因分布（用于判定新路径成熟度）。

---

## 8. 任务排期建议（最小闭环优先）

### 8.1 推荐节奏
1. 第一阶段（最小闭环）：`LIFNodeATGF` 在 `step_mode="m"` 完成 M1-M4。
2. 第二阶段（模式复制）：将同一模式扩展到 `IFNodeATGF` 与 `ParametricLIFNodeATGF`。
3. 第三阶段（稳定化）：完成 M5，整理回滚与观察指标，形成可持续维护基线。

### 8.2 每次提交建议
- 每个 PR 保持单一目标（一个里程碑或一个神经元类型）。
- 每次提交都包含最小必要验证（相关测试 + 结果摘要）。
- 不在同一 PR 内混入无关重构。

---

## 附录：执行默认假设
- 文档路径固定为仓库根目录：`plan_phase5_cupy.md`。
- 本轮仅展开 Phase 5，不包含 Phase 6 全量重排。
- `step_mode="m"` 为当前闭环优先路径；`step_mode="s"` 作为后续扩展项。
- 遵循最小正确改动原则：不改无关模块，不调整公共 API。
