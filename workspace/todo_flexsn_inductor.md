# FlexSN Inductor 后端 — 后续 TODO

**分支**：`feature/flexsn-inductor-backend`
**最后更新**：2026-04-21

---

## P0 — 已完成

| 项目 | 状态 |
|------|------|
| M1: HOP + eager scan | ✅ |
| M2: AOTAutograd backward | ✅ |
| M3.a: torch.compile 端到端 | ✅ |
| M3.b: 推理单核 scan（`tl.static_range(T)`） | ✅ |
| M3.b: 训练 Triton fwd+bwd 核 | ✅ |
| `torch.compile()` 训练不再退化（`@dynamo.disable`） | ✅ |
| FX_TO_TRITON 算子覆盖扩展（40+ ops） | ✅ |
| `template.py` n=0 backward bug fix | ✅ |
| PR #658 code review 修复（logical_*/zeros_like/import guard 等） | ✅ |
| M5: API + 测试 + 文档 | ✅ |

---

## P1 — 训练/推理性能

### 真正的突触+神经元内核融合

**背景**：当前 FlexSN 无论如何都是独立的 Triton 核，与前后 Conv/Linear 之间存在 HBM 往返。
要消除这个边界需要三个条件同时满足：

1. **Linear 由 Inductor 控制**（目前走 cuBLAS）
   - 可尝试 `torch._inductor.config.force_disable_caches = True` 或
     `torch._inductor.config.use_mixed_mm = False`，强制 Inductor 自己生成矩阵乘

2. **FlexSN 注册为 Inductor 原生算子**（M3.b 完整方案，下方详述）

3. **跳过 BN 或用 BNTT**
   - 普通 BN 是归约算子，形成天然融合屏障
   - BNTT（逐时步 BN）可消除此边界

**预估工作量**：2–3 周

---

### M3.b 完整方案：FlexSN scan 注册为 Inductor 原生算子

**目标**：`torch.compile(model, fullgraph=True)` 下，Linear epilogue 和 FlexSN
scan prologue 合并为一个 Triton 核，消除 HBM 往返。

**技术路径**（参考 FlexAttention 实现）：

```
1. torch.library.custom_op("spikingjelly::flex_sn_scan", ...)
   - register_fake: shape/dtype 推断
   - CUDA 实现：调用现有 Triton fwd+bwd 核

2. torch._inductor.lowering.register_lowering(flex_sn_scan)
   - 用 make_fx 追踪 core_fn 得到 FX 图
   - 用 PointwiseSubgraphLowering lower 得到 Inductor IR 节点
   - 构造 TritonTemplate：外层 tl.static_range(T) + 内层 core_fn body
   - 返回 TritonTemplateCaller 供 Inductor scheduler 调度和融合

3. flexsn.py: 移除 @torch._dynamo.disable，改用 custom_op 调用
   - Dynamo 原生支持 custom_op，无需 graph break
   - fullgraph=True 下也可工作
```

**关键文件参考**：
- `torch/_inductor/kernel/flex_attention.py` — TritonTemplate 完整示例
- `torch/_inductor/select_algorithm.py` — TritonTemplate/TritonTemplateCaller API
- `torch/_inductor/subgraph_lowering.py` — PointwiseSubgraphLowering

**风险**：
- TritonTemplate API 跨版本稳定性（锁定 PyTorch 2.7.x）
- `make_fx` 追踪 core_fn 生成的 FX 图需要全部在 PointwiseLowering 可表达范围内
- backward TritonTemplate 更复杂（需要保存中间值、逆序扫描）

**预估工作量**：2 周

---

## P2 — 训练效率

### Gradient Checkpointing

**背景**：当前训练路径保存全部 T 个时步的中间激活（正向核的 `c2k_return_mapping`
保存的中间值）。T 较大时显存占用与 T 成正比。

**方案**：将 T 步扫描分成若干 segment，每段用 `torch.utils.checkpoint.checkpoint`
包装——反向时重算该段的前向，用计算换显存。

**实现位置**：`FlexSN.multi_step_forward`，可加 `gradient_checkpointing=True` 参数。

**预估工作量**：3–5 天

---

### AMP（自动混合精度）支持完善

**背景**：`FlexSNFunction` 继承自 triton 后端，有 `@amp_custom_fwd/bwd`，但
inductor 训练路径通过 `@dynamo.disable` 调用，AMP context 可能未正确传递。

**验证方法**：
```python
with torch.autocast('cuda', dtype=torch.float16):
    out = neuron(x)  # 确认 core_fn 内部用 fp16 运算
```

**预估工作量**：1 天

---

## P3 — 易用性

### 状态张量复用（避免重复 `zeros_like`）

**背景**：每次 `reset_net` 后 `self.states = None`，下一次 `forward` 调用
`init_states` 重新 `torch.zeros_like` 分配。对 VGG 这类 13 层网络，每轮训练
有 13 次内存分配。

**方案**：shape 不变时用 `zero_()` 原地清零复用，避免 GPU 内存分配。

```python
def _reset_states(self):
    if self.states is None:
        return  # 等 forward 时 init
    for s in self.states:
        s.zero_()  # 原地清零，不重分配
```

**预估工作量**：半天

---

### `torch.compile(fullgraph=True)` 训练支持

**背景**：目前 `fullgraph=True` 下训练会报错（`@dynamo.disable` 不允许 graph break）。
上面 M3.b 完整方案（custom_op）解决后此问题自动消除。

---

## P4 — 生态扩展

| 项目 | 说明 |
|------|------|
| DDP / FSDP 兼容性验证 | FlexSNFunction 的 grad 同步需要验证 |
| AOTInductor 导出（`.so`） | 推理部署，需 custom_op 先完成 |
| ROCm / XPU | 目前仅 CUDA；需替换 `tl.extra.cuda.libdevice` 调用 |
| `FX_TO_TRITON` 继续扩展 | `exp2`、`log1p`、`atan`、`asin` 等；按需添加 |

---

## 参考

- 设计文档：`workspace/design_flexsn_inductor.md`
- GPU 验证脚本：`workspace/gpu_verify.sh`
- PR #658：`https://github.com/fangwei123456/spikingjelly/pull/658`
