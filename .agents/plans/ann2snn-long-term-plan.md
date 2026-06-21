# SpikingJelly ann2snn 模块长期改进计划

> 创建日期：2026-06-18
> 最近更新：2026-06-21
> 状态：Phase 0 已通过 PR #691 合并；Phase 0.5 已完成，下一步进入 Phase 1 设计复核

## 目标

将 SpikingJelly 的 `ann2snn` 模块从单一的 ReLU→IFNode 转换器，升级为：

1. **一套通用转换框架**：用清晰的 extension points 支撑大多数 ANN2SNN 算法，而不是把某篇论文的细节硬编码进 `Converter`。
2. **几个高价值算法实例**：内置少数有名、效果好、泛用性强的 ANN2SNN conversion recipes，覆盖从经典 CNN/MLP 到 Transformer/LLM 的主要使用场景。

> [!note]
> “通用框架”不等于承诺一个单一接口可以无损表达所有论文。更现实的目标是：公共转换流水线稳定，核心扩展点可组合；遇到本质不同的 execution strategy 时，通过独立 strategy/helper 接入，而不是污染默认 `Converter`。

## 框架与算法实例的分层

长期设计应明确分成三层：

| 层级 | 责任 | 示例 |
|------|------|------|
| Framework core | 负责 FX tracing、module matching、hook insertion、calibration data flow、replacement orchestration、向后兼容 | `Converter`, `ActivationRule`, `NeuronFactory`, `ThresholdOptimizer` |
| Reusable primitives | 把论文中的通用机制落成可测组件，可被多个算法实例复用 | spike-equivalent operators, channel-wise threshold, outlier-aware scaling, encoding/decoding, neuron dynamics |
| Algorithm recipes | 把一组 rules / factories / optimizers / operators / calibration hooks 组合成用户可直接调用的方案 | rate coding baseline, Transformer spike-equivalent conversion, AAR-style redistribution, LAS-style LLM conversion |

评估每个新算法时，优先判断它贡献的是：

- 一个新的 framework extension point；
- 一个可复用 primitive；
- 一个 recipe 组合；
- 还是仅适合作为研究 track 的一次性实验。

## 方法选择结论（2026-06-21 调研）

Differential Coding 不是当前最有代表性的主线。近年来真正面向 Transformer / 大模型的 ANN2SNN 进展，核心不在于单一低延迟编码，而在于**把 Transformer 的非线性和注意力组件转换为 spike-equivalent 形式**，并处理大模型常见的 activation outliers。

因此，SpikingJelly 更应优先实现：

> **Transformer/LLM spike-equivalent conversion family**：以 SpikeZIP-TF / Activation-Aware Redistribution / LAS 为主线，逐步支持 Spike-Softmax、Spike-LayerNorm、Spike-GELU、spike-equivalent attention、outlier-aware threshold / neuron 等能力。

其中，**LAS-style fully spike-driven LLM conversion** 是长期旗舰目标；但它复杂度高，不应一步到位。更合理的路线是先实现通用 Transformer operator primitives，再进入 LAS 的 OAT/HG neurons 和 full LLM conversion。

## 内置算法实例选择标准

不建议把每篇 ANN2SNN 论文都做成内置算法。进入 SpikingJelly 主线的算法实例至少应满足多数条件：

1. **代表性强**：对应 ANN2SNN 的主流问题，而不是只优化单一 niche case。
2. **泛用性强**：能覆盖常见架构族，例如 CNN/MLP、ViT/BERT、decoder-only LLM/VLM，而不是只服务一个 benchmark。
3. **效果有说服力**：精度、延迟、能耗或转换稳定性有明确优势。
4. **能沉淀 primitives**：实现过程中产生的组件能被后续算法复用。
5. **API 风险可控**：不会迫使 `Converter` 过早承诺不稳定参数。

按这个标准，近期更合理的内置实例候选是：

| 算法实例 | 定位 | 进入主线的理由 |
|----------|------|----------------|
| Classic rate-coding ReLU→IF conversion | 默认 baseline | 已存在、兼容性最高、作为所有新算法的对照组 |
| Transformer spike-equivalent conversion | Transformer/ViT/BERT 基础实例 | 覆盖 Softmax / LayerNorm / GELU / attention 等关键障碍，primitives 可复用 |
| Activation-Aware Redistribution style conversion | outlier/channel-wise threshold 实例 | 面向 Transformer/BERT 的 activation outlier 和 channel imbalance，补齐大模型转换基础设施 |
| LAS-style fully spike-driven LLM conversion | 长期旗舰实例 | 直接面向 LLM/VLM，覆盖 OAT/HG neurons 与 spike-equivalent LLM components |

Negative Spikes、Differential Coding、SignGD、Parallel Spiking 仍有研究价值，但暂不作为第一批内置主线实例。

## 执行原则（来自 Phase 0 / PR #691 的经验）

1. **先做可验证的最小垂直切片**：每个新算法先支持一个最小模型、一个校准路径、一个端到端输出 shape/数值 sanity test，再扩展到论文完整能力。
2. **不要在一个 PR 里同时改架构和算法**：Phase 0 已完成架构抽象；后续 PR 应围绕单个算法或单个公共能力展开。
3. **把论文公式落成独立可测组件**：阈值优化、编码/解码、神经元动力学、联合校准各自先有单元测试，再接入 `Converter`。
4. **严格区分“默认行为”和“实验策略”**：`Converter(dataloader, mode="Max")` 的 ReLU→IFNode 默认路径必须持续保持兼容；新算法通过显式 rule/factory/optimizer 或 helper strategy 接入。
5. **先不承诺 per-channel threshold**：当前实现只对标量 threshold 路径有明确语义和测试。任何多元素阈值都需要先定义 shape、broadcast 和 `VoltageScaler` 行为。
6. **所有 agent/review 发现必须 triage 后再改**：Phase 0 的 review loop 证明，很多建议是 nit 或范围扩张；后续应继续只修真实问题。

## 当前进展

| Phase | 类型 | 内容 | 状态 |
|-------|------|------|------|
| Phase 0 | Framework core | 可扩展框架搭建（规则系统 + 工厂模式 + 阈值优化接口） | ✅ 已完成并合并 |
| Phase 0.5 | Framework hardening | 稳定扩展点与文档同步 | ✅ 已完成 |
| Phase 1 | Reusable primitives | Transformer spike-equivalent operator foundation | 下一步 gate |
| Phase 2 | Reusable primitives + recipe POC | Activation/outlier-aware threshold 与 channel-wise redistribution | 待启动 |
| Phase 3 | Algorithm recipe | LAS-style fully spike-driven LLM conversion | 待启动 |
| Phase 4 | Public recipes | 策略 helper API | 待启动 |
| Phase 5+ | Research tracks | 其他研究 track（Negative Spikes / Differential Coding / SignGD / Parallel Spiking） | 待评估 |

---

## Phase 0：可扩展框架搭建（✅ 已完成）

搭建规则系统基础设施，保持向后兼容。所有后续 Phase 的基础。

### 已完成内容

- `rules.py`：`ActivationRule` Protocol + `ReLURule` 实现
- `factories.py`：`NeuronFactory` + `HookFactory`
- `threshold.py`：`ThresholdOptimizer`（"fixed" 策略）
- `converter.py`：重构为规则驱动，新增 `rules`/`neuron_factory`/`threshold_optimizer` 可选参数
- `__init__.py`：导出新公共 API
- `test_ann2snn.py`：58 个测试用例，覆盖向后兼容、规则系统、工厂、阈值优化、异常路径和端到端

### 设计要点

- `ActivationRule` 用 Protocol 而非 ABC，支持 duck-typing；当前协议包含 `match`、`insert_hooks`、`find_replacements`、`replace_with_neurons`
- `NeuronFactory.create(scale)` 接收 `scale` 但默认不用，为阈值优化留口子
- `ThresholdOptimizer.compute_threshold(hook)` 返回 `hook.scale`，后续可扩展 `bn_aware` 等策略
- `Converter` 旧参数全部保留默认值，零破坏性变更
- `replace_by_ifnode()` 已 deprecated，内部委托给 `_replace_by_neurons_impl()`
- `set_voltagehook()` 在规则插入 hook/子模块后刷新 `modules`，避免后续规则看到 stale dict
- 校准输入已支持 tensor、tuple/list、dict、numpy/array-like；空容器抛清晰 `ValueError`

---

## Phase 0.5：稳定扩展点与文档同步（✅ 已完成）

Phase 0 已能支撑自定义 rule/factory/threshold optimizer，但真正接入新算法前，建议先补齐开发者体验和边界文档，避免 Phase 1 把 API 缺口和算法复杂度混在一起。

| 步骤 | 内容 | 价值 |
|------|------|------|
| 0.5.1 | 在 Sphinx API / tutorial 中记录 `rules`、`NeuronFactory`、`ThresholdOptimizer` 的扩展方式 | ✅ 已完成：API 页新增 rules/factories/threshold 入口，教程新增扩展点说明 |
| 0.5.2 | 写一个最小自定义 rule 示例（例如 Identity/Marker rule 或 LeakyReLU skeleton，不宣称论文算法） | ✅ 已完成：中英文教程新增 `IdentityRule` 示例 |
| 0.5.3 | 明确标注当前不支持 per-channel threshold | ✅ 已完成：教程明确内置 `ReLURule` 路径只承诺标量 threshold |
| 0.5.4 | 为 `ActivationRule` protocol 本身增加结构性测试或示例测试 | ✅ 已完成：新增 duck-typed `ActivationRule` 端到端测试 |

### 已完成内容

- `docs/source/APIs/spikingjelly.activation_based.ann2snn.rst`：补充 `rules`、`factories`、`threshold` 三个 extension point 的 Sphinx API 入口。
- `docs/source/tutorials/{cn,en}/ann2snn.rst`：补充自定义转换规则小节、`IdentityRule` 示例、标量 threshold 边界说明。
- `test/activation_based/test_ann2snn.py`：新增不继承 `ReLURule` 的 duck-typed rule 测试，验证 `match`、`insert_hooks`、`find_replacements`、`replace_with_neurons` 完整协议链路。
- 文档构建降噪：修复本次触及页面与 legacy ann2snn 相关的 orphan/toctree、图片路径和未引用脚注 warning，降低增量文档构建噪声。
- `.gitignore`：将本地 agent 状态忽略范围从 `.agents/skills/` 扩大到 `.agents/`；计划文件通过显式 force-add 跟踪。

### 验收结果

- `.venv/bin/pytest test/activation_based/test_ann2snn.py -q`：59 passed。
- `cd docs && make SPHINXBUILD=../.venv/bin/sphinx-build html`：build succeeded。完整构建仍有仓库既有的非 ann2snn docstring/RST warning；本阶段新增或触及的 ann2snn API/tutorial 页面无新增 Sphinx warning。
- minimax review：无进一步意见。

---

## Phase 1：Transformer spike-equivalent operator foundation

**代表论文**：
- SpikeZIP-TF: "Conversion is All You Need for Transformer-based SNN", ICML 2024
- SpikedAttention: "Training-Free and Fully Spike-Driven Transformer-to-SNN Conversion with Winner-Oriented Spike Shift for Softmax Operation", NeurIPS 2024
- STA: "Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers", ICLR 2024

**选择理由**：Transformer/LLM 的主要障碍不是 ReLU replacement，而是 Softmax、LayerNorm、GELU、Self-Attention 等 SNN-unfriendly operators。实现这些通用 operator primitives，收益比优先实现某个 CNN/MLP 向算法更大。

> [!warning] 可行性备注
> 这一步不能直接做 ViT/BERT/LLM 全模型转换。先实现和测试 spike-equivalent operators，再把它们接入一个极小 Transformer block。目标是建立可复用的 operator substrate。

### 子阶段

| 步骤 | 内容 | 新增/修改文件 |
|------|------|-------------|
| 1A | 论文复核与最小设计：对比 SpikeZIP-TF / SpikedAttention / STA 的 operator 公式、输入输出语义和限制 | `.agents/plans/` 或 `ObsidianVault` |
| 1B | Spike-Softmax primitive：先张量级测试，再接入 FX toy block | 新 `operators.py` 或 `modules.py` |
| 1C | Spike-LayerNorm primitive：确认是否需要累积态、除法/平方根如何处理 | 新 `operators.py` 或 `modules.py` |
| 1D | Spike-GELU primitive：先支持 GELU/MLP toy block，不承诺所有激活 | 新 `operators.py` 或 `modules.py` |
| 1E | Spike-equivalent attention POC：先做 single-head toy attention，不接大模型 | 新 `operators.py`, `rules.py`, tests |
| 1F | Transformer block smoke test：极小 ViT/BERT-style block 输出 shape/误差 sanity | `test_ann2snn_transformer.py` 更合适 |

### 核心思想

**Spike-equivalent non-linear operator**：
- 对输入 spike 在时间维累积，得到对应 ANN operator 的累积输入。
- 对累积输入执行 Softmax / LayerNorm / GELU 等函数。
- 用当前累积输出与上一时间步累积输出的差分，得到该时间步 spike-equivalent 输出。

> [!note]
> Spike-equivalent operator 可能仍包含浮点函数、除法或指数运算。是否“fully spike-driven”要按算法分层标注，不能把所有 Transformer conversion 都宣传成纯加法/事件驱动。

---

## Phase 2：Activation/outlier-aware threshold 与 channel-wise redistribution

**代表论文**：
- "Towards Training-Free and Accurate ANN-to-SNN Conversion via Activation-Aware Redistribution", AAAI 2026
- LAS: "Loss-less ANN-SNN Conversion for Fully Spike-Driven Large Language Models", AAAI 2026

Transformer/LLM 转换的核心难点之一是 activation outliers。相比普通 CNN 的 layer-wise scale，Transformer/LLM 往往需要 channel-wise threshold、offset、redistribution 或 outlier-aware neurons。

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| 2.1 | 定义 channel-wise threshold / offset 的 shape 与 broadcast 规则 | `VoltageScaler` / operator primitive 单元测试 |
| 2.2 | Activation-aware redistribution POC：先做纯函数和小 MLP/attention block | 张量级误差测试 |
| 2.3 | Outlier-aware threshold POC：为后续 OAT neuron 做准备 | 分布/离群值 toy test |
| 2.4 | 接入 Transformer toy block | 端到端误差 sanity test |

> [!note]
> 这一步会改变 Phase 0 里“只明确支持标量 threshold”的边界。必须先稳定 shape 语义，再允许任何算法使用多元素 threshold。

---

## Phase 3：LAS-style fully spike-driven LLM conversion

**代表论文**：LAS: "Loss-less ANN-SNN Conversion for Fully Spike-Driven Large Language Models", AAAI 2026

LAS 目前更符合“泛用、效果好、面向大模型”的目标：它直接面向 LLM/VLM，处理 activation outliers、GELU/Softmax/LayerNorm、Self-Attention 等核心组件，并追求 fully spike-driven conversion。

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| 3.1 | OAT neuron POC：处理 activation outliers | toy outlier distribution |
| 3.2 | HG neuron POC：模拟 LLM nonlinear functions | GELU/FFN toy block |
| 3.3 | Spike-equivalent self-attention | single-head / tiny multi-head attention |
| 3.4 | Tiny decoder-only block conversion | shape + perplexity smoke test |
| 3.5 | 小型公开 LLM/VLM benchmark | 只在依赖和算力允许时做 |

> [!warning]
> LAS 是长期旗舰目标，不是下一轮 PR。它需要 Phase 1 的 spike-equivalent operators 和 Phase 2 的 outlier/channel-wise 基础设施先成熟。

---

## Phase 4：策略 helper API

**新建 `strategies.py`（等 Transformer operator 和至少一个 LLM/ViT conversion 路径稳定后再做）**：

```python
class ConversionStrategy:
    @staticmethod
    def rate_coding(mode="Max", momentum=0.1):
        """标准 rate coding（现有默认）。高精度，高延迟 T=256+"""
        return ConversionConfig(rules=[ReLURule()], ...)

    @staticmethod
    def transformer_spike_equivalent(...):
        """Transformer spike-equivalent operators."""
        return ConversionConfig(rules=[...], operators=[...], ...)

    @staticmethod
    def las_llm(...):
        """LAS-style fully spike-driven LLM conversion."""
        return ConversionConfig(rules=[...], operators=[...], calibration=...)
```

用法：
```python
# 旧 API — 不变
converter = Converter(dataloader=loader, mode="max")

# 策略快捷方式（建议优先用显式 helper，不急着改 Converter.__init__ 签名）
strategy = ConversionStrategy.transformer_spike_equivalent()
converter = Converter(dataloader=loader, **strategy.to_converter_kwargs())

# 完全自定义
converter = Converter(
    dataloader=loader,
    rules=[...],
    neuron_factory=...,
)
```

可行性备注：
- 不建议现在给 `Converter.__init__` 加 `strategy="..."` 字符串参数；这会把实验算法纳入核心 API 承诺。
- 先用 helper/dataclass 返回 `rules`、`neuron_factory`、`threshold_optimizer` 和可选 calibration hooks。
- 等策略稳定、文档和测试覆盖足够后，再考虑是否把 `strategy` 提升为公共参数。

---

## Phase 5+：其他研究 track

这些方向有价值，但不再作为主线。每个都应单独立项，先做可行性 spike，再决定是否进入产品化实现。

### 5.1 IJCAI 2025：Negative Spikes

**论文**：Xu Wang, Dongchen Zhu, Jiamao Li, "A Fast and Accurate ANN-SNN Conversion Algorithm with Negative Spikes", IJCAI 2025

| 组件 | 做法 |
|------|------|
| 激活 | LeakyReLU |
| 神经元 | 支持负脉冲的新神经元 |
| 阈值 | activation-variance threshold optimization |
| 校准 | 全层联合校准 |

可行性备注：
- 适合作为 CNN/MLP 的后续增强，但不如 Transformer/LLM operator conversion 泛用。
- 若做，仍应按“primitive → toy conversion → threshold → calibration”拆开。

### 5.2 ICML 2025：Differential Coding

**论文**：Huang et al., "Differential Coding for Training-Free ANN-to-SNN Conversion"

| 组件 | 做法 |
|------|------|
| 编码方式 | 差分编码替代 rate coding；具体 recurrence 需按论文复核 |
| 神经元 | 多阈值 IF 神经元（±阈值，`MultiThresholdIFNode`） |
| 优势 | T 降到 4-16 |

可行性备注：
- 低延迟价值明确，但代表性不如 Transformer/LLM conversion。
- 依赖多阈值/编码基础设施，因此不适合作为主线早期阶段。

### 5.3 ICML 2024：SignGD Neuronal Dynamics

**论文**：Oh & Lee, "Sign Gradient Descent-based Neuronal Dynamics: ANN-to-SNN Conversion Beyond ReLU Network"

| 组件 | 做法 |
|------|------|
| 神经元 | signGD 神经元动力学，近似多种非线性激活 |
| 优势 | 转换 ConvNeXt、MLP-Mixer、ResMLP |
| 新增文件 | `SignGDNeuronFactory` |

可行性备注：
- 这不是简单的 `NeuronFactory` 替换。ConvNeXt/MLP-Mixer 会涉及 LayerNorm、GELU、残差块和 FX tracing 边界。
- 建议先把 signGD neuron dynamics 写成独立模块和单元测试，不承诺整网转换。
- 若要支持 ConvNeXt，可能需要自定义 tracer 或对 function/module pattern 做更系统的匹配层。

### 5.4 ICML 2025：Parallel Spiking Calculation

**论文**：Hao et al., "Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation"

| 组件 | 做法 |
|------|------|
| 计算方式 | 时间步与累积发放率的数学映射，每个时间步独立可并行 |
| 优势 | 超低延迟，分布感知误差校准 |
| 新增文件 | `ParallelConversionStrategy` |

可行性备注：
- 这类算法更像新的 execution strategy，而不是 activation replacement rule。
- 需要先定义与 SpikingJelly 多步模式、`functional.reset_net`、batch/time 维约定的关系。
- 放在至少一个新 neuron/encoding 算法落地之后更稳妥。

---

## 实施依赖关系

```
Phase 0 (已完成)
    │
    ├── Phase 0.5 (文档/示例/扩展点稳定化)
    │
    ├── Phase 1 (Transformer spike-equivalent operator foundation)
    │       ├── Spike-Softmax
    │       ├── Spike-LayerNorm
    │       ├── Spike-GELU
    │       └── Spike-equivalent attention POC
    │
    ├── Phase 2 (Activation/outlier-aware threshold 与 redistribution)
    │       └── 依赖：channel-wise threshold / offset / outlier 语义明确
    │
    ├── Phase 3 (LAS-style fully spike-driven LLM conversion)
    │       └── 依赖：Phase 2 完成
    │
    ├── Phase 4 (策略 helper API)
    │       └── 依赖：Transformer/LLM conversion 路径稳定
    │
    └── Phase 5+ (其他研究 track)
            ├── Negative Spikes
            ├── Differential Coding
            ├── SignGD / ConvNeXt / MLP-Mixer
            ├── Parallel Spiking Calculation
```

## 关键参考文献

| 论文 | 会议 | 核心贡献 |
|------|------|----------|
| Rueckauer et al., *Conversion of Continuous-Valued Deep Networks* | Frontiers 2017 | 理论基础：IF 发放率 ≈ ReLU，RobustNorm |
| Diehl et al., *Fast Classifying, High-Accuracy Spiking Deep Networks* | IJCNN 2015 | MaxNorm |
| You et al., *SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN* ([arXiv](https://arxiv.org/abs/2406.03470)) | ICML 2024 | Spike-equivalent Transformer operators |
| Hwang et al., *SpikedAttention* ([OpenReview](https://openreview.net/forum?id=fs28jccJj5)) | NeurIPS 2024 | Fully spike-driven attention / winner-oriented spike shift |
| Jiang et al., *Spatio-Temporal Approximation* ([OpenReview](https://openreview.net/forum?id=XrunSYwoLr)) | ICLR 2024 | Training-free Transformer conversion, temporal/spatial approximation |
| Cao et al., *Activation-Aware Redistribution* ([AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/37148/41110)) | AAAI 2026 | Channel-wise threshold/offset, Transformer/BERT conversion |
| Chen et al., *LAS: Loss-less ANN-SNN Conversion for Fully Spike-Driven Large Language Models* ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37151)) | AAAI 2026 | OAT/HG neurons, spike-equivalent LLM components |
| Wang et al., *A Fast and Accurate ANN-SNN Conversion with Negative Spikes* | IJCAI 2025 | 负脉冲、LeakyReLU、阈值优化、多层校准 |
| Huang et al., *Differential Coding for ANN-to-SNN Conversion* | ICML 2025 | 差分编码、多阈值神经元、T=4-16 |
| Oh & Lee, *SignGD-based Neuronal Dynamics* | ICML 2024 | 非 ReLU 激活、ConvNeXt/MLP-Mixer |
| Hao et al., *Faster and Stronger: Parallel Spiking Calculation* | ICML 2025 | 并行脉冲计算、超低延迟 |
