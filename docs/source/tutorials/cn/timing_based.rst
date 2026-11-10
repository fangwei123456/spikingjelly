基于时刻的连续事件 SNN
======================

你将把输入编码为脉冲时刻，并在 ``timing_based`` 神经元之间传递事件。
这些模型在连续时间上求解响应，不构造均匀时间网格。首次脉冲时刻编码（TTFS）
用第一个脉冲的时刻表示数值。按离散时间步传递二值激活的TTFS网络使用 ``activation_based``。

先将脉冲时刻打包为紧凑事件格式，再选择模型：

* Tempotron：双指数突触后电位（PSP）、单次输出发放、发放后屏蔽新输入和峰值输出。
* SRM：带复位的线性上升 PSP，允许多次输出发放。
* CubicSRM：有限支撑三次 PSP、单次输出发放及通用求根器。
* CubicConv2d：在二维感受野间共享 CubicSRM 权重。

所有模块在调用间无状态，参数是普通PyTorch参数；损失、初始化约束和恢复规则
属于具体任务。事件的创建、删除和次序切换是非光滑边界，文档中的梯度适用于
所选事件拓扑保持不变的局部分支。

神经元响应与输出
----------------

每条输入连接可以有多个脉冲，共享对应突触权重。静息电位为零，双指数 PSP
归一化为单位峰值：

.. math::

   V(t)=\sum_i w_i\sum_r h(t-t_{i,r}),\qquad
   h(u)=\frac{e^{-u/\tau}-e^{-u/\tau_s}}
               {e^{-u_p/\tau}-e^{-u_p/\tau_s}}\mathbf{1}_{u\ge0},

   u_p=\frac{\tau\tau_s\log(\tau/\tau_s)}{\tau-\tau_s}.

支持两个不同的正时间常数，包括顺序交换和接近相等的情况。内部使用稳定的
电压/电流状态、``expm1`` 和 ``log1p``，避免累计全局正指数。

观察窗为 ``[0,duration]``，各时刻与时间常数使用同一单位。首次上穿阈值时发放；
与阈值相切不算发放。发放后屏蔽同刻及更晚的输入，已有 PSP 继续演化，不执行
硬重置。因此电压峰值可能晚于首次发放。

:class:`spikingjelly.timing_based.neuron.Tempotron` 返回三个独立的 ``[B,O]`` 张量：

* ``v_max``：实际输入屏蔽轨迹的最大电压。
* ``t_peak``：最早达到最大电压的时刻。
* ``t_spike``：首次上穿时刻，未发放为 ``inf``。

判断是否发放使用 ``torch.isfinite(t_spike)``。相切时即使 ``v_max`` 达到阈值也不
发放。空样本返回 ``(0,0,inf)``；调用间无状态持久化，不需要 ``reset()``。

紧凑事件输入
------------

将B个样本的E个事件打包为三个张量：有限时刻 ``spike_times[E]``、int64输入列索引
``input_indices[E]`` 和int64边界 ``batch_offsets[B+1]``。这种压缩稀疏行（CSR）格式
用边界划分样本。边界从0开始、非递减、以E结束。
样本内事件可以无序；同刻事件聚合；重复边界表示空样本。窗外事件忽略且梯度为零。
所有张量必须与权重同设备；时刻和权重支持相同数据类型的FP32/FP64，运行于CPU或CUDA。

.. code-block:: python

   import torch
   from spikingjelly.timing_based.neuron import Tempotron

   neuron = Tempotron(2, 1, duration=5.0, tau=2.0, tau_s=0.5).double()
   with torch.no_grad():
       neuron.weight.copy_(torch.tensor([[1.0, 0.5]], dtype=torch.float64))
   times = torch.tensor([0.3, 0.4, 3.0], dtype=torch.float64,
                        requires_grad=True)
   indices = torch.tensor([0, 0, 1], dtype=torch.int64)
   offsets = torch.tensor([0, 3, 3], dtype=torch.int64)
   v_max, t_peak, t_spike = neuron(times, indices, offsets)
   assert v_max.shape == t_peak.shape == t_spike.shape == (2, 1)
   assert torch.isinf(t_spike[1]).all()
   target = torch.tensor([[1.5], [0.0]], dtype=torch.float64)
   loss = (v_max - target).square().mean()
   loss.backward()
   assert torch.isfinite(times.grad).all()

``offsets=[0,3,3]`` 表示两个样本：第一个有三个事件，第二个为空。
三个输出的形状均为 ``[2,1]``，空样本不发放，因此其 ``t_spike`` 为 ``inf``。

无状态计算入口为 :func:`spikingjelly.timing_based.functional.tempotron`；模块只拥有
``weight[O,I]`` 和超参数。权重以 ``N(0,0.001²)`` 初始化，检查点键为
``weight``。归一化不随阈值变化。

连续高斯编码
------------

:class:`spikingjelly.timing_based.encoding.GaussianTuning` 是普通 ``nn.Module``，
感受野中心 ``mu`` 与方差 ``sigma2`` 是固定缓冲区。输入为
``[B,n,samples_count]``；输出直接是上述事件三张量。

.. code-block:: python

   import torch
   from spikingjelly.timing_based.encoding import GaussianTuning
   from spikingjelly.timing_based.neuron import Tempotron

   encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1)).double()
   x = torch.tensor([[[0.2, 0.7]]], dtype=torch.float64, requires_grad=True)
   events = encoder(x, duration=5.0)
   net = Tempotron(1 * 2 * 4, 2, duration=5.0).double()
   logits, _, _ = net(*events)
   torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()

编码时刻为 ``duration * (1-response)``，不取整。输入列按特征、样本、感受野
展平；仅输出 ``0 <= time < duration`` 的事件，没有 ``-1`` 或 ``inf`` 填充。
有效事件时刻可对输入值求导，事件选择边界不保证唯一导数。

自动求导与损失
----------------

内部仅支持权重及输入时刻的一阶反向。首次发放时刻使用阈值方程的隐式导数，
峰值与峰值时刻使用局部解析导数。共享连接及事件的
贡献累加。并列峰值选择最早分支；事件边界时间使用稳定排序中首个原事件的
分支。固定窗口端点的峰值时间梯度为零，未发放的 ``t_spike`` 分支梯度为零。

非光滑点不保证唯一导数。只有损失使用相应时间输出时才检查其奇异导数；
不隐藏裁剪、替代梯度或沉默恢复。二阶反向明确不支持。

损失由调用方选择；上例展示峰值平方误差及交叉熵的一阶反向。MNIST 示例
使用尺度校准交叉熵加发放错误线性罚项，避免原始峰值交叉熵与输入屏蔽的训练
目标不匹配。含 ``inf``
的发放时刻必须由任务自己的损失正确处理。全负响应或无输入的数学峰值可能
是静息平台，其梯度可以为零；初始化、正则化或恢复策略由任务决定。

MNIST 示例与性能范围
--------------------

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.tempotron_mnist \
       --device cpu --duration 100 --batch-size 64 --dataset-dir ./data

示例使用 ``5*v_max/v_threshold`` 的交叉熵加错误发放状态的线性阈值罚项，以 ``v_max.argmax`` 分类。

Tempotron求解器内部使用FP64。CUDA算子默认依次选择可用的原生CUDA、Triton
或PyTorch实现。仅安装Triton不能保证选中它，还取决于设备兼容性和算子是否可用。

默认训练40轮，也可以用 ``--epoch 100`` 指定100轮。脚本每轮输出训练和测试正确率，
最后将模型保存为 ``--model-output-dir`` 下的 ``tempotron_snn_mnist.ckpt``。
脚本运行指定轮数，不早停，也不按测试正确率选择检查点。

参考
----

Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based
decisions. Nature Neuroscience 9, 420–428 (2006).
`原文 <https://www.nature.com/articles/nn1643>`_。

多发 SRM 与两层 MNIST
------------------------------------------

:class:`spikingjelly.timing_based.neuron.SRM` 是实验性固定核模型：

.. math::

   u_j(t)=\sum_i w_{ji}\sum_e (t-t_{ie})_+
          -\vartheta\sum_{f:s_{jf}<t}1.

输入响应是线性上升 PSP，发放后加入阶跃复位响应：电压归零，电流保留。
模型无泄漏和绝对不应期，允许多输入及多输出脉冲，继续接收发放后输入。
线性上升响应使它具有不同于双指数Tempotron的动力学。
未复位响应首次上穿 ``k*threshold`` 对应第 k 次发放。排序后通过前缀和、
端点累积最大值和解析根求解，内部 FP64，不展开时间网格；所选解析分支
通过 PyTorch 对权重和输入时刻求导。事件创建/删除和排序边界不保证唯一导数。

SRM 接受同样的 CSR 输入，输出也是 ``(times,indices,offsets)``，可直接串联。
输出按样本、神经元索引和发放次序排列；下一层内部排序。空样本无事件，
窗口端点上穿计入，同刻输出优先。每样本每输出的 ``max_spikes`` 默认 64，
容量不足明确报错，不截断事件。使用普通 ``nn.Module.to`` 和 ``state_dict``。

.. code-block:: python

   import torch
   from spikingjelly.timing_based.neuron import SRM

   first = SRM(2, 3, duration=2.0).double()
   second = SRM(3, 1, duration=2.0).double()
   with torch.no_grad():
       first.weight.fill_(1.0)
       second.weight.fill_(0.5)
   times = torch.tensor([0.1, 0.4], dtype=torch.float64, requires_grad=True)
   events = first(times, torch.tensor([0, 1]), torch.tensor([0, 2]))
   output_times, _, _ = second(*events)
   output_times.sum().backward()

MNIST 示例使用 784→128→10 两个可训练 SRM 层，像素时刻为 ``1-pixel``，
观察窗终点为 2。隐藏层可多发，事件直接进入第二层。每个输出事件对类别
分数贡献 ``2-time``，分类分数为累积贡献的 5 倍，损失是交叉熵。
Adam（默认学习率0.01）训练原始参数，示例通过 PyTorch 参数化使用
``gain*softmax(weight)``，两层行增益固定为 8、2，以保持发放并控制事件数量。
约束和读出均在示例，神经元支持负权重，不含损失、替代梯度或恢复规则。

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./srm-mnist \
       --batch-size 128 --epochs 10 --seed 0

使用完整 MNIST：55000/5000 训练/验证划分，验证集选择检查点和早停
（默认连续3轮验证正确率未提高时停止），最终仅评估完整 10000 测试样本。保存配置、源码指纹、
逐轮指标、``best.pt`` 和包含测试正确数的 ``result.json``。CPU 可将设备改为
``cpu``；CUDA使用相同的PyTorch计算，无专用SRM核。

完整分段与第三种 PSP 模型
--------------------------------

:func:`spikingjelly.timing_based.solver.first_spike_time` 求最早上穿时刻。调用者必须提供加权总响应的全部驻点，求解器不验证分段的完整性。

.. code-block:: python

   import torch
   from spikingjelly.timing_based.solver import first_spike_time

   def response(time, piece, arrival, weight):
       return weight * (time - arrival) - 1, weight.expand_as(time)

   arrival = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
   weight = torch.tensor([[2.0]], dtype=torch.float32, requires_grad=True)
   # 残差在整个区间内线性，因此此分段完整。
   root = first_spike_time(response, torch.tensor([[0., 2.]], dtype=torch.float32),
                           (arrival, weight))
   root.sum().backward()  # root=0.6，arrival.grad=1，weight.grad=-0.25

:class:`spikingjelly.timing_based.neuron.CubicSRM` 是第三个具体模型：

.. math::

   h(u)=\begin{cases}\frac{27}{4}z(1-z)^2,&0<z=u/L<1,\\0,&\text{otherwise},\end{cases}
   \qquad V_j(t)=\sum_e w_{j,i_e}h(t-t_e).

每输出至多发一次，时刻为最早上穿阈值的时刻，没有输出历史核或重置。
支持重复输入及负权重。在输入到达与支撑终点划分的区间内，总响应是三次
多项式；解析求一阶导数（二次多项式）的全部实根，得到完整单调分段，包括移位
输入造成的多个峰。局部电压/导数前缀推进避免全局事件时刻幂；聚合同刻
更新时仍保留各原事件的时刻梯度。共享求解器负责数值根及隐式梯度。
不使用均匀时间网格；CUDA 可选使用融合 Triton 核。Tempotron 保留双指数求解、
输入屏蔽与峰值输出；线性上升SRM 保留解析多水平扫描与阶跃复位。通用求解器
不替换这两条已优化路径。

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.cubic_srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./cubic-srm-mnist \
       --hidden-features 64 --batch-size 128 --epochs 10 --seed 0

示例为 784→hidden→10 两层 CubicSRM，像素时刻 ``1-pixel``、窗口 4、支撑 2、
阈值 1。可微的 ``gain*softmax(weight)`` 行约束（隐藏/输出增益 8/2）属于示例。类别越早发放得分
越高：分类分数为 ``50*(4-time)``，沉默得分零。使用交叉熵与 Adam，复用
线性上升SRM 示例的验证选检查点/早停流程，记录完整 55000/5000/10000
划分与源码哈希。默认隐藏宽度 128、批大小32，上述命令使用宽度 64、批大小128。
驻点超阈幅度小于求解器文档所述 FP32 舍入界时无法与相切区分。极端支撑
比例仍受浮点分辨率限制。没有替代梯度或沉默恢复。

通用求解器与 CubicSRM 只接受 FP32 浮点张量，无精度选择参数或 FP64 路径。
极端绝对时刻/支撑比例可能使区间无法解析；应重新设定时间原点或尺度。

连续事件空间卷积
-------------------

:class:`spikingjelly.timing_based.neuron.CubicConv2d` 在二维感受野间共享三次PSP核。
输入仍为CubicSRM的三个CSR事件张量，另传显式空间大小 ``(H,W)``。
输入索引展平 ``(Cin,H,W)``，输出展平 ``(Cout,Hout,Wout)``，仍按原B个样本分段。
填充位置无事件，保留同一位置的多个输入事件；支持方形核和两个轴共用的整数
步幅/填充，无偏置、分组、膨胀率、输出复位历史或跨调用状态。

将事件路由成局部试次、空间位置并入批次后，直接调用既有FP32三次核求解器
及可选Triton算术，所有位置/通道用张量并行。权重梯度跨位置累加，重叠感受野
的时刻梯度累加回原事件。保留CubicSRM最早上穿、相切、沉默及仅一阶梯度契约；
空间卷积不引入时间采样。

CSR 路由与 unfold
~~~~~~~~~~~~~~~~~~

公开卷积使用 CSR 事件，因为同一空间位置可以沉默或包含多个脉冲。路由从每个
事件出发，找出覆盖该位置的全部感受野，并记录对应的局部
``(channel, kernel_y, kernel_x)`` 权重列。输出通道共享这份路由结果。
每个感受野成为一个 CubicSRM 试次，普通自动求导将梯度累加到共享权重和被
多个感受野引用的输入事件。

对于每位置恰好一发的稠密时刻图，``torch.nn.functional.unfold`` 能以较低成本
提取固定大小的时刻块。该表示仍需用显式掩码标记沉默，为多发位置补齐
事件槽位，并单独处理填充，避免将零值解释成时刻 0 的事件。提取块后，
两种表示使用相同的 PSP、分段和求根器。公开实现接受 CSR 事件；MNIST 每像素
恰好一发，因此下文用 unfold 作为等价性能参考。

.. code-block:: python

    import torch
    from spikingjelly.timing_based.neuron import CubicConv2d

    layer = CubicConv2d(1, 2, 3, duration=4.0, stride=2, padding=1)
    with torch.no_grad():
        layer.weight.fill_(1.2)
    times = torch.tensor([0.1, 0.3])
    indices = torch.tensor([0, 0])  # 同一个输入像素的重复事件
    offsets = torch.tensor([0, 2])
    output_times, output_indices, output_offsets = layer(
        times, indices, offsets, (4, 4)
    )

最小MNIST网络为 ``1×28×28 → 8C5/stride2 → 8×12×12 → 1152→10``，
两层脉冲神经元、11720参数。像素时刻 ``1-pixel``，每像素都有事件，黑背景在时刻1发放。
示例按输出行使用softmax增益8/2，分类分数为 ``50*(4-time)``（沉默0），交叉熵+Adam。
共享训练流程使用55000/5000/10000划分、验证选检查点、最终一次测试；
``--hidden-features`` 表示卷积输出通道数。

.. code-block:: bash

    python -m spikingjelly.timing_based.examples.cubic_conv_mnist \
        --device cuda:0 --batch-size 128 --epochs 5 --hidden-features 8 \
        --dataset-dir ./data --output-dir ./cubic-conv-mnist

卷积降低局部输入连接数，同时增加输出神经元、事件路由与排序。临时系数随位置、
通道及局部事件数增长。路由方案应在相同网络、事件和权重上比较，并包含
反向传播和峰值内存。分块可以降低部分构造临时量，自动求导仍要保存每个块
反向所需的张量。以下测量比较该网络的两种事件路由方式。

参考结果（commit `7341418a <https://github.com/fangwei123456/spikingjelly/commit/7341418a>`_，Tesla V100S）：

使用Torch 2.7.1+cu118、Triton 3.3.1，随机种子0、批大小128，完整训练5轮后，
验证集选中第5轮：验证正确率92.24%，测试正确率93.06%（9306/10000）。这是单种子结果。

三个独立进程使用相同训练后网络/检查点/MNIST首批，比较交叉熵前向+反向，不含优化器。
各3次热身、10轮交替重复；耗时为进程中位数的中位数，峰值内存取三个进程最大值：

.. list-table:: 密集单发输入的等价路由对照
   :header-rows: 1

   * - Batch
     - 多事件CSR路由（ms）
     - 单发unfold参考（ms）
     - CSR峰值内存（MiB）
   * - 1
     - 40.90
     - 38.46
     - 10.10
   * - 8
     - 44.79
     - 42.83
     - 87.83
   * - 32
     - 53.37
     - 52.77
     - 361.71
   * - 128
     - 104.59
     - 104.00
     - 1583.22

unfold参考只处理每像素一发，复用相同三次核求解器和输出打包。
该密集负载下，两条路径的总延迟接近。B128独立路由前向+反向2.14ms（unfold1.28ms）；
它包含输入时刻梯度，而图像训练步不请求像素梯度，因此只可比较量级，不能直接从总时延相减。
现有分段与求根仍占主要成本。
