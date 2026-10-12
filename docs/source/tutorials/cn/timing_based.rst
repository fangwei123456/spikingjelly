连续事件神经元
==============

``timing_based`` 使用显式事件时刻作为接口。TTFS 是编码约束，不意味着实现一定
使用时间戳；按时间步传递 01 激活的 TTFS 网络仍属于 ``activation_based``。
本页的 Tempotron 在输入事件区间上求解，不构造均匀时间网格。

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

* ``v_max``：实际 shunting 轨迹的最大电压。
* ``t_peak``：最早达到最大电压的时刻。
* ``t_spike``：首次上穿时刻，未发放为 ``inf``。

判断是否发放使用 ``torch.isfinite(t_spike)``。相切时即使 ``v_max`` 达到阈值也不
发放。空样本返回 ``(0,0,inf)``；调用间无状态持久化，不需要 ``reset()``。

紧凑事件输入
------------

输入包含有限时刻 ``spike_times[E]``、int64 输入列索引 ``input_indices[E]`` 和
int64 CSR 边界 ``batch_offsets[B+1]``。边界从 0 开始、非递减、以 E 结束。
样本内事件可以无序；同刻事件聚合；重复边界表示空样本。窗外事件忽略且梯度为零。
所有张量必须与权重同设备；时刻和权重支持同 dtype 的 FP32/FP64 CPU/CUDA。

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

无状态计算入口为 :func:`spikingjelly.timing_based.functional.tempotron`；模块只拥有
``weight[O,I]`` 和超参数。权重以 ``N(0,0.001²)`` 初始化，checkpoint 键为
``weight``。归一化不随阈值变化。

连续高斯编码
------------

:class:`spikingjelly.timing_based.encoding.GaussianTuning` 是普通 ``nn.Module``，
感受野中心 ``mu`` 与方差 ``sigma2`` 是固定 buffers。输入为
``[B,n,samples_count]``；输出直接是上述事件三张量。

.. code-block:: python

   from spikingjelly.timing_based.encoding import GaussianTuning

   encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1)).double()
   x = torch.tensor([[[0.2, 0.7]]], dtype=torch.float64, requires_grad=True)
   events = encoder(x, duration=5.0)
   net = Tempotron(1 * 2 * 4, 2, duration=5.0).double()
   logits, _, _ = net(*events)
   torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()

编码时刻为 ``duration * (1-response)``，不取整。输入列按特征、样本、感受野
展平；仅输出 ``0 <= time < duration`` 的事件，没有 ``-1`` 或 ``inf`` 填充。
有效事件时刻可对输入值求导，事件选择边界不保证唯一导数。

Autograd 与 loss
----------------

内部仅支持权重及输入时刻的一阶反向。首次发放时刻使用阈值方程的隐式导数，
不是对二分分支求导；峰值与峰值时刻使用局部解析导数。共享连接及事件的
贡献累加。并列峰值选择最早分支；事件边界时间使用稳定排序中首个原事件的
分支。固定窗口端点的峰值时间梯度为零，未发放的 ``t_spike`` 分支梯度为零。

非光滑点不保证唯一导数。只有 loss 使用相应时间输出时才检查其奇异导数；
不隐藏裁剪、替代梯度或沉默恢复。二阶反向明确不支持。

Loss 由调用方选择；上例展示峰值平方误差及交叉熵的一阶反向。MNIST 示例
使用尺度校准交叉熵加发放错误线性罚项，避免原始峰值 CE 与 shunting 的训练
目标不匹配。含 ``inf``
的发放时刻必须由任务自己的 loss 正确处理。全负响应或无输入的数学峰值可能
是静息平台，其梯度可以为零；初始化、正则化或恢复策略由任务决定。

MNIST 示例与性能范围
--------------------

.. code-block:: bash

   uv run python -m spikingjelly.timing_based.examples.tempotron_mnist \
       --device cpu --duration 100 --batch-size 64 --dataset-dir ./data

示例将 ``5*v_max/v_threshold`` 作为 logits，并加入只针对错误发放状态的
线性阈值偏差（按阈值归一化）。正确类别应发放，其他类别应沉默；错误提前
发放受到抑制，使后续证据重新可用。预测仍用 ``v_max.argmax``。
loss 是任务层选择，未改变神经元的阈值、shunting 或内部 autograd；这不是
直接对含 ``inf`` 的 ``t_spike`` 使用交叉熵。

求解器使用 eager 执行与内部 FP64：按有界事件块批量求状态、驻点及首次上穿，
整批样本并行推进，同刻事件一次聚合；局部时差矩阵替代逐事件小算子。
CPU 块宽为 128，CUDA 为 512 个事件组，不构造均匀时间网格。Torch 参考的临时矩阵内存
随 batch 大小及块宽平方增长；更大的块用内存换取更少的调度。反向按整批事件
重算被使用输出所需的解析导数，保存张量规模保持不变。
安装 Triton 时，CUDA 自动使用融合 FP64 核计算仿射前缀扫描块状态、阈值求根
及仅对峰值电压求导的权重反向。扫描按时序合成连续电压/电流的精确状态转移，
块内存随事件数线性增长，避免二次方响应矩阵；只改变浮点归约顺序，不引入时间
采样。其他解析导数使用 PyTorch。未安装 Triton 时 CUDA
使用相同的 Torch 参考。仍有块级同步，并非整模型融合或 CUDA Graph 执行。
旧实现的历史正确率与速度不代表新动力学的结果。

示例默认最多训练 40 轮；完整复现需显式指定 ``--epoch 100``。两次已完成的
单随机种子 100 轮实验在 20–40 轮进入测试平台，此后继续拟合训练数据。较短
上限是实验预算，不保证所有任务在 40 轮收敛；不根据测试准确率早停或选择模型。

本次接口替换了整数 T、ret_type、mse_loss、旧 checkpoint 包装和
GaussianTuning.encode；当前用法以上述接口为准。

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
这是不同于双指数 Tempotron 的 SRM 核族，不是其等价加速。
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
分数贡献 ``2-time``，logits 为累积贡献的 5 倍，loss 是交叉熵。
Adam（默认 lr=0.01）训练原始参数，示例通过 PyTorch parametrization 使用
``gain*softmax(weight)``，两层行增益固定为 8、2，以保持发放并控制事件数量。
约束和读出均在示例，神经元支持负权重，不含 loss、替代梯度或恢复规则。

.. code-block:: bash

   uv run python -m spikingjelly.timing_based.examples.srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./srm-mnist \
       --batch-size 128 --epochs 10 --seed 0

使用完整 MNIST：55000/5000 训练/验证划分，验证集选择 checkpoint 和早停
（默认 patience=3），最终仅评估完整 10000 测试样本。保存配置、源码指纹、
逐轮指标、``best.pt`` 和包含测试正确数的 ``result.json``。CPU 可将设备改为
``cpu``；CUDA 使用相同 PyTorch 数学，无专用 SRM kernel，不声称普遍提速。
