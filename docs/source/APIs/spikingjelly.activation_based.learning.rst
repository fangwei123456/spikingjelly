spikingjelly.activation_based.learning module
==========================================================================================

Local synaptic plasticity / 局部突触可塑性
--------------------------------------------------------------------------------

These are weight-update tools, not general-purpose deep-network trainers.
The caller supplies actual pre/post activity, encoding, competition, homeostasis
and a task readout. Only synaptic weights are updated, not biases.

这些工具实现局部权重更新，不是通用深层网络训练器。调用者提供实际 pre/post
活动、编码、竞争、稳态控制和任务读出。只更新突触权重，不更新 bias。

.. list-table:: Supported scope / 支持范围
   :header-rows: 1

   * - Rule / 规则
     - Synapse and mode / 突触及模式
     - Activity shape / 活动形状
   * - STDP
     - Linear, Conv1d, Conv2d; ``s`` / ``m``
     - Batch dimension; ``m`` adds leading time dimension
   * - MSTDP
     - Linear; ``s``
     - ``[N, I]`` / ``[N, O]``
   * - MSTDPET
     - Linear; ``s``
     - ``[I]`` / ``[O]``

Convolutions require ``groups=1`` and ``dilation=1``. STDP uses Euler trace
decay ``1-1/tau`` with finite ``tau>=1``. Reward rules use exponential decay
``exp(-1/tau)`` with finite ``tau>0``. All time constants are in simulated steps.
Current spikes enter traces before computing pair updates; simultaneous spikes
only cancel for suitable modulation functions. STDP sums over batch, spatial
positions and time; it does not average these dimensions.

卷积要求 ``groups=1``、``dilation=1``。STDP 使用 Euler 衰减 ``1-1/tau``，
要求有限 ``tau>=1``；奖励规则使用指数衰减 ``exp(-1/tau)``，要求有限 ``tau>0``。
时间常数单位均为模拟时间步。计算更新前先加入当前脉冲；同时发放是否抵消取决于
调制函数。STDP 对 batch、空间位置和时间求和，不取平均。

Reward timing / 奖励时间对齐
--------------------------------------------------------------------------------

Each reward learner consumes exactly one forward activity pair. At call ``t``:

每个奖励学习器每次消费恰好一对前向活动。第 ``t`` 次调用：

.. math::

   \Delta W_t^{MSTDP} &= \sum_n r_{t,n}\,\xi_{t-1,n} \\
   e_t^{MSTDPET} &= \exp(-1/\tau_e)e_{t-1}+\xi_{t-1}/\tau_e \\
   \Delta W_t^{MSTDPET} &= r_t e_t

The learner then computes ``xi[t]`` from current activity. Initial eligibility
is zero, so the first update is zero. A reward after the last activity requires
another forward step, including a quiet step where appropriate. Calling
``step()`` without activity returns ``None`` and does not advance time.

随后学习器根据当前活动计算 ``xi[t]``。初始 eligibility 为零，因此第一步更新为零。
最终活动之后的奖励需要后续前向推进，可按任务使用静默输入。
无活动时调用 ``step()`` 返回 ``None``，不会推进时间。

State and migration / 状态与迁移
--------------------------------------------------------------------------------

* ``step(on_grad=True)`` adds the negative scaled update to ``weight.grad`` and
  returns ``None``. ``on_grad=False`` returns the scaled update. Neither updates
  weights directly. This optimizer channel is not a task-loss gradient.
* ``step`` consumes records and keeps learning states; ``reset`` clears states
  and records without changing weights, gradients or recording enablement.
* ``disable`` stops new records; it does not clear existing records. Evaluation
  requires explicit disablement; ``net.eval()`` alone does not disable learners.
* ``close`` permanently removes hooks and clears learning states/records. A
  closed learner cannot be enabled or stepped again.
* Remove ``batch_size`` from ``MSTDPLearner`` construction. Batch size comes from
  actual activity; reset before changing it. Call reward ``step`` immediately
  after every forward instead of accumulating multiple activity pairs.

* ``step(on_grad=True)`` 向 ``weight.grad`` 累加负的缩放更新量，返回 ``None``；
  ``on_grad=False`` 返回缩放更新量。两者均不直接改权重，这个优化器通道不是任务梯度。
* ``step`` 消费记录、保留学习状态；``reset`` 清状态和记录，不修改权重、梯度或启停状态。
* ``disable`` 停止新增记录，不清已有记录。评估时需要显式停用；``net.eval()`` 不会停用学习器。
* ``close`` 永久移除钩子并清空状态和记录；关闭后不能重新启用或推进。
* 创建 ``MSTDPLearner`` 时删除 ``batch_size`` 参数，batch 从实际活动确定。
  更换 batch 大小前 reset；每次前向后立即调用奖励 step，不积压多对活动。

For runnable examples and the MNIST protocol, see :doc:`../tutorials/en/stdp`
and :doc:`../tutorials/cn/stdp`.

.. automodule:: spikingjelly.activation_based.learning
   :members:
   :undoc-members:
   :show-inheritance:
