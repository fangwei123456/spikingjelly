spikingjelly.activation_based.learning package
==========================================================================================

Learning method organization / 学习方法组织
--------------------------------------------------------------------------------

``learning.plasticity`` owns local update records, traces and learner lifecycles.
The existing ``learning.STDPLearner``, reward learners and layer-object helpers
remain direct exports of the same implementations. Tensor-only rules stay in
``functional.learning``. Gradient-based FPTT and OTTT/SLTT training pipelines
live in ``learning.online_learning``; their neurons and layers remain in
``neuron.online_learning`` and ``layer.online_learning``.

``learning.plasticity`` 管理局部更新记录、trace 和学习器生命周期。
已有 ``learning.STDPLearner``、奖励学习器及接收层对象的 helper 继续直接导出同一实现；
纯张量规则保留在 ``functional.learning``。FPTT、OTTT/SLTT 梯度训练流程位于
``learning.online_learning``，对应神经元和层仍位于 ``neuron.online_learning``、
``layer.online_learning``。两类方法分别管理自己的更新流程。

Migration / 迁移： replace ``functional.ottt_online_training(...)`` and
``functional.fptt_online_training(...)`` with calls from
``from spikingjelly.activation_based.learning import online_learning``.
Move direct imports from ``functional.online_learning`` to
``learning.online_learning``. The old module and functional exports are removed;
there are no forwarding modules. ``fptt_online_training_init_w_ra`` moves likewise.

将原来的 ``functional.ottt_online_training(...)``、``functional.fptt_online_training(...)``
改为 ``from spikingjelly.activation_based.learning import online_learning`` 后调用该模块；
直接导入路径从 ``functional.online_learning`` 改为 ``learning.online_learning``。
旧模块和 functional 导出已移除，不保留转发；``fptt_online_training_init_w_ra`` 同样迁移。

Local synaptic plasticity / 局部突触可塑性
--------------------------------------------------------------------------------

The learners compute synaptic weight updates from pre/post activity. The caller
chooses the encoding, competition, homeostasis and readout. Biases are unchanged.

学习器根据 pre/post 活动计算突触权重更新。编码、竞争、稳态控制和读出由调用者提供，
bias 保持不变。

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
   :imported-members:
   :exclude-members: online_learning, plasticity
   :undoc-members:
   :show-inheritance:

Online learning pipelines
+++++++++++++++++++++++++++++++

基于梯度的在线训练函数，执行前向、反传和优化器更新。

----

Gradient-based online training functions that run forward, backward and optimizer updates.

.. list-table::

   * - :func:`fptt_online_training_init_w_ra <spikingjelly.activation_based.learning.online_learning.fptt_online_training_init_w_ra>`
     - Initialize weight for FPTT.
   * - :func:`fptt_online_training <spikingjelly.activation_based.learning.online_learning.fptt_online_training>`
     - Online training with FPTT.
   * - :func:`ottt_online_training <spikingjelly.activation_based.learning.online_learning.ottt_online_training>`
     - Online training with OTTT or SLTT.

.. toctree::
   :hidden:

   online_learning <spikingjelly.activation_based.learning.online_learning>
