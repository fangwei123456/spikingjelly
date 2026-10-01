spikingjelly.nsys module
========================

.. automodule:: spikingjelly.nsys
   :members:
   :undoc-members:

单机多卡采集 / Single-host multi-GPU capture
--------------------------------------------------------------

采集器位于 launcher 外层，一次 session 生成一份包含所有子进程与 GPU 的报告。
分布式训练接口不需要 NSYS 配置。先启动负载，确认热身完成，再执行 shell 输出的
``nsys start/stop/shutdown`` 命令。

Put the collector outside the launcher: one session produces one report for
the entire process tree. No NSYS configuration is added to distributed
training interfaces. Start collection after the workload has warmed up,
using the native start/stop/shutdown commands printed by the shell:

.. code-block:: bash

   bash benchmark/nsys_snn.sh capture \
     --control=manual --session=sj-ddp \
     --trace=cuda,nvtx,osrt,python-gil,nccl \
     output/ddp -- \
     torchrun --standalone --nnodes=1 --nproc-per-node=2 workload.py

短实验可使用 ``--control=api``。各 rank 调用 :func:`capture` 控制其设备；
单进程多卡传入 ``devices=[0, 1]``。调用者在窗口边界协调所有参与者，并在停止前
等待 GPU 完成；接口自身不添加 barrier 或每步同步。

Short experiments can use ``--control=api``. Each rank calls :func:`capture`
for its device; one process using multiple GPUs passes ``devices=[0, 1]``.
The caller coordinates participants at window boundaries and drains GPU work
before stopping. These helpers add no barriers or per-step synchronization.

.. code-block:: python

   from spikingjelly import nsys

   with nsys.step(0, "training", True, rank=rank, world_size=world_size):
       with nsys.region("forward", True, stage=stage, microbatch=microbatch):
           output = model(inputs)

没有标记时仍输出进程、GPU、CUDA API 与通信总览；rank、step、stage 和 microbatch
不从 kernel 名称推断。只有各 rank 的 phase、index 和 world_size 一致时才合并逻辑
step。相同 index 在不同线程/进程中不是同一个本地范围。

Unmarked programs still produce process, GPU, CUDA API and communication
overviews. Rank, step, stage and microbatch identities are never inferred from
kernel names. Logical steps require matching explicit phase, index and
world_size across ranks. Local scopes remain distinct across processes/threads.

分析与迁移 / Analysis and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash benchmark/nsys_snn.sh analyze output/ddp.nsys-rep output/ddp-analysis
   python benchmark/analyze_nsys_snn.py analyze output/ddp-analysis/trace.sqlite \
     --output-dir output/rank1-step8 --step-index 8 --rank 1
   python benchmark/analyze_nsys_snn.py analyze output/ddp-analysis/trace.sqlite \
     --output-dir output/window --time-range-ms 10 30

``--pid``、``--rank``、``--device`` 和时间窗口只筛选图表视图；完整报告统计仍保留。
设备筛选使用报告中的 GPU UUID 或 device key。schema v2 将事件存储一次，并用
step ID 与 CUDA API ID 关联；旧 v1 summary 需从原始 SQLite 重新分析。
CUDA Graph stage 需要 capture-graph 的构建期 NVTX；没有投影证据时保持未归类。

``--pid``, ``--rank``, ``--device`` and time windows filter timeline views;
full-capture statistics remain available. Device filters use a GPU UUID or
device key from the report. Schema v2 stores each event once, linking by
step and CUDA API IDs. Re-analyze original SQLite exports to replace v1
summaries. Graph stage projection requires capture-graph build-time NVTX;
missing projection evidence stays unclassified.

各 GPU 独立计算 busy union 和计算/通信重叠；多卡时间求和不是 step 时延。
NCCL kernel 时间可能包含 GPU 等待，不能当作纯链路传输时间。无 profiler 测量才
用于性能结论。报告目录必须是新目录，避免混入旧图表。

Busy union and compute/communication overlap are calculated per GPU.
Summing GPU times does not yield step latency. NCCL kernel duration may include
GPU waiting and is not pure link-transfer time. Use unprofiled measurements for
performance conclusions. Use a fresh output directory to avoid stale plots.

``benchmark/nsys_multigpu_example.py`` 提供独立 DP/DDP/PP 训练/推理示例。
手动模式指定 ``--gate-dir``：所有 ``ready-N`` 文件出现后启动 NSYS 并创建
``go``，所有 ``done-N`` 出现后停止采集并创建 ``exit``，最后关闭 session。
此协议只用于可复现示例，不是分布式训练框架的一部分。

``benchmark/nsys_multigpu_example.py`` provides independent DP/DDP/PP
training/inference workloads. With manual control and ``--gate-dir``, wait
for every ``ready-N``, start NSYS and create ``go``; wait for every
``done-N``, stop collection and create ``exit``, then shut down the
session. This gating protocol belongs only to the reproducible example.

``complete`` 只表示逻辑 step 的预期 rank 标记齐全且已闭合，不证明训练成功。
结合 launcher 退出状态、各 rank 结果和预期 step 数检查整个实验；失败时仍可分析
部分报告。API 采集使用 ``--kill=none``，避免会话结束时终止仍在清理的 worker。

``complete`` means the expected rank ranges are present and closed, not that
training succeeded. Check launcher status, per-rank results and expected step
counts to validate a run; partial failed reports remain useful for diagnosis.
API capture uses ``--kill=none`` so session shutdown does not terminate workers
that are still cleaning up.

未闭合的 NVTX step 在图中标为 ``open``；其结束边界只用于展示采集到的区间，
CPU 耗时为 ``null``。缺少 rank 或存在未闭合范围的逻辑 step 不报告 CPU 总跨度或
起止偏斜。示例的 ``--validate`` 仅支持 small 模型的 DP/DDP/PP 模式。

Unclosed NVTX steps are marked ``open`` in the timeline. Their end bounds only
delimit the observed interval; CPU duration is ``null``. Logical steps with
missing ranks or unclosed ranges do not report CPU envelope or start/end skew.
The example's ``--validate`` supports only DP/DDP/PP with the small model.
