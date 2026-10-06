神经元算子代码组织
===========================

神经元用户选择设备，而不是 kernel 库。CPU 执行使用 Torch；CUDA 执行通过 PyTorch
注册算子并自动选择兼容实现。当前 CUDA 候选包括原生 CUDA、Triton、CuPy 和 Torch
回退实现。

实现代码集中在仓库根目录 ``ops/``，并按神经元家族拆分子包。安装后它们位于同一
Python 发行包中的 ``spikingjelly._ops``。用户应调用神经元类或公开的
``functional.*_step``、``functional.*_multi_step`` 函数；provider 模块属于内部实现。

自动选择、诊断和可选 provider 控制见 :doc:`triton_backend`。

固定内核与自定义神经元
----------------------------------------

固定神经元的原生 CUDA 和 CuPy 实现使用各神经元子包中的显式 ``kernels.cuh``；
CuPy 只负责 JIT 编译和调用这些源码，不再通过 Auto CUDA 生成神经元代码。
融合 IF/LIF-Linear 也使用显式 CUDA 源码，反向使用与前向相同的充电与重置公式
重新计算脉冲。

Auto CUDA 转译器及 ``surrogate.cuda_codes()`` 已删除。自定义替代梯度只需提供
PyTorch 前向与梯度行为；自定义多步神经元请使用 :doc:`flexsn`。FlexSN 生成
Triton 内核，不提供旧转译器的 CUDA 源码输出或独立单步 kernel 生成功能。

旧生成框架的全局线程数、编译选项、编译器选择以及神经元 bool 脉冲保存配置也已移除。
显式算子管理各自的编译与启动参数；二值 Linear/卷积仍可使用
``configure.save_bool_spike_level`` 配置反向输入的压缩方式。
