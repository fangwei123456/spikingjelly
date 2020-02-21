学习规则
=======================================

本节教程主要关注SpikingFlow.learning，包括如何使用已有学习规则、如何定义新的学习规则

学习规则是什么
-------------
“学习”在ANN中或许更多地被称作是“训练”。ANN中基于梯度的反向传播优化算法，就是应用最为广泛的学习规则

在SNN中，发放脉冲这一过程通常使用阶跃函数去描述，这是一个不可微分的过程；SNN比较注重生物可解释性，生物神经系统中似乎并没有使\
用反向传播这种训练成千上万次才能达到较好结果的“低效率”方法。在SNN中如何使用反向传播算法也是一个研究热点，使用反向传播算法的\
SNN一般为事件驱动模型（例如SpikeProp和Tempotron，在SpikingFlow.event_driven中可以找到），而SpikingFlow.learning中更多的聚\
焦于生物可解释性的学习算法，例如STDP

STDP(Spike Timing Dependent Plasticity)
----

STDP(Spike Timing Dependent Plasticity)学习规则是在生物实验中发现的一种突触可塑性机制。实验发现，突触的连接强度受到突触连接\
的前（pre）后（post）神经元脉冲活动的影响

如果pre神经元先发放脉冲，post神经元后发放脉冲，则突触强度增大；反之，如果post神经元先发放脉冲，pre神经元后发放脉冲，则突触强度\
减小。生物实验数据如下图所示，横轴是pre神经元和post神经元释放的一对脉冲的时间差，也就是 :math:`t_{pre} - t_{post}`，纵轴表示\
突触强度变化的百分比：

.. image:: ./_static/tutorials/3.png

这种突触强度和前后脉冲发放时间的关系，可以用以下公式进行拟合：

.. math::
    \begin{align}
    \Delta w=
    \left\{ \begin{aligned}
    & A e^{-\frac{-(t_{pre} - t_{post})}{\tau}}, t_{pre} - t_{post} \leq 0, A > 0\\
    & B e^{-\frac{-(t_{pre} - t_{post})}{\tau}}, t_{pre} - t_{post} \geq 0, B < 0
    \end{aligned} \right.
    \end{align}

一般认为，突触连接权重的改变，是在脉冲发放的瞬间完成。不过，上图中的公式并不适合代码去实现，因为它需要分别记录前后神经元的脉冲\
发放时间。使用[Ref1]_提供的基于双脉冲的迹的方式来实现STDP更为优雅





.. [Ref1] Morrison A, Diesmann M, Gerstner W. Phenomenological models of synaptic plasticity based on spike\
timing[J]. Biological cybernetics, 2008, 98(6): 459-478.