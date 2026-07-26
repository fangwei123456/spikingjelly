ANN-to-SNN Functional Helpers
+++++++++++++++++++++++++++++

这些函数实现 STA 和 SpikeZIP 中具有独立状态转移语义的叶子运算。TD operator
本身是围绕可替换 ``ann_forward`` 的状态适配器，不为其 ANN 数值路径提供冗余的
函数式包装。

----

These functions implement leaf operations with independent state-transition
semantics in STA and SpikeZIP. TD operators are state adapters around a
replaceable ``ann_forward`` and therefore do not expose redundant functional
wrappers for their ANN numeric paths.

.. automodule:: spikingjelly.activation_based.functional.ann2snn
   :members:
   :undoc-members:
   :show-inheritance:
