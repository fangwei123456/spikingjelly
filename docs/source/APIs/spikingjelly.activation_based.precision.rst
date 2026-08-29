spikingjelly.activation_based.precision package
================================================

``precision`` prepares ordinary PyTorch models before optimizer construction.
Model precision and Triton-neuron storage/compute precision are independently
configured by :class:`PrecisionConfig`. MCore language-model precision remains
owned by ``distributed.llm`` and its native MCore configuration.

Tutorials: :doc:`中文 </tutorials/cn/precision>` |
:doc:`English </tutorials/en/precision>`.

Transformer Engine delayed-scaling checkpoints may contain pickled extra state.
TE 2.18 and later require ``NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`` to load that
metadata. Enable it only for checkpoints from a trusted source; SpikingJelly does
not set it automatically.

.. automodule:: spikingjelly.activation_based.precision
   :members:
   :undoc-members:
   :show-inheritance:
