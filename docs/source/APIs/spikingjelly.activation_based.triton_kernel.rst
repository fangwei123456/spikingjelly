spikingjelly.activation\_based.triton\_kernel package
=====================================================

.. note::

    Developers should **decide whether Triton backend is available** by:

    .. code:: python

        try:
            import triton
        except ImportError:
            triton = None

        if triton is not None:
            # Triton backend is available
            ...
        else:
            # Triton backend is not available
            ...

    :class:`MemoryModule <spikingjelly.activation_based.base.MemoryModule>` encapsulates this logic.

Predefined Neuron Kernels
----------------------------------

.. toctree::
   :maxdepth: 3

   neuron_kernel <spikingjelly.activation_based.triton_kernel.neuron_kernel>

FlexSN Implementation
---------------------------------------------------------------------

.. toctree::
   :maxdepth: 3

   flexsn <spikingjelly.activation_based.triton_kernel.flexsn>

Torch-to-Triton Transpiler
-----------------------------------------------------------------------------

.. toctree::
   :maxdepth: 3

   torch2triton <spikingjelly.activation_based.triton_kernel.torch2triton>

Spike Compressors
---------------------------------------------------------------------

.. toctree::
    :maxdepth: 2

    compress <spikingjelly.activation_based.triton_kernel.compress>

Utilities
---------------------------------------------------------------------

.. automodule:: spikingjelly.activation_based.triton_kernel.triton_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spikingjelly.activation_based.triton_kernel.dummy
   :members:
   :undoc-members:
   :show-inheritance:
