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

   spikingjelly.activation_based.triton_kernel.neuron_kernel

FlexSN Implementation
---------------------------------------------------------------------

.. toctree::
   :maxdepth: 3

   spikingjelly.activation_based.triton_kernel.flexsn

Torch-to-Triton Transpiler
-----------------------------------------------------------------------------

.. toctree::
   :maxdepth: 3

   spikingjelly.activation_based.triton_kernel.torch2triton

Spike Compressors
---------------------------------------------------------------------

.. toctree::
    :maxdepth: 2

    spikingjelly.activation_based.triton_kernel.compress

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
