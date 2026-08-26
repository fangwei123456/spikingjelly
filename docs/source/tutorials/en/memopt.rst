Training Memory Optimization
============================

中文版： :doc:`../cn/memopt`

``spikingjelly.activation_based.memopt`` has two layers. The low-level API lets
you choose checkpoint boundaries directly. :func:`optimize_memory
<spikingjelly.activation_based.memopt.optimize_memory>` is an optional high-level
preset based on `Towards Lossless Memory-efficient Training of Spiking Neural
Networks via Gradient Checkpointing and Spike Compression
<https://openreview.net/forum?id=nrBJ0Uvj7c>`_. A network does not need to adopt
the preset to use memopt.

Custom Checkpoints
------------------

Use :func:`checkpoint <spikingjelly.activation_based.memopt.checkpoint>` for a
callable:

.. code-block:: python

    from spikingjelly.activation_based import memopt, neuron

    y = memopt.checkpoint(block, x)

Use :func:`checkpoint_module
<spikingjelly.activation_based.memopt.checkpoint_module>` when a module boundary
already expresses the intended recomputation region:

.. code-block:: python

    model.blocks[2] = memopt.checkpoint_module(model.blocks[2])

The wrapper preserves parameter names, parameter identities, and ``state_dict``
keys. Stateful neurons are recomputed from explicit functional state. Module
buffers such as BatchNorm running statistics are committed once, rather than
updated again during backward recomputation.

Temporal chunking is explicit. Only use it when splitting the selected inputs
along ``time_dim`` preserves the module's semantics:

.. code-block:: python

    model.neuron = memopt.checkpoint_module(
        model.neuron,
        chunks=2,
        chunked_args=(0,),
        time_dim=0,
    )

The chunked inputs must have the same nonzero temporal length, and the number of
chunks cannot exceed that length. Tensor outputs are concatenated along
``time_dim``; non-tensor output leaves must be identical for every chunk. Do not
temporally chunk training BatchNorm, attention that mixes time steps, or another
operation whose result depends on the complete temporal batch.

Input Compression
-----------------

A compressor is any stateless object with ``compress(tensor)`` and
``decompress(payload)`` methods. The payload owns all shape and dtype metadata,
so one compressor instance can safely serve concurrent calls.

.. code-block:: python

    model.spike = memopt.checkpoint_module(
        model.spike,
        compressor=memopt.BitSpikeCompressor(),
    )

``BitSpikeCompressor`` and ``BooleanSpikeCompressor`` require values that are
strictly zero or one. ``Uint8SpikeCompressor`` is for integer-valued spikes.
``SparseSpikeCompressor`` stores nonzero indices and is useful only when its
index payload is smaller than a dense representation. ``NullSpikeCompressor``
keeps the input dtype and values unchanged.

The Paper Preset
----------------

The high-level preset mutates a model in place and returns that same object:

.. code-block:: python

    sample = torch.zeros(4, 8, 128, device="cuda")
    memopt.optimize_memory(
        model,
        targets=(ResidualBlock,),
        example_forward=lambda current: current(sample),
        level=3,
        checkpoint_budget="balanced",
        split_fn=lambda block: (block.conv, block.neuron),
        can_chunk=lambda module: isinstance(module, neuron.BaseNode),
    )

The levels are cumulative:

``0``
    Strict no-op. ``example_forward`` is not required.
``1``
    Observe one representative forward and checkpoint the target modules with
    the largest first tensor inputs.
``2``
    Try ``split_fn`` descendants and retain a split only when measured training
    peak memory strictly decreases.
``3``
    Apply ``can_chunk`` once to the final checkpoint leaves and try increasing
    temporal chunk counts.
``4``
    Measure forward cost with five warmups and ten samples, then greedily remove
    expensive checkpoints while staying within the achieved memory peak.

``checkpoint_budget`` selects 50%, 75%, or 100% of eligible target modules for
``"speed"``, ``"balanced"``, or ``"memory"`` respectively. Ties follow model
order. Automatic compression is used only when every observed rank sees a
strictly binary first tensor input.

Levels 2-4 require the model and representative inputs on CUDA. Profiling
restores RNG state, training flags, buffers, neuron memories, and existing
parameter gradients after each trial. ``split_fn`` must return at least two
non-overlapping registered descendants of its argument. A failed or
non-improving candidate is reverted.

Distributed Training and Backends
---------------------------------

Pass a process group containing every DP and TP rank for the current PP stage.
Activation sizes and memory peaks use the group maximum, binary eligibility uses
the group minimum, and the stage leader broadcasts structural choices. Every
rank must call ``optimize_memory`` in the same order.

The built-in distributed Vision recipes construct this DP-by-TP stage group and
accept ``memopt_level``, ``memopt_checkpoint_budget``, and
``memopt_compress_inputs``. MCore training exposes the same fields. Evaluation,
prediction, generation, and artifact export always build an unwrapped model;
the transparent state dict makes training-time wrappers unnecessary there.

The public ``memopt.checkpoint`` API uses PyTorch's non-reentrant checkpoint
implementation and does not select a neuron backend. Torch, CuPy, and Triton
neuron support therefore
follows each neuron's normal functional-forward support. Test the exact model,
dtype, backend, compiler, and distributed topology used for training; a custom
backend that cannot run the functional neuron path is not made compatible by
memopt.

``torch.compile(..., fullgraph=True)`` supports the core uncompressed path and
dense Boolean/bit compression. Sparse payload size is data-dependent and may
require the compiler's dynamic-shape support.

Migration from the Previous API
-------------------------------

The old ``memory_optimization``, ``input_compressed_gc``, ``GCContainer``,
``TCGCContainer``, mutable compressor base class, summary/profile objects, and
module-side ``__spatial_split__`` protocol were removed. Use
``optimize_memory`` for the paper preset, or compose ``checkpoint`` and
``checkpoint_module`` for a smaller architecture-specific policy. No
compatibility aliases are retained.
