Operation Counters and Energy Estimation
=========================================

Author: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版： :doc:`../cn/op_counter`

This tutorial introduces the ``spikingjelly.activation_based.op_counter`` module.

The module serves two closely related goals:

1. Count model-side runtime costs such as FLOPs, memory accesses, SynOps, MACs, and ACs;
2. Build higher-level energy estimators on top of those counters.

The examples in this tutorial are intentionally small so that they can be run on CPU in a few seconds.

When you profile your own model, always use representative input shapes and representative spike sparsity.
``op_counter`` is runtime-driven, so changing the input can change the measured counts and the estimated energy.

Overview
++++++++++++++++++++++++

What Is ``op_counter``?
-------------------------

``op_counter`` is a runtime profiling toolkit built on PyTorch dispatch,
function interception, and module hooks.
Instead of estimating counts only from static layer shapes, it observes one real execution of your
model and records what actually happened under the given input.

This is especially useful for SNNs because many quantities depend on runtime activity:

* binary spikes and dense activations should not be interpreted in the same way;
* the same layer can behave differently under different input sparsities;
* some energy models need forward-only profiling, while others also need backward and optimizer stages.

Why Counter Modes Matter
-------------------------

Counters do not modify your model. They are activated only inside a context manager.
This design keeps the profiling logic explicit:

* outside the context, the model behaves normally;
* inside the context, supported operators are intercepted and counted;
* multiple counters can run together during the same execution.

The three entry points are ``DispatchCounterMode``, ``FunctionCounterMode``, and
``ModuleCounterMode``. They route ATen operators, ``torch.*`` functions, and
executed ``nn.Module`` forward/backward events, respectively.

Basic Counting Workflow
++++++++++++++++++++++++

Using ``DispatchCounterMode``
------------------------------

The basic workflow is:

1. instantiate one or more counters;
2. run one real forward or forward-backward pass inside ``DispatchCounterMode``;
3. read per-scope counts from ``get_counts()`` or the global total from ``get_total()``.

For plain counting, ``train()`` and ``eval()`` are both usable. But if your model contains modules whose runtime behavior changes with mode, such as dropout or batch normalization, choose the mode that matches the scenario you actually want to profile.

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, op_counter

    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        neuron.IFNode(),
        nn.Linear(16, 4, bias=False),
    )
    x = (torch.rand(2, 8) > 0.5).float()

    flop_counter = op_counter.FlopCounter()
    mem_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode(
        [flop_counter, mem_counter],
        strict=False,
    ):
        _ = model(x)

    print("FLOPs:", flop_counter.get_total())
    print("Memory access (bytes):", mem_counter.get_total())
    print("Global FLOP record:", flop_counter.get_counts()["Global"])

Operation-counter diagnostics are emitted as ``DEBUG`` records by the package logger.
They include per-operation details and can be expensive for large models, so enable
them only while diagnosing a counting run:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

The examples in this tutorial use ``strict=False`` so that unsupported auxiliary operators do not stop execution immediately.
When you want unsupported operators to fail immediately instead of being skipped, switch to ``strict=True`` after you have confirmed that the relevant operator path is fully covered by the counters you selected.

Using ``ModuleCounterMode``
---------------------------

Module-counter rule keys are ``("forward" | "backward", module_type)``. The
mode owns hook lifetime, module scopes, ignored subtrees, and exception cleanup;
it does not reset counters automatically.

.. code-block:: python

    memory_counter = op_counter.NeuromorphicMemoryAccessCounter()
    with op_counter.ModuleCounterMode(
        [memory_counter], model=model, strict=True
    ):
        _ = model(x)

    print(memory_counter.get_counts()["Global"])

After any of the three modes exits, call ``mode.get_unsupported(counter)`` to
inspect subjects skipped by a non-strict run.

Although this first example already uses an SNN-style block with ``IFNode``, it still focuses only on the generic counter workflow.
The SNN-specific interpretation of spike-driven metrics such as ``SynOps`` is introduced separately below.

Available Counters
-------------------

The most commonly used counters are:

* :class:`FlopCounter <spikingjelly.activation_based.op_counter.flop.FlopCounter>`:
  counts floating-point operations. It is useful for ANN-style compute intensity analysis.
* :class:`MemoryAccessCounter <spikingjelly.activation_based.op_counter.memory_access.MemoryAccessCounter>`:
  counts runtime memory traffic in bytes.
* :class:`SynOpCounter <spikingjelly.activation_based.op_counter.synop.SynOpCounter>`:
  counts spike-driven synaptic additions. Dense floating-point inputs do not contribute to SynOps.
* :class:`MACCounter <spikingjelly.activation_based.op_counter.mac.MACCounter>`:
  counts multiply-accumulate operations.
* :class:`ACCounter <spikingjelly.activation_based.op_counter.ac.ACCounter>`:
  counts addition-like arithmetic work that is not modeled as MAC.

These counters are complementary rather than interchangeable. For example, a spike-driven linear layer may have non-zero SynOps and ACs while having zero MACs.

``SynOpCounter`` deserves one extra remark: it only becomes meaningful when the relevant layer really receives binary spike inputs.
If the same layer receives dense floating-point activations, the SynOp count can legitimately be zero.

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False)
    spike_x = (torch.rand(2, 8) > 0.5).float()

    synop_counter = op_counter.SynOpCounter()
    with op_counter.DispatchCounterMode([synop_counter], strict=False):
        _ = model(spike_x)

    print("SynOps:", synop_counter.get_total())

Roofline Analysis Example
--------------------------

The following example reproduces the basic roofline ingredients: FLOPs, memory access, and arithmetic intensity for one training step.
If you only care about inference roofline, remove the ``backward()`` call.

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
        nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
    )
    x = torch.rand(1, 2, 16, 16)

    flop_counter = op_counter.FlopCounter()
    mem_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([flop_counter, mem_counter], strict=False):
        y = model(x)
        y.sum().backward()

    flops = flop_counter.get_total()
    mem_bytes = mem_counter.get_total()
    intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")

    print("total FLOPs:", flops)
    print("total memory access (bytes):", mem_bytes)
    print("arithmetic intensity (FLOPs/byte):", intensity)

This example does not draw the roofline figure for you. It provides the measured workload point
that you can place on a roofline chart after combining it with hardware peak FLOPs and bandwidth.

The GPU regime is an ideal analytical estimate, not a kernel simulator: matrix
multiplication uses two FLOPs per MAC, and logical tensor traffic reads each input
once and writes each output once. Tiling, caches, fusion, bank conflicts, and real
DRAM traffic are outside scope. After a non-strict run, call
``mode.get_unsupported(counter)`` to inspect skipped ATen operators.

High-Level Energy Models
++++++++++++++++++++++++

Model Overview and Boundaries
------------------------------

``op_counter`` currently exposes four high-level energy estimators:

* ``estimate_simple_energy``: simple runtime MAC/AC/memory energy;
* ``estimate_lemaire_energy``: Lemaire-aligned analytical forward inference energy;
* ``estimate_neuromc_runtime_energy``: runtime NeuroMC-style energy;
* ``estimate_spikesim_energy``: runtime SpikeSim-style Conv2d energy.

They do not answer the same question. Their intended use and boundaries are:

.. list-table::
    :header-rows: 1

    * - Estimator
      - Main purpose
      - Covers
      - Main boundary
    * - ``estimate_simple_energy``
      - normalized runtime energy comparison
      - MAC, AC, weight/bias reads, and persistent neuron-state accesses
      - excludes signal flow, FIFOs, routing, addressing, and hardware mapping
    * - ``estimate_lemaire_energy``
      - forward SNN inference estimate aligned with Lemaire-style formulas
      - ops, addressing, runtime-sized memory traffic, neuron-state traffic
      - forward inference only; analytical estimate, not hardware simulation
    * - ``estimate_neuromc_runtime_energy``
      - more complete runtime energy for forward, backward, and optimizer stages
      - compute and memory under NeuroMC-like mapping rules
      - exact only for the supported fragment set and stage semantics
    * - ``estimate_spikesim_energy``
      - SpikeSim-style Conv2d accelerator estimate
      - Conv2d stage energy with SpikeSim coefficients
      - only for supported Conv2d inference stages; not a general full-model energy estimator

The most important rule is: do not compare absolute numbers across different estimators as if they shared the same hardware assumptions.
Each estimator uses its own cost regime and modeling scope.

Every report exposes ``model_info`` with a stable model ID, sources, technology,
precision, scope, and fidelity. ``config`` (``memory_config`` for NeuroMC) records
the actual cost configuration. ``paper`` and ``reference-code`` identify upstream
parity, ``source-aligned`` identifies an upstream-cost runtime adapter, and
``spikingjelly-defined`` identifies a formula specified by this project.

Simple Runtime Energy
---------------------

``estimate_simple_energy`` is the simplest high-level estimator.
It runs one real forward pass and converts runtime MAC, AC, and logical
neuromorphic-memory byte counts into energy using three explicit costs:
``MAC * E_MAC + AC * E_AC + bytes * E_memory``.

Its intended use is normalized comparison, for example:

* comparing two architectures under the same cost regime;
* comparing spike-driven and dense execution under the same runtime workload;
* reporting a transparent compute-versus-memory energy breakdown;
* applying the FP32, FP16, or INT8 arithmetic presets.

Its boundary is equally important:

* ``NeuromorphicMemoryAccessCounter`` independently counts weights and biases
  that are actually used, plus one read and one write per timestep for persistent
  neuron states;
* input currents and output spikes are treated as on-chip signal flow, not memory;
* it does not model FIFOs, addressing, routing, cache reuse, or hardware mapping;
* ``SynOps`` is an auxiliary subset of AC and is not charged a second time;
* the default memory cost is STEP Table 9's ``3.12 pJ/bit`` or
  ``24.96 pJ/byte``; the traffic formula itself is not STEP's formula;
* FP16 and INT8 presets select comparison costs and do not quantize the model.

By default it uses Horowitz 2014 FP32 arithmetic and the documented memory-cost
reference above. If you want a different regime, pass an explicit
``SimpleEnergyCostConfig``.

Lemaire Analytical Inference Energy
------------------------------------

``estimate_lemaire_energy`` is an inference-only analytical estimator aligned with the Lemaire-style SNN energy literature.
Unlike a purely static formula, the current implementation still runs one real forward pass to collect runtime counts and memory bytes.

It includes:

* synaptic operation counts;
* MAC and AC-like work;
* addressing counts;
* neuron-state reads, writes, and state arithmetic;
* memory energy estimated from runtime accesses and per-layer local-SRAM capacity.

Its boundaries are:

* forward inference only;
* analytical estimation rather than cycle-accurate hardware simulation;
* operations and accesses keep the paper's fixed 32-bit regime regardless of the
  host tensor dtype;
* parameter, FIFO, and potential accesses are priced against each layer's local
  SRAM capacity before energy is aggregated;
* binary inputs use the SNN event formulas, while sparse non-binary inputs remain
  on the dense FNN path;
* grouped and depthwise convolutions use output channels per group for spike fanout;
* neuron modules are limited to the paper's IF/LIF scope; other ``BaseNode`` types
  are rejected by default;
* SNN FIFOs hold 1000 messages by default; override
  ``snn_fifo_capacity_elements`` for another assumption;
* ``strict=True`` is the default and rejects unsupported transposed convolutions;
  explicitly setting it to ``False`` warns and omits them.

Use this estimator when you need a richer forward SNN inference estimate than simple runtime energy,
but do not need backward or optimizer modeling.

NeuroMC Runtime Energy
-----------------------

``estimate_neuromc_runtime_energy`` profiles real execution fragments and maps
them to the fixed NeuroMC v1 constants and per-variable memory directions and
multipliers. It is a source-aligned runtime adapter, not a reproduction of the
complete ZigZag mapping.

It supports several usage levels:

* forward inference only;
* one full training step through ``forward -> backward -> optimizer``;
* manual staged profiling through
  :class:`NeuroMCEnergyProfiler <spikingjelly.activation_based.op_counter.neuromc.core.NeuroMCEnergyProfiler>`.

Its strengths are:

* stage-aware reports such as ``forward``, ``backward``, and ``optimizer``;
* support for ANN, SNN, and mixed execution paths under supported fragments;
* explicit handling of process categories such as spike generation, batch normalization, and optimizer work.

Its boundaries are:

* unsupported energy-bearing operators reject the report;
* manual profiling passes mapping semantics explicitly with
  ``stage(name, phase=..., reuse_weights=..., batch_norm_backward=...)``;
* the training convenience path captures backward and estimates the optimizer but
  does not call ``optimizer.step()`` or update parameters;
* it is still a hardware-model-based estimate, not a measurement from a real chip.

Use this estimator when you need training-stage energy or online-learning stage breakdowns.

SpikeSim Runtime Energy
-----------------------

``estimate_spikesim_energy`` targets a much narrower question:
how much energy do the supported Conv2d inference stages consume under a SpikeSim-style accelerator model?

By default, it preserves the dense PE-cycle energy path of the released SpikeSim implementation,
while using runtime profiling to discover the actual Conv2d stages and shapes.

Its boundaries are strict:

* ``strict=True`` is the default, so unsupported Conv2d stages and empty reports fail;
* it is only for supported Conv2d forward inference stages;
* it is not a general-purpose full-model energy estimator;
* with the default ``activity_mode="dense"``, runtime spike sparsity does not reduce energy;
* ``activity_mode="event"`` is the stable ``spikingjelly_spikesim_event_v1``
  model using SpikeSim constants and SpikingJelly's documented A/R/Z sparse formula;
* when ``require_if_lif_neurons=True``, the model is expected to stay within IF/LIF-style neuron assumptions;
* non-Conv2d work is outside the main energy path.

Use this estimator when your target question is specifically about SpikeSim-style Conv2d accelerator energy.
If your target is broader forward or training energy, consider ``estimate_lemaire_energy`` or ``estimate_neuromc_runtime_energy`` instead.

Inference Energy Estimation Example
+++++++++++++++++++++++++++++++++++

Simple Energy Example
---------------------

Before using an inference-oriented energy estimator, call ``model.eval()`` first.
If you want to include backward or optimizer stages, switch to ``estimate_neuromc_runtime_energy`` instead.

The following example uses the simplest energy model to estimate forward inference energy.

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import op_counter

    model = nn.Linear(8, 4, bias=False).eval()
    x = torch.rand(2, 8)

    report = op_counter.estimate_simple_energy(model, x)

    print("total energy (pJ):", report.energy_total_pj)
    print("compute energy (pJ):", report.energy_compute_pj)
    print("MAC energy (pJ):", report.energy_mac_pj)
    print("AC energy (pJ):", report.energy_ac_pj)
    print("memory energy (pJ):", report.energy_memory_pj)
    print("counts:", report.counts)

If you want a different cost regime:

.. code-block:: python

    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig.fp16()
    )
    report_fp16 = op_counter.estimate_simple_energy(model, x, config=cfg)
    print("FP16-regime energy (pJ):", report_fp16.energy_total_pj)

This example is intentionally kept simple so that it focuses on the basic energy-estimation workflow.
For more detailed forward SNN inference modeling, replace the entry point with ``estimate_lemaire_energy``.

If you want a richer forward-only inference estimate that also includes memory, addressing, and neuron-state effects,
you can switch to the Lemaire-style estimator:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, op_counter

    model_snn = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        neuron.IFNode(),
        nn.Linear(16, 4, bias=False),
    ).eval()
    spike_x = (torch.rand(2, 8) > 0.5).float()

    lemaire_report = op_counter.estimate_lemaire_energy(model_snn, spike_x)
    print("Lemaire total (pJ):", lemaire_report.total_pj)
    print("Lemaire breakdown:", lemaire_report.breakdown_pj)

Practical Advice
+++++++++++++++++

Choosing a Counter or Energy Model
-----------------------------------

Use the tool that matches your question:

* if you need host-execution tensor traffic, use ``MemoryAccessCounter``;
* if you need simple neuromorphic weight/state traffic, use ``NeuromorphicMemoryAccessCounter``;
* if you need FLOPs or SynOps, use the corresponding basic counter;
* if you need roofline inputs, combine ``FlopCounter`` and ``MemoryAccessCounter``;
* if you need a simple normalized runtime compute-and-memory estimate, use ``estimate_simple_energy``;
* if you need forward SNN inference energy with memory and neuron-state effects, use ``estimate_lemaire_energy``;
* if you need training-stage breakdowns or optimizer energy, use ``estimate_neuromc_runtime_energy``;
* if you need SpikeSim-style Conv2d accelerator energy, use ``estimate_spikesim_energy``.

When reporting results, always state:

* which estimator you used;
* whether the run was forward-only or included backward/optimizer;
* the cost regime or hardware assumptions;
* the input type and sparsity conditions.

Primary Sources
---------------

* Simple arithmetic costs: `Horowitz 2014 <https://doi.org/10.1109/ISSCC.2014.6757323>`_;
  memory coefficient: `STEP Table 9 <https://openreview.net/pdf?id=SzwU2XrXIS>`_.
* Lemaire: `An Analytical Estimation of Spiking Neural Networks Energy Efficiency
  <https://arxiv.org/abs/2210.13107>`_.
* SpikeSim dense: author-code commit
  `c2627bc <https://github.com/Intelligent-Computing-Lab-Panda/SpikeSim/commit/c2627bc091a47bdcb630ca6207eaf44a00bd1da4>`_.
* NeuroMC: author-code commit
  `712c66f <https://github.com/dayanhn/NeuroMC/commit/712c66f47cf76ae530a55f8bcad3858bd68788de>`_.

Summary
++++++++++++++++++++++++

``op_counter`` is more than a single counter. It is a profiling framework that lets you move from low-level runtime counts to higher-level energy estimates.

For most workflows, the practical progression is:

1. start with direct counters to understand runtime behavior;
2. use FLOP and memory-access counts for roofline-style analysis;
3. choose the energy estimator whose scope matches your target question;
4. interpret the result under its own modeling boundary rather than as universal ground truth.
