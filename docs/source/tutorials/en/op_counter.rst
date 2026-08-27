Operation Counters and Energy Estimation
=========================================

Author: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版： :doc:`../cn/op_counter`

This tutorial covers ``spikingjelly.activation_based.op_counter``. It counts
FLOPs, memory accesses, SynOps, MACs, and ACs during one execution, then uses
those counts to estimate energy. Results depend on input shape, spike sparsity,
and ``train``/``eval`` mode, so profile a configuration that represents the
target workload.

Overview
++++++++++++++++++++++++

Runtime Counting Modes
----------------------

``op_counter`` observes runtime calls through three context managers:

* ``DispatchCounterMode`` intercepts ATen operators;
* ``FunctionCounterMode`` intercepts ``torch.*`` functions;
* ``ModuleCounterMode`` records executed ``nn.Module`` forward and backward events.

The modes are active only inside their contexts and do not modify the model.
Multiple counters can observe the same execution. Unlike static shape analysis,
runtime counting distinguishes binary spikes from dense activations and reflects
changes in input sparsity and execution stage.

Basic Counting Workflow
++++++++++++++++++++++++

Using ``DispatchCounterMode``
------------------------------

1. instantiate one or more counters;
2. run one real forward or forward-backward pass inside ``DispatchCounterMode``;
3. read per-scope counts from ``get_counts()`` or the global total from ``get_total()``.

Both ``train()`` and ``eval()`` can be counted. For modules such as dropout or
batch normalization, use the mode that matches the target workload.

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

The package logger emits per-operation ``DEBUG`` records. They can be expensive
on large models, so enable them only while diagnosing counts:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

The dispatch examples use ``strict=False`` and skip unsupported auxiliary
operators. After confirming coverage, use ``strict=True`` to fail on unsupported
operators.

Using ``ModuleCounterMode``
---------------------------

Module-counter rule keys are ``("forward" | "backward", module_type)``.
``ModuleCounterMode`` manages hooks, scopes, and exception cleanup, but does not
reset counters. Scopes start with ``Global``, followed by the root module type
and qualified child paths.

.. code-block:: python

    memory_counter = op_counter.NeuromorphicMemoryAccessCounter()
    with op_counter.ModuleCounterMode(
        [memory_counter], model=model, strict=True
    ):
        _ = model(x)

    print(memory_counter.get_counts()["Global"])

After any mode exits, call ``mode.get_unsupported(counter)`` to inspect subjects
skipped by a non-strict run.

Available Counters
-------------------

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

The counters measure different work. A spike-driven linear layer may produce
SynOps and ACs but no MACs. ``SynOpCounter`` counts only binary spike inputs;
dense floating-point inputs produce zero SynOps.

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

The following example counts FLOPs, memory access, and arithmetic intensity for
one training step. Remove ``backward()`` for inference.

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

The result is the workload point for a roofline chart; combine it with hardware
peak FLOPs and bandwidth. The counts use an idealized model: two FLOPs per MAC,
one read per logical input, and one write per logical output. They exclude
tiling, caches, fusion, bank conflicts, and physical DRAM traffic.

High-Level Energy Models
++++++++++++++++++++++++

Model Overview
--------------

``op_counter`` currently exposes four high-level energy estimators:

* ``estimate_simple_energy``: simple runtime MAC/AC/memory energy;
* ``estimate_lemaire_energy``: Lemaire-aligned analytical forward inference energy;
* ``estimate_neuromc_runtime_energy``: runtime NeuroMC-style energy;
* ``estimate_spikesim_energy``: runtime SpikeSim-style Conv2d energy.

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
      - runtime energy for forward, backward, and optimizer stages
      - compute and memory under NeuroMC-like mapping rules
      - covers only supported fragments and stage semantics
    * - ``estimate_spikesim_energy``
      - SpikeSim-style Conv2d accelerator estimate
      - Conv2d stage energy with SpikeSim coefficients
      - only for supported Conv2d inference stages; not a general full-model energy estimator

The four estimators use different cost regimes and hardware assumptions. Do not
compare their absolute values.

Every report exposes ``model_info`` with a stable model ID, sources, technology,
precision, scope, and fidelity. ``config`` (``memory_config`` for NeuroMC) records
the actual cost configuration. ``paper`` and ``reference-code`` identify upstream
parity, ``source-aligned`` identifies an upstream-cost runtime adapter, and
``spikingjelly-defined`` identifies a formula specified by this project. Report
the estimator, execution stages, cost configuration, input type, and sparsity.

Simple Runtime Energy
---------------------

``estimate_simple_energy`` runs one forward pass and converts runtime counts with
``MAC * E_MAC + AC * E_AC + bytes * E_memory``.

Its main assumptions are:

* ``NeuromorphicMemoryAccessCounter`` independently counts weights and biases
  that are actually used, plus one read and one write per timestep for persistent
  neuron states;
* input currents and output spikes are treated as on-chip signal flow, not memory;
* it does not model FIFOs, addressing, routing, cache reuse, or hardware mapping;
* ``SynOps`` is an auxiliary subset of AC and is not charged a second time;
* defaults use Horowitz 2014 FP32 arithmetic and STEP Table 9's
  ``24.96 pJ/byte`` memory cost; SpikingJelly defines the traffic formula;
* FP16 and INT8 presets change arithmetic costs but do not quantize the model.

Lemaire Analytical Inference Energy
------------------------------------

``estimate_lemaire_energy`` runs one forward pass and maps synaptic operations,
MAC/AC work, addressing, neuron state, and per-layer SRAM accesses to the
Lemaire equations. Its main limits are:

* forward inference only;
* analytical estimation, not cycle-accurate hardware simulation;
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

NeuroMC Runtime Energy
-----------------------

``estimate_neuromc_runtime_energy`` profiles real execution fragments and maps
them to the fixed NeuroMC v1 constants and per-variable memory directions and
multipliers. It is a source-aligned runtime adapter and does not reproduce the
complete ZigZag mapping. The convenience function always runs forward. Adding
``target`` and ``loss_fn`` runs backward; adding ``optimizer`` also estimates
the optimizer stage. Use
:class:`NeuroMCEnergyProfiler <spikingjelly.activation_based.op_counter.neuromc.core.NeuroMCEnergyProfiler>`
for manual stages.

Its main limits are:

* unsupported energy-bearing operators reject totals;
* manual profiling passes mapping semantics explicitly with
  ``stage(name, phase=..., reuse_weights=..., batch_norm_backward=...)``; a stage
  name reused in one context must keep the same options;
* the convenience function clears existing gradients but does not call
  ``optimizer.step()`` or update parameters;
* repeated calls to one module must all participate in backward; selective
  backward through only some calls is rejected as ambiguous;
* results come from a hardware model, not measurements from a real chip.

SpikeSim Runtime Energy
-----------------------

``estimate_spikesim_energy`` counts executed Conv2d inference stages. The default
``dense`` mode uses the author-code PE-cycle formula; ``event`` uses a sparse
formula defined by SpikingJelly. Its main limits are:

* the model should be in ``eval`` mode; with the default ``strict=True``,
  unsupported Conv2d stages and empty reports fail;
* only supported Conv2d forward stages enter the main energy path;
* with the default ``activity_mode="dense"``, runtime spike sparsity does not reduce energy;
* ``activity_mode="event"`` selects ``spikingjelly_spikesim_event_v1``
  model using SpikeSim constants and SpikingJelly's documented A/R/Z sparse formula;
* ``require_if_lif_neurons=True`` accepts only IF/LIF-style neurons.

Energy Estimation Example
+++++++++++++++++++++++++++++++++++

Simple Energy Example
---------------------

Run inference estimators after ``model.eval()``. The first example uses Simple
Energy:

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

Select another cost regime explicitly:

.. code-block:: python

    cfg = op_counter.SimpleEnergyConfig(
        cost_config=op_counter.SimpleEnergyCostConfig.fp16()
    )
    report_fp16 = op_counter.estimate_simple_energy(model, x, config=cfg)
    print("FP16-regime energy (pJ):", report_fp16.energy_total_pj)

The Lemaire estimator also includes addressing and neuron state:

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

Validation and Sources
++++++++++++++++++++++

Primary Sources
---------------

* Simple arithmetic costs: `Horowitz 2014 <https://doi.org/10.1109/ISSCC.2014.6757323>`_;
  memory coefficient: `STEP Table 9 <https://openreview.net/pdf?id=SzwU2XrXIS>`_.
* Lemaire: `An Analytical Estimation of Spiking Neural Networks Energy Efficiency
  <https://arxiv.org/abs/2210.13107>`_.
* SpikeSim dense: author-code commit
  `c2627bc <https://github.com/Intelligent-Computing-Lab-Yale/SpikeSim/commit/c2627bc091a47bdcb630ca6207eaf44a00bd1da4>`_.
* NeuroMC: author-code commit
  `712c66f <https://github.com/dayanhn/NeuroMC/commit/712c66f47cf76ae530a55f8bcad3858bd68788de>`_.

Relative-Trend Cross-Check
--------------------------

This benchmark checks one limited question: does SpikingJelly preserve the
source model's relative trends on selected cases? Each case records one
``(E_origin, E_SJ)`` pair. SpikeSim and NeuroMC use pinned author code. Lemaire
has no public code, so its reference values follow equations (1)--(20). The
reference path receives only static topology, tensor dimensions, and
independently observed firing counts; it does not read SpikingJelly reports.

Kendall's tau-b is the primary metric, with a paired 2,000-sample 95% bootstrap
interval. Spearman's rho and log-Pearson ``r`` provide secondary views. The P90
symmetric factor removes the median multiplicative scale before measuring
relative error. The predeclared ``tau-b >= 0.80`` and ``P90 <= 1.50x`` lines are
comparison guides, not accuracy criteria.

* **Kendall's tau-b** compares the ordering of every pair of cases. ``1`` means
  identical ordering, ``0`` means no consistent ranking association, and
  ``-1`` means completely reversed ordering.
* **Spearman's rho** correlates the ranks of the two score sets. It also ranges
  from ``-1`` to ``1`` and is more sensitive than tau-b to how far individual
  cases move in the ranking.
* The **P90 symmetric factor** is the empirical 90th percentile of relative
  error factors after removing a fixed scale difference. ``1.0x`` is ideal;
  ``1.5x`` places that percentile between ``1 / 1.5`` and ``1.5`` times the
  reference relative value.

.. list-table:: Validation results
   :header-rows: 1
   :widths: 20 12 23 14 14 14 14

   * - Estimator mode
     - Comparable cases
     - Kendall tau-b (95% bootstrap interval)
     - Spearman rho
     - Log-Pearson r
     - P90 factor
     - Median scale E_SJ/E_origin
   * - Lemaire
     - 12
     - 0.939 [0.729, 1.000]
     - 0.979
     - 0.998
     - 1.478x
     - 0.877x
   * - SpikeSim dense
     - 7 (+5 stress)
     - 1.000 [1.000, 1.000]
     - 1.000
     - 1.000
     - 1.000x
     - 1.000x
   * - NeuroMC
     - 13
     - 0.795 [0.541, 0.971]
     - 0.934
     - 0.981
     - 1.189x
     - 0.396x

.. figure:: ../../_static/tutorials/op_counter/energy_model_validation.png
   :alt: Normalized reference and SpikingJelly scores and per-model tau-b and P90 minus one
   :align: center

   The left panel compares normalized score pairs. Proximity to the diagonal
   indicates similar relative trends, not accurate absolute energy. The right
   panel summarizes per-model tau-b and ``P90 - 1``.

Lemaire's tau-b and P90 meet both comparison lines. NeuroMC meets the P90 line;
its ``0.795`` tau-b is slightly below ``0.80``. SpikeSim's seven comparable
cases match because the dense runtime directly implements the author formula;
this result mainly checks integration and calculation. Five dynamic stress
cases are excluded from the correlation, with runtime/static ratios from
``0.500x`` to ``3.000x``. The benchmark does not assign an overall Pass.

**Limitations:**

* Each group contains only 7, 12, or 13 selected cases. They do not represent
  broader networks or firing patterns, and the bootstrap interval only reports
  resampling stability within these cases.
* Both paths share topology, dimensions, and firing counts, so network scale can
  produce high correlation by itself. High tau or rho does not validate each
  energy term or coefficient.
* The references are analytical models, not hardware measurements. Correlation
  shows similarity to their trends but cannot establish physical energy
  accuracy.
* Ranking and P90 weaken or remove absolute scale. A fixed absolute bias can
  coexist with favorable metrics.
* Coverage remains incomplete: published equations reconstruct Lemaire, this
  benchmark validates only NeuroMC forward energy, and Simple Energy and
  SpikeSim event have no independent end-to-end external estimator.

Run the benchmark manually with:

.. code-block:: bash

    uv run python benchmark/energy_model_validation.py \
        --spikesim-root /path/to/SpikeSim \
        --neuromc-root /path/to/NeuroMC

The exact case inputs, paired scores, metrics, repository revision, dependency
versions, and reference revisions are available in the
:download:`case-level CSV <../../_static/tutorials/op_counter/energy_model_validation.csv>`.
The benchmark depends on pinned external repositories and does not run in CI.

Summary
++++++++++++++++++++++++

``op_counter`` records the executed work for a given input. Use basic counters
for operations and traffic, and energy estimators for relative comparisons
within their stated scope. Do not compare absolute values across estimators.
