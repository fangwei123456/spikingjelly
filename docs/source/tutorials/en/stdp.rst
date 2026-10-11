Local Synaptic Plasticity and STDP
===================================

中文版本: :doc:`../cn/stdp`

Purpose and supported scope
-----------------------------

``learning`` provides local synaptic weight-update tools. STDP activity
correlations are not gradients of a classification loss. Writing updates to
``weight.grad`` borrows an optimizer update channel and does not guarantee task
improvement. Encoding, competition, homeostasis, initialization and readout are
parts of the training method.

See :doc:`../../APIs/spikingjelly.activation_based.learning` for supported modes,
time constants, layouts, reward alignment and migration. Ordinary STDP uses
``1-1/tau`` decay; MSTDP/MSTDPET use ``exp(-1/tau)``. Their distinct
discretizations are retained rather than unified for appearance.

Rule updates and state
------------------------

For current pre/post spikes ``p[t]`` and ``q[t]``, ordinary STDP computes:

.. math::

   a_t &= (1-1/\tau_{pre})a_{t-1}+p_t \\
   b_t &= (1-1/\tau_{post})b_{t-1}+q_t \\
   \Delta W_t &= F_{post}(W)(q_t\otimes a_t)-F_{pre}(W)(b_t\otimes p_t)

Updates sum over batch and convolution spatial positions; multi-step execution
also sums over time. Changing batch/spatial size changes update magnitude, so
callers explicitly adjust learning rate or ``scale``. Current spikes enter
traces first; simultaneous spikes only cancel for suitable modulation functions.
Fixed-weight sequence accumulation differs from updating weights every step.

This example applies returned updates without gradients or an optimizer:

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import learning, neuron

    fc = nn.Linear(4, 3, bias=False)
    sn = neuron.IFNode()
    learner = learning.STDPLearner("s", fc, sn, 20., 20.)
    try:
        with torch.no_grad():
            sn(fc((torch.rand(2, 4) > 0.7).float()))
            delta_w = learner.step(on_grad=False, scale=1 / 2)
            fc.weight.add_(delta_w, alpha=0.01)
        learner.reset()
        sn.reset()
    finally:
        learner.close()


``step`` consumes records but retains traces. Reset between independent samples
or episodes; retain traces for continuous activity. Reset both neuron and learner
before changing batch size. Disable explicitly for evaluation: ``net.eval()``
does not disable recording. Close afterwards. A silent post neuron with zero
post trace can have no update; a displayed loss does not demonstrate learning.

Mechanism demonstrations
--------------------------

The scripts below show traces, activity and weight changes. They run without a
GUI by default and write to an untracked directory. They are not task benchmarks.
Use ``--show`` for a window and ``--output-dir`` for a different directory.

.. code-block:: bash

    uv run python -m spikingjelly.activation_based.examples.stdp_trace
    uv run python -m spikingjelly.activation_based.examples.mstdp
    uv run python -m spikingjelly.activation_based.examples.mstdpet


Reward learners require exactly one activity pair per step. ``reward[t]`` acts
on ``eligibility[t-1]``; the initial update is zero. Reward after the final
activity needs a subsequent forward; an empty step does not advance time.
MSTDP accepts scalar or ``[N]`` reward; MSTDPET accepts scalar reward. Remove
``batch_size`` from MSTDP construction. Replace accumulated forwards with one
reward step immediately after every forward.

Combining with gradient training
----------------------------------

Training different layers with different rules and adding both updates to the
same parameter are different experiments. For layer-wise separation, clear
task gradients of STDP-only parameters after backward, then inject STDP updates.
For additive updates, retain task gradients. Do not detach the loss before
backward. ``sn`` must produce actual post spikes; an adjacent BatchNorm is not
a spiking neuron. Only SGD without momentum or weight decay corresponds to
simple ``W += lr * delta_w``; other optimizers alter the dynamics. The following
only demonstrates additive update wiring, not task effectiveness:

.. code-block:: python

    import torch
    from torch import nn
    from spikingjelly.activation_based import learning, neuron

    fc = nn.Linear(4, 3, bias=False)
    sn = neuron.IFNode()
    learner = learning.STDPLearner("s", fc, sn, 20., 20.)
    optimizer = torch.optim.SGD(fc.parameters(), lr=0.01)
    try:
        optimizer.zero_grad()
        output = sn(fc((torch.rand(2, 4) > 0.5).float()))
        loss = (output - 1).square().mean()
        loss.backward()
        learner.step(on_grad=True, scale=0.01)
        optimizer.step()
        learner.reset()
        sn.reset()
    finally:
        learner.close()


Complete MNIST example
----------------------

``stdp_mnist`` uses a bias-free 784→3200 feedforward layer, identical current LIF neurons,
and ``ElementWiseRecurrentContainer``. Feedforward weights start randomly and use the
existing ``STDPLearner`` plus explicit weight constraints, without gradients, pretraining
or a trainable readout. Fixed negative recurrent currents provide soft competition.
Multiple neurons can fire together; there is no single-winner filter or neighbor reset.

Model and local adaptation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For input spikes x and feedforward current z=Wx:

.. math::

    U_j[t] &= g_j\left(z_j[t]-\frac{1}{N}\sum_k z_k[t]\right)
      -a\left(\sum_k s_k[t-1]-s_j[t-1]\right),\\
    I_j[t] &= I_j[t-1]+\frac{U_j[t]-I_j[t-1]}{5},\\
    H_j[t] &= V_j[t-1]+\frac{I_j[t]+B_j[t-1]-A_j[t-1]-V_j[t-1]}{100}.

This is ordinary SJ ``SimpleLIFNode(tau=100, decay_input=True)`` current integration.
The base threshold is 0.01, own reset is zero, and own refractory period is five steps;
only the neuron's own voltage is held during its refractory period. Signed currents
can produce negative voltage; no voltage floor or conductance denominator is used.
Population-mean subtraction suppresses common drive within each sample. Recurrent
inhibition depends only on other neurons' previous-step spikes.

Every neuron has the same parameters; its history produces different adaptive states:

.. math::

    A_j[t] &= \exp(-1/200)A_j[t-1]+0.01s_j[t],\\
    B_j[t] &= B_j[t-1]+\eta_B\left(\frac{120}{NT}
        -\frac{1}{M}\sum_{n=1}^{M}s_{nj}[t]\right).

A is fast own-spike adaptation current, reducing repeated responses. B is a slow signed
intrinsic excitability current, using unlabeled local rate error to adjust participation.
M is actual batch size, N neuron count, and T sample duration. B starts uniformly in
[-0.01,0] and freezes after training. This is engineering homeostasis, not an exact
biophysical channel model, global stability theorem, or guarantee of exact target rate
while weights change. Positive B may permit spontaneous activity; sparsity and balanced
mean rates do not substitute for task accuracy.

Input gain updates at the end of each training batch. C is the pixel sample covariance
of the fixed split-seed42 learning 50,000 images, without labels. With
``f_j=W_j-mean(W,dim=0)``:

.. math::

    \sigma_j^2=f_j C f_j^\top,\qquad
    g_j=\sqrt{\sigma_{j,\mathrm{initial}}^2/\sigma_j^2}.

This preserves each initial random projected-current variance scale, reducing operating
point drift from weight learning. Degenerate/nonfinite variance fails explicitly. C,
initial reference variance and gain are saved in the checkpoint and frozen in evaluation.
This is an unlabeled training-data constraint, not task gradients or test-statistic adaptation.

Encoding, plasticity and readout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input uses ordinary ``SimpleIFNode(v_threshold=1, v_reset=None)`` with an independent
uniform initial phase in [0,1) per sample/pixel. Each step adds ``pixel*0.064`` and applies
own soft reset. Across 700 steps, pixel counts differ from expectation by at most about
one spike. Regular firing reduces count noise and changes temporal correlations; mean
rate preservation does not imply identical Poisson learning trajectories.

Pre/post STDP traces both use 20 steps and include the current events.
``f_pre=nu*weight`` and ``f_post=nu*(1-weight)``, with default nu=0.00229619779275505.
Each step returns ``step(on_grad=False, scale=1/actual_batch_size)`` and directly adds
its delta to weights, without averaging time. Weights are clipped to [0,1], normalized
to row sum 78.4, then clipped again. Writing ``.grad`` elsewhere remains an optimizer
delivery channel; this example does not use it.

Batches clear voltage, A, refractory/filter state, recurrent output and STDP traces,
retaining B, weights and gain. Assignment/evaluation freeze every persistent learning
state while fast dynamics continue. Labels only provide post-training statistical neuron
class assignment and metrics. Each neuron receives its highest mean-count class;
nonfiring neurons remain unassigned. Predictions average raw counts across assigned
neurons per class; all-zero scores are unclassified and counted as errors.

Fixed data protocol: learning 50,000, assignment 5,000, validation 5,000, official test 10,000.
Defaults are one epoch, batch 32, 700 steps. Negative recurrent strength a is
0.09978118161925603; ``--homeostasis-gain`` is 0.007774997243242785,
``--bias-initial`` is 0.01 and native ``--v-threshold`` is 0.01. Time constants are in
simulation steps; current-filter tau is at least one.

.. code-block:: bash

    uv run --no-sync python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --seed 0 --data-dir logs/plasticity/data --download \
        --output-dir logs/plasticity/current/seed-0
    uv run --no-sync python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/current/seed-0/selected.pt \
        --data-dir logs/plasticity/data --output-dir logs/plasticity/current/seed-0

Use the same defaults and one epoch with ``--seed`` 1/2 and separate directories.
``--checkpoint`` uses saved dynamics/mapping without reassignment or training;
replaying results does not require retraining. Small sample-count options are smoke
checks, not evidence of complete-test performance.

Migration and measured results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This replaces the previous conductance example. ``--theta-plus`` and ``--theta-initial``
become native-current ``--homeostasis-gain`` and ``--bias-initial``. Old conductance/hard
competition checkpoints are not loaded automatically; use their archived source.
Legacy fields of the measured current experiment undergo a one-off explicit configuration
conversion, keeping all tensors/maps unchanged. There is no permanent compatibility
branch. New output uses ``logs/plasticity/current`` separately from old conductance experiments; use a fresh directory for repeated runs.

Independent validation selected the recipe and one epoch: same-width Poisson full
training gave 88.76%; IF evaluation of that frozen model gave 89.16%; fresh IF short
training gave 86.72%/control 82.94%; fresh IF full training gave 89.58%/control 83.04%.
These are distinct stages, not interchangeable descriptions of one run. Seeds 1/2
then trained from scratch with the same frozen recipe/epoch count. All three models/maps
were fixed before official testing; test scores did not tune the recipe or exclude seeds.

.. list-table:: Full MNIST tests: regular IF input and current LIF
   :header-rows: 1

   * - Seed
     - STDP
     - Frozen feedforward control
     - Difference (percentage points)
   * - 0
     - 90.63%
     - 84.01%
     - +6.62
   * - 1
     - 89.98%
     - 83.20%
     - +6.78
   * - 2
     - 90.06%
     - 84.06%
     - +6.00

Mean accuracy is **90.2233%**, population standard deviation **0.2894 percentage points**,
and mean paired improvement **6.4667 percentage points**. Every seed uses all 10,000 test
samples. One silent seed 0 sample is counted as an error. Environment: Python 3.11.14,
PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, V100S-PCIE-32GB testing.
These measured shallow-recipe results are not a Diehl–Cook paper reproduction or a
claim about arbitrary deep pure-STDP networks. Raw JSON/config/checkpoints/source hashes
and commands are retained. Migration requires same-checkpoint numerical and full
evaluation verification.


References
------------

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
