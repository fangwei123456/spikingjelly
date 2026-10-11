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
------------------------

``stdp_mnist`` uses a bias-free 784→400 feedforward layer and adaptive LIF neurons.
All trainable feedforward weights learn through STDP, without backward,
pretraining or a trainable readout; recurrent connections are fixed.
Competition comes from inhibitory conductance delayed by one step, without
argmax winner selection. Neurons independently cross threshold and reset only
their own voltage; multiple neurons may fire together. A firing neuron's
refractory period applies only to that neuron, without resetting its neighbors.

Existing ``LinearRecurrentContainer`` receives concatenated ``[I_ff, 0]`` and
outputs separate excitatory current and inhibitory conductance increments.
Recurrent matrix R has zero diagonal and positive off-diagonal magnitude g;
it sums other neurons' previous spikes and supplies negative feedback through
the inhibitory conductance channel. The child neuron stores conductance G and
refractory counter r, with these discrete dynamics:

.. math::

    G[t] &= e^{-1/\tau_{inh}}G[t-1] + R s[t-1] \\
    H[t] &= \frac{0.99V[t-1]+I_{ff}[t]-G[t]}{1+G[t]} \\
    s_j[t] &= \mathbf{1}\{H_j[t]\ge 1+\theta_j^{-}[t],\ r_j[t-1]=0\}.

Refractory neurons stay at voltage 0 without firing. Others continue integrating,
and firing neurons reset themselves to 0. The inhibitory reversal potential is
-1; implicit conductance integration avoids Euler overshoot under strong
inhibition. Time is measured in simulation steps, with membrane time constant
100. During training the comparison offset is first
``theta_minus=exp(-1/1e5)*theta``; after firing, mean batch spikes increase it
to obtain the offset retained for the next step. Initial offsets are independently uniform in
``[0, theta_initial)``, breaking equal-threshold symmetry. These simplified
teaching dynamics are not a Diehl–Cook paper reproduction.

Each batch uses ``functional.reset_net`` to clear recurrent output, voltage,
conductance and refractory counters, then resets STDP traces. Threshold offsets
persist. Assignment and evaluation freeze weights and offsets and disable
monitor recording while simulating the same neuron dynamics. Fixed seeds control
initialization and encoding.

Split seed 42 divides training data into 50,000 unlabeled learning images,
5,000 assignment images and 5,000 validation images. The official 10,000 test
images are evaluated after recipe freezing. Labels never enter weight learning.
Neuron assignment uses its mean spikes per class; classification averages spikes
of assigned neurons per class. Samples without valid activity count as errors.
The frozen-feedforward control uses identical initialization, input spikes,
recurrence and threshold-update rules, with a separate fixed assignment.

.. code-block:: bash

    uv run python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --download --seed 0 \
        --output-dir logs/plasticity/conductance/seed-0
    uv run python -m spikingjelly.activation_based.examples.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/conductance/seed-0/selected.pt \
        --output-dir logs/plasticity/conductance/seed-0

Defaults are 700 steps, batch 32 and one epoch; spike probability ``pixel*0.064``,
base threshold 1, inhibitory increment 1, conductance time constant 5,
refractory period 5 and initial offset upper bound 1. Pre/post STDP trace time
constants are 20, LTD modulation is ``1e-4*weight``, LTP modulation is
``0.02*(1-weight)``, and threshold increment coefficient is 0.5. Updates divide
by actual batch size without time averaging. Initialization is positive uniform
random weights, with target row sum 78.4. Each update clamps to ``[0,1]``,
normalizes rows, then clamps to ``[0,1]`` again.

``--inhibition``, ``--inhibition-tau`` and ``--refractory-steps`` configure feedback;
``--v-threshold``, ``--theta-plus`` and ``--theta-initial`` configure thresholds.
The conductance time constant must be finite and positive; refractory steps are
a nonnegative integer, and the initial offset bound is finite and nonnegative.
``--inhibition 0`` disables feedback; zero refractory steps allow spikes on
adjacent steps. Use ``--epochs`` and ``--select-best`` to compare epochs using
validation only, breaking ties in favor of earlier epochs. Other seeds use the
same frozen epoch count. Small sample-count options are for smoke checks and
cannot prove full-test accuracy. ``--checkpoint`` restores saved dynamics,
encoding and assignments without training or reassignment. Former hard-winner,
negative-current or non-unit-step experimental checkpoints are unsupported;
use archived experiment source or retrain. New output goes to the separate
``logs/plasticity/conductance`` directory.

Validation status
----------------------

Recipe selection used validation only. Three full-training candidates were
``(theta_plus, ltp)=(0.2,0.01), (0.5,0.01), (0.5,0.02)``. Each completed two
epochs, then stopped during the third at its experiment budget; incomplete
states were excluded from selection. Their best validation accuracies were
85.82%, 87.00% and 87.20%; the last recipe's first-epoch checkpoint was selected.
Its mean receptive-field cosine similarity was about 0.744, with 134/400 neurons
assigned to digit 8, versus about 0.959 and 352/400 in earlier short training
with weak homeostasis. These diagnostics suggest better specialization but
cannot replace task acceptance. The full tests below used frozen models and class assignments, all 10,000
samples per seed, and no recipe selection or seed omission based on test results.

.. list-table:: Full MNIST tests with conductance competition
   :header-rows: 1

   * - Seed
     - STDP
     - Frozen feedforward control
     - Difference (percentage points)
   * - 0
     - 87.51%
     - 29.89%
     - +57.62
   * - 1
     - 86.66%
     - 29.56%
     - +57.10
   * - 2
     - 85.88%
     - 28.70%
     - +57.18

Mean accuracy is **86.68%**, with sample standard deviation
**0.82 percentage points**. The mean exceeds the 85% gate,
and every seed also exceeds 85%. Every test sample produced activity. The
environment was Python 3.11.14, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118;
formal tests ran on g2 A100-SXM4-80GB GPUs. These results support this shallow
recipe, not arbitrary deep-network training with pure STDP.

Repeat the commands with ``--seed`` 1 and 2 and separate output directories,
keeping the same default recipe and one epoch. Checkpoints save weights,
threshold offsets, class assignments and configuration; environment metadata
and metrics are stored in JSON. Independent evaluation does not require retraining. Recipe-search, failed-run
and acceptance ledgers are retained in experiment output directories. No silent
sample retries or automatic input-intensity increases were introduced.


References
------------

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
