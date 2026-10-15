STDP Learning
=======================================

Author: `fangwei123456 <https://github.com/fangwei123456>`_

中文版： :doc:`../cn/stdp`

STDP updates synaptic weights from neuronal activity. This tutorial covers traces,
gradient training and a MNIST example. For supported scope, time units, reward
alignment and interface migration, see :doc:`../../APIs/spikingjelly.activation_based.learning`.

Researchers of SNNs are always interested in biological learning rules. In SpikingJelly, STDP(Spike Timing Dependent Plasticity) \
is also provided and can be applied to convolutional or linear layers.

STDP(Spike Timing Dependent Plasticity)
-----------------------------------------------------

A classic experimental study of STDP(Spike Timing Dependent Plasticity) is [#STDP]_, which is a synaptic plasticity rule found in biological \
neural system. The experiments in the biological neural systems find that the weight of synapse is influenced by the firing time of spikes \
of the pre and post neuron. More specific, STDP can be formulated as:

If the pre neuron fires early and the post neuron fires later, then the weight will increase;
If the pre neuron fires later while the post neuron fires early, then the weight will decrease.

The curve [#STDP_figure]_ that fits the experiments data is as follows:

.. image:: ../../_static/tutorials/stdp/stdp.*
    :width: 100%

We can use the following equation to describe STDP:

.. math::

    \begin{align}
    \begin{split}
    \Delta w_{ij} =
    \begin{cases}
        A\exp(\frac{-|t_{i}-t_{j}|}{\tau_{+}}) , t_{i} \leq t_{j}, A > 0\\
        B\exp(\frac{-|t_{i}-t_{j}|}{\tau_{-}}) , t_{i} > t_{j}, B < 0
    \end{cases}
    \end{split}
    \end{align}

where :math:`A, B` are the maximum of weight variation, and :math:`\tau_{+}, \tau_{-}` are time constants.

However, the above equation is seldom used in practicals because it needs to record all firing times of pre and post neurons.\
The trace method [#Trace]_  is a more popular method to implement STDP.

For the pre neuron :math:`i` and the post neuron :math:`j`, we use the traces :math:`tr_{pre}[i]` and :math:`tr_{post}[j]` to track their firing. The update of \
traces are similar to the LIF neuron:

.. math::

    tr_{pre}[i][t] = tr_{pre}[i][t-1] -\frac{tr_{pre}[i][t-1]}{\tau_{pre}} + s[i][t]

    tr_{post}[j][t] = tr_{post}[j][t-1] -\frac{tr_{post}[j][t-1]}{\tau_{post}} + s[j][t]

where :math:`\tau_{pre}, \tau_{post}` are time constants of the pre and post neuron. :math:`s[i][t], s[j][t]` are the \
spikes at time-step :math:`t` of the pre neuron :math:`i` and the post neuron :math:`j`, which can only be 0 or 1.

The update of weight is:

.. math::

    \Delta W[i][j][t] = F_{post}(w[i][j][t]) \cdot tr_{pre}[i][t] \cdot s[j][t] - F_{pre}(w[i][j][t]) \cdot tr_{post}[j][t] \cdot s[i][t]

where :math:`F_{pre}, F_{post}` are functions that control how weight changes.

The implementation uses ``W[post, pre]``; the biological notation above uses
pre=i and post=j. STDP uses ``1-1/tau`` decay with finite ``tau>=1``. Current spikes
enter traces before computing updates. Batch, convolution positions and multi-step
time are summed; callers adjust ``scale`` explicitly. Simultaneous-spike cancellation
depends on weight modulation. Fixed-weight sequence accumulation is not stepwise learning.

STDP Learner
-----------------------------------------------------
:class:`spikingjelly.activation_based.learning.STDPLearner` can apply STDP learning on convolutional or linear layers. \
Please read the api doc first to learn how to use it.

Now let us use ``STDPLearner`` to build a ``4x3`` network, initializing weights to ``0.4``:

.. code-block:: python

    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, layer, learning
    from matplotlib import pyplot as plt

    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    torch.manual_seed(0)

    w_min, w_max = -1., 1.
    tau_pre, tau_post = 2., 2.
    N_in, N_out = 4, 3
    T = 128
    batch_size = 2
    lr = 0.01
    net = nn.Sequential(
        layer.Linear(N_in, N_out, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.4)

``STDPLearner`` adds ``-delta_w*scale`` to ``weight.grad`` for the optimizer to apply.
The update comes from activity correlations and differs from a task-loss gradient;
task performance needs separate evaluation. Plain SGD
without momentum or weight decay implements:

.. math::

    W = W - lr \cdot \nabla W = W + lr \cdot delta_w \cdot scale

.. code-block:: python

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

Create input spikes and the learner:

.. code-block:: python

    in_spike = (torch.rand([T, batch_size, N_in]) > 0.7).float()
    learner = learning.STDPLearner(step_mode='s', synapse=net[0], sn=net[1],
                                   tau_pre=tau_pre, tau_post=tau_post,
                                   f_pre=f_weight, f_post=f_weight)

Run each step and record trajectories. Slice batch/feature zero for plotting:

.. code-block:: python

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    with torch.no_grad():
        for t in range(T):
            optimizer.zero_grad()
            out_spike.append(net(in_spike[t]))
            learner.step(on_grad=True)
            optimizer.step()
            net[0].weight.data.clamp_(w_min, w_max)
            weight.append(net[0].weight.data.clone())
            trace_pre.append(learner.trace_pre)
            trace_post.append(learner.trace_post)

    out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    weight = torch.stack(weight)         # [T, N_out, N_in]

    in_spike = in_spike[:, 0, 0]
    out_spike = out_spike[:, 0, 0]
    trace_pre = trace_pre[:, 0, 0]
    trace_post = trace_post[:, 0, 0]
    weight = weight[:, 0, 0]

    learner.close()


The complete codes are available at ``spikingjelly/activation_based/examples/plasticity/stdp_trace.py``.

Let us plot ``in_spike, out_spike, trace_pre, trace_post, weight``:

.. image:: ../../_static/tutorials/stdp/stdp_trace.*
    :width: 100%

This figure is similar to Fig.3 in [#Trace]_  (note that they use `j` as the pre neuron and `i` as the post neuron, while we use the opposite symbol):

.. image:: ../../_static/tutorials/stdp/trace_paper_fig3.*
    :width: 100%


The random data below illustrate layer-wise updates without measuring classification performance.
Layer-wise separation clears task gradients on STDP layers after backward; additive
updates on the same parameter retain them. ``sn`` must produce actual post activity.
``step()`` consumes activity records and keeps traces; ``reset()`` clears learning
states and records. Explicitly disable recording for evaluation: ``net.eval()`` does
not stop it. Release hooks with ``close()`` when finished.

Combine STDP Learning with Gradient Descent
-----------------------------------------------------
A widely used method with STDP is using gradient descent and STDP to train different layers in an SNN. \
With ``STDPLearner``, we can combine STDP learning with gradient descent easily.

Our goal is to build a deep SNN, train convolutional layers with STDP, and train linear layers with gradient descent. \
First, let us define the hyper-parameters:


.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import SGD, Adam
    from spikingjelly.activation_based import learning, layer, neuron, functional

    T = 8
    N = 2
    C = 3
    H = 32
    W = 32
    lr = 0.1
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'

Here we use the input with ``shape = [T, N, C, H, W] = [8, 2, 3, 32, 32]``.

Then we define the weight function and the SNN. Here we build a convolutional SNN with a multi-step mode:

.. code-block:: python

    def f_weight(x):
        return torch.clamp(x, -1, 1.)


    net = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(16 * 8 * 8, 64, bias=False),
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    )

    functional.set_step_mode(net, step_mode)

We want to use STDP to train ``layer.Conv2d`` while other layers are to be trained with gradient descent. \
We use ``instances_stdp`` as the layers which are trained by STDP:

.. code-block:: python

    instances_stdp = (layer.Conv2d, )

We create an STDP learner for each layer in the SNN with the instance in ``instances_stdp``:

.. code-block:: python

    stdp_learners = []

    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight)
            )

Now we split parameters into two groups. The parameters from layers whose instances are in or not in ``instances_stdp`` \
will be set to two optimizers. Here we use ``Adam`` to optimize the parameters which are trained by gradient descent, and ``SGD`` \
to optimize the parameters which are trained by STDP:

.. code-block:: python

    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gradient_descent = []
    for p in net.parameters():
        if p not in params_stdp_set:
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)

When we train the SNN in actual tasks, e.g., classifying CIFAR-10, we get samples from the dataset. But here we only want to \
implement an example. Hence, we create the samples manually:

.. code-block:: python

    x_seq = (torch.rand([T, N, C, H, W]) > 0.5).float()
    target = torch.randint(low=0, high=10, size=[N])

Then we will use the two optimizers to update the parameters. Note that the following codes are different from the plain \
gradient descent we use before.


First, let us clear all gradients, do a forward, calculate the loss and do a backward:

.. code-block:: python

    optimizer_gd.zero_grad()
    optimizer_stdp.zero_grad()
    y = net(x_seq).mean(0)
    loss = F.cross_entropy(y, target)
    loss.backward()

Note that even though ``optimizer_gd`` will only update parameters in ``params_gradient_descent``, ``loss.backward()`` will \
calculate and set ``.grad`` to all parameters including those we want to calculate the weight variation (implemented by on ``.grad``) by STDP.

Thus, we need to clear the gradients of ``params_stdp``:

.. code-block:: python

    optimizer_stdp.zero_grad()


Then we need to use ``STDPLearner`` to get "gradients", and use two optimizers to update all parameters:

.. code-block:: python

    for i in range(stdp_learners.__len__()):
        stdp_learners[i].step(on_grad=True)

    optimizer_gd.step()
    optimizer_stdp.step()


All the learners ( ``STDPLearner`` , for instance) inherit from ``MemoryModule``. \
Hence, they have internal memories ( ``trace_pre, trace_post`` for ``STDPLearner`` ). \
In addition, the monitors inside the learners record the firing histories of the pre-synaptic and post-synaptic neurons; these histories \
may also be considered as internal memories of the learners. We should call the ``reset()`` method to clear the internal memory promptly \
to delimit independent episodes. ``step()`` already consumes activity records; unconsumed records can accumulate. We suggest resetting the learners together with the network after each batch:

.. code-block:: python

    functional.reset_net(net)
    for i in range(stdp_learners.__len__()):
        stdp_learners[i].reset()


The complete codes are as follows:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import SGD, Adam
    from spikingjelly.activation_based import learning, layer, neuron, functional

    T = 8
    N = 2
    C = 3
    H = 32
    W = 32
    lr = 0.1
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'

    def f_weight(x):
        return torch.clamp(x, -1, 1.)


    net = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(16 * 8 * 8, 64, bias=False),
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    )

    functional.set_step_mode(net, step_mode)

    instances_stdp = (layer.Conv2d, )

    stdp_learners = []

    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight)
            )


    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gradient_descent = []
    for p in net.parameters():
        if p not in params_stdp_set:
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)



    x_seq = (torch.rand([T, N, C, H, W]) > 0.5).float()
    target = torch.randint(low=0, high=10, size=[N])

    optimizer_gd.zero_grad()
    optimizer_stdp.zero_grad()

    y = net(x_seq).mean(0)
    loss = F.cross_entropy(y, target)
    loss.backward()



    optimizer_stdp.zero_grad()

    for i in range(stdp_learners.__len__()):
        stdp_learners[i].step(on_grad=True)

    optimizer_gd.step()
    optimizer_stdp.step()

    functional.reset_net(net)
    for i in range(stdp_learners.__len__()):
        stdp_learners[i].reset()

    for learner in stdp_learners:
        learner.close()

Plasticity mechanism demonstrations
-------------------------------------

These scripts show STDP traces, reward modulation and eligibility traces.
They run without a GUI and save to ``logs/plasticity`` by default;
use ``--show`` and ``--output-dir`` to change this.

.. code-block:: bash

    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_trace
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.mstdp
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.mstdpet

MSTDP/MSTDPET use ``exp(-1/tau)`` decay; ordinary STDP uses ``1-1/tau``.
Call reward step immediately after each forward. ``reward[t]`` modulates previous
eligibility; the first update is zero. An empty step does not advance time; reward
after final activity needs another forward. MSTDP accepts scalar or ``[M]`` reward;
MSTDPET accepts scalar reward. Remove ``batch_size`` from MSTDP construction.

Complete MNIST example
----------------------

``stdp_mnist`` uses a bias-free 784→3200 feedforward layer, identical current LIF neurons,
and ``ElementWiseRecurrentContainer``. Feedforward weights start randomly and use the
existing ``STDPLearner`` plus explicit weight constraints, without gradients, pretraining
or a trainable readout. Fixed negative recurrent currents provide soft competition.
Each neuron fires independently and resets only itself. Multiple neurons can fire in the same step.

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
can produce negative voltage. Integration has no voltage floor or conductance denominator.
Population-mean subtraction suppresses common drive within each sample. Recurrent
inhibition depends only on other neurons' previous-step spikes.

Every neuron has the same parameters; its history produces different adaptive states:

.. math::

    A_{nj}[t] &= \exp(-1/200)A_{nj}[t-1]+0.01s_{nj}[t],\\
    B_j[t] &= B_j[t-1]+\eta_B\left(\frac{120}{NT}
        -\frac{1}{M}\sum_{n=1}^{M}s_{nj}[t]\right).

A is fast own-spike adaptation current, reducing repeated responses. B is a slow signed
intrinsic excitability current, using unlabeled local rate error to adjust participation.
M is actual batch size, N neuron count, and T sample duration. B starts uniformly in
[-0.01,0] and freezes after training. This rate controller does not guarantee the target
while weights change; its biophysical accuracy and global stability have not been verified.
Positive B may cause spontaneous activity. Sparsity and mean rates are diagnostic
measures; accuracy measures task performance.

Input gain updates at the end of each training batch. C is the pixel sample covariance
of the fixed split-seed42 learning 50,000 images, without labels. With
``f_j=W_j-mean(W,dim=0)``:

.. math::

    \sigma_j^2=f_j C f_j^\top,\qquad
    g_j=\sqrt{\sigma_{j,\mathrm{initial}}^2/\sigma_j^2}.

This preserves each initial random projected-current variance scale, reducing operating
point drift from weight learning. Degenerate/nonfinite variance fails explicitly. C,
initial reference variance and gain are saved in the checkpoint and frozen in evaluation.
The gain constraint uses only unlabeled learning-set statistics, without task gradients
or test-set statistics.

Encoding, plasticity and readout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input uses ordinary ``SimpleIFNode(v_threshold=1, v_reset=None)`` with an independent
uniform initial phase in [0,1) per sample/pixel. Each step adds ``pixel*0.064`` and applies
own soft reset. Across 700 steps, pixel counts differ from expectation by at most about
one spike. Regular firing reduces count noise and changes temporal correlations,
so its learning trajectories differ from Poisson coding.

Pre/post STDP traces both use 20 steps and include the current events.
``f_pre=nu*weight`` and ``f_post=nu*(1-weight)``, with default nu=0.00229619779275505.
Each step returns ``step(on_grad=False, scale=1/actual_batch_size)`` and directly adds
its delta to weights, without averaging time. Weights are clipped to [0,1], normalized
to row sum 78.4, then clipped again. This example adds updates directly without
writing ``.grad``.

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

    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_mnist \
        --device cuda:0 --seed 0 --data-dir logs/plasticity/data --download \
        --output-dir logs/plasticity/current/seed-0
    uv run --no-sync python -m spikingjelly.activation_based.examples.plasticity.stdp_mnist \
        --device cuda:0 --checkpoint logs/plasticity/current/seed-0/selected.pt \
        --data-dir logs/plasticity/data --output-dir logs/plasticity/current/seed-0

Use the same defaults and one epoch with ``--seed`` 1/2 and separate directories.
``--checkpoint`` uses saved dynamics/mapping without reassignment or training;
replaying results does not require retraining. Small sample-count options check the
workflow; use the full test set when reporting performance.

Measured results
^^^^^^^^^^^^^^^^

Independent validation selected the recipe and one epoch: same-width Poisson full
training gave 88.76%; IF evaluation of that frozen model gave 89.16%; fresh IF short
training gave 86.72%/control 82.94%; fresh IF full training gave 89.58%/control 83.04%.
These four measurements come from separate training or evaluation stages. Seeds 1/2
then trained from scratch with the selected recipe and epoch count. All three models/maps
were fixed before official testing. Test scores did not select the recipe, and all seeds
are included in the report.

The frozen feedforward control receives identical initial weights and input spikes,
disabling only STDP weight updates. Its B still adapts during training, its fast
dynamics remain active, and its neuron classes are assigned independently afterwards.
The example targets a three-seed mean of at least 90%; individual seeds may fall below 90%.

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

Historical documentation record: `da6ff9a9 <https://github.com/fangwei123456/spikingjelly/commit/da6ff9a9>`_ (2026-10-12); V100S-PCIE-32GB; not rerun on current V2.

Mean accuracy is 90.2233%, population standard deviation is 0.2894 percentage points,
and mean paired improvement is 6.4667 percentage points. Every seed uses all 10,000 test
samples. One silent seed 0 sample is counted as an error. Environment: Python 3.11.14,
PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, V100S-PCIE-32GB testing.
These results apply to this shallow recipe. Arbitrary deep networks have not been
verified, and the model does not reproduce the original Diehl/Cook network. The experiment
retains raw JSON, configurations, checkpoints, source hashes and commands.

.. [#STDP] Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." Journal of neuroscience 18.24 (1998): 10464-10472.

.. [#STDP_figure] Froemke, Robert C., et al. "Contribution of individual spikes in burst-induced long-term synaptic modification." Journal of neurophysiology (2006).

.. [#Trace] Morrison, Abigail, Markus Diesmann, and Wulfram Gerstner. "Phenomenological models of synaptic plasticity based on spike timing." Biological cybernetics 98.6 (2008): 459-478.

* `Diehl and Cook (2015) <https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full>`_
* `Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
