Timing-based SNNs with continuous events
========================================

You will encode inputs as spike times and pass these events between neurons in
``timing_based``. The models solve responses in continuous time without a uniform
time grid. Time to first spike (TTFS) means that a value is represented by the
first spike time. TTFS networks that pass binary activations at discrete steps
use ``activation_based``.

Start by packing spike times into the compact event format, then choose a model:

* Tempotron: double-exponential postsynaptic potential (PSP), one output spike,
  shunting (blocking new inputs after firing) and peak outputs.
* SRM: linear-ramp PSP with reset, allowing multiple output spikes.
* CubicSRM: finite-support cubic PSP, one output spike and the general root solver.
* CubicConv2d: shared CubicSRM weights over 2D receptive fields.

All modules are stateless across calls. They expose ordinary PyTorch parameters;
losses, initialization constraints and recovery rules belong to the task. Event
creation, deletion and ordering boundaries are nonsmooth, so the documented
gradients apply while the selected event topology remains unchanged.

Response and outputs
--------------------

An input connection can carry multiple spikes sharing its synaptic weight. Resting
voltage is zero and the double-exponential PSP has unit peak:

.. math::

   V(t)=\sum_i w_i\sum_r h(t-t_{i,r}),\qquad
   h(u)=\frac{e^{-u/\tau}-e^{-u/\tau_s}}
               {e^{-u_p/\tau}-e^{-u_p/\tau_s}}\mathbf{1}_{u\ge0},

   u_p=\frac{\tau\tau_s\log(\tau/\tau_s)}{\tau-\tau_s}.

Two distinct positive time constants are supported, including swapped order and
near equality. Stable voltage/current states, ``expm1`` and ``log1p`` avoid
accumulating global positive exponentials.

The observation window is ``[0,duration]`` in the same units as all times and time
constants. Only the first upward threshold crossing fires; grazing does not.
Firing shunts simultaneous and later inputs while existing PSPs continue without
a hard reset. The peak may therefore follow the first spike.

:class:`spikingjelly.timing_based.neuron.Tempotron` returns three ``[B,O]`` tensors:

* ``v_max``: maximum voltage of the actual shunted trajectory.
* ``t_peak``: earliest time of that maximum.
* ``t_spike``: first upward crossing, or ``inf`` for non-firing neurons.

Use ``torch.isfinite(t_spike)`` to identify firing. A grazing peak can reach the
threshold without firing. Empty trials yield ``(0,0,inf)``. No state persists
between calls; ``reset()`` is unnecessary.

Compact event inputs
--------------------

Pack E events from B samples into three tensors: finite ``spike_times[E]``,
int64 input-column ``input_indices[E]``, and int64 ``batch_offsets[B+1]``.
This compressed sparse row (CSR) format uses offsets to delimit each sample. Offsets start at 0, are nondecreasing and end at E.
Segments may be unordered; simultaneous events are grouped and repeated offsets
represent empty trials. Out-of-window events are ignored with zero gradient.
All tensors share the weights' device; times and weights share FP32/FP64 dtype
on CPU/CUDA.

.. code-block:: python

   import torch
   from spikingjelly.timing_based.neuron import Tempotron

   neuron = Tempotron(2, 1, duration=5.0, tau=2.0, tau_s=0.5).double()
   with torch.no_grad():
       neuron.weight.copy_(torch.tensor([[1.0, 0.5]], dtype=torch.float64))
   times = torch.tensor([0.3, 0.4, 3.0], dtype=torch.float64,
                        requires_grad=True)
   indices = torch.tensor([0, 0, 1], dtype=torch.int64)
   offsets = torch.tensor([0, 3, 3], dtype=torch.int64)
   v_max, t_peak, t_spike = neuron(times, indices, offsets)
   assert v_max.shape == t_peak.shape == t_spike.shape == (2, 1)
   assert torch.isinf(t_spike[1]).all()
   target = torch.tensor([[1.5], [0.0]], dtype=torch.float64)
   loss = (v_max - target).square().mean()
   loss.backward()
   assert torch.isfinite(times.grad).all()

``offsets=[0,3,3]`` packs two samples: three events in the first and none in
the second. All three outputs have shape ``[2,1]``; the empty sample stays silent,
so its ``t_spike`` is ``inf``.

The stateless operator is :func:`spikingjelly.timing_based.functional.tempotron`.
The module owns only ``weight[O,I]`` and hyperparameters. Weights are initialized
from ``N(0,0.001²)`` with checkpoint key ``weight``. PSP normalization is independent
of the threshold.

Continuous Gaussian encoding
----------------------------

:class:`spikingjelly.timing_based.encoding.GaussianTuning` is an ``nn.Module`` with
fixed receptive-field buffers ``mu`` and ``sigma2``. Inputs have shape
``[B,n,samples_count]``; outputs are the three compact event tensors.

.. code-block:: python

   import torch
   from spikingjelly.timing_based.encoding import GaussianTuning
   from spikingjelly.timing_based.neuron import Tempotron

   encoder = GaussianTuning(1, 4, torch.zeros(1), torch.ones(1)).double()
   x = torch.tensor([[[0.2, 0.7]]], dtype=torch.float64, requires_grad=True)
   events = encoder(x, duration=5.0)
   net = Tempotron(1 * 2 * 4, 2, duration=5.0).double()
   logits, _, _ = net(*events)
   torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()

Times are ``duration * (1-response)`` without rounding. Input columns flatten
feature, sample and receptive-field axes. Only ``0 <= time < duration`` is emitted,
without ``-1`` or ``inf`` placeholders. Selected times differentiate to input
values; event-selection boundaries need not have unique derivatives.

Autograd and losses
-------------------

Only first-order weight and input-time derivatives are supported. First-spike
backward uses implicit differentiation of the threshold equation. Peak values and times use local analytical derivatives. Shared-connection
and event contributions accumulate. Equal peaks select the earliest branch;
event-boundary time derivatives select the first original event in stable order.
Fixed-window peak times and non-firing ``t_spike`` branches have zero gradient.

Nonsmooth points need not have unique derivatives. Singular time derivatives are
checked only when the loss uses the corresponding output. No clipping, surrogate
or silent-neuron recovery is hidden. Double backward is explicitly unsupported.

Callers choose the loss. The snippets demonstrate first-order backward through
peak squared error or cross-entropy. The MNIST example combines calibrated
cross-entropy with a linear misfiring penalty, addressing the mismatch between
raw peak logits and shunting.
A task using spike times must handle ``inf`` correctly. All-negative responses or
empty inputs can select the resting platform with zero gradient. Initialization,
regularization and recovery are task-level decisions.

MNIST example and performance scope
-----------------------------------

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.tempotron_mnist \
       --device cpu --duration 100 --batch-size 64 --dataset-dir ./data

The example uses cross-entropy on ``5*v_max/v_threshold`` plus a linear threshold penalty for misfiring outputs, and classifies with ``v_max.argmax``.

The Tempotron solver uses FP64 internally. CUDA operators select an available
native CUDA, Triton or PyTorch implementation, in that order by default.
Installing Triton alone does not guarantee its selection; device compatibility
and operator availability also matter.

Train for 40 epochs by default, or set ``--epoch 100`` for 100 epochs.
The script reports training and test accuracy each epoch and saves the final model
as ``tempotron_snn_mnist.ckpt`` in ``--model-output-dir``. It runs the requested
number of epochs without early stopping or test-based checkpoint selection.

Reference
---------

Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based
decisions. Nature Neuroscience 9, 420–428 (2006).
`Paper <https://www.nature.com/articles/nn1643>`_.

Multi-spike SRM and two-layer MNIST
------------------------------------------

:class:`spikingjelly.timing_based.neuron.SRM` is an experimental fixed-kernel model:

.. math::

   u_j(t)=\sum_i w_{ji}\sum_e (t-t_{ie})_+
          -\vartheta\sum_{f:s_{jf}<t}1.

The input response is a linear-ramp PSP; each spike adds a step reset response,
resetting voltage to zero while retaining current. With no leak or absolute
refractory period, the model accepts multiple inputs, emits multiple outputs and
continues accepting inputs after firing. Its ramp response gives different dynamics from the double-exponential Tempotron. Spike k is the first
upward crossing of raw level ``k*threshold``. Sorted prefix sums, cumulative endpoint
maxima and analytic roots solve FP64 times without a uniform grid. PyTorch
differentiates the selected time/weight branch; event creation/deletion and ordering
boundaries need not have unique derivatives.

Inputs and outputs are compact CSR ``(times,indices,offsets)``, directly composable.
Outputs are ordered by trial, neuron index and spike number; the next layer sorts
them. Empty trials emit nothing, upward crossings at the endpoint count, and
outputs precede simultaneous inputs. Per-trial, per-output ``max_spikes`` defaults
to 64; overflow raises instead of truncating. Standard ``nn.Module.to`` and
``state_dict`` apply.

.. code-block:: python

   import torch
   from spikingjelly.timing_based.neuron import SRM

   first = SRM(2, 3, duration=2.0).double()
   second = SRM(3, 1, duration=2.0).double()
   with torch.no_grad():
       first.weight.fill_(1.0)
       second.weight.fill_(0.5)
   times = torch.tensor([0.1, 0.4], dtype=torch.float64, requires_grad=True)
   events = first(times, torch.tensor([0, 1]), torch.tensor([0, 2]))
   output_times, _, _ = second(*events)
   output_times.sum().backward()

The MNIST example has two trainable SRM layers, 784→128→10. Pixels become
``1-pixel`` times and the observation endpoint is 2. Hidden neurons can emit
multiple spikes, which directly enter layer two. Each output event contributes
``2-time`` to its class score; logits are five times the accumulated contribution
and the loss is cross-entropy. Adam (default lr=0.01) trains raw parameters;
the demo's PyTorch parametrization applies ``gain*softmax(weight)`` with fixed row
gains 8 and 2, maintaining firing and bounding event counts. The constraints and
readout belong to the example. SRM allows negative weights and contains no loss,
surrogate or recovery rule.

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./srm-mnist \
       --batch-size 128 --epochs 10 --seed 0

Use the complete MNIST dataset: a 55000/5000 train/validation split, validation-only
checkpoint selection and early stopping (default patience=3), followed by final
evaluation on all 10000 test examples. Save configuration, source fingerprints,
epoch metrics, ``best.pt`` and ``result.json`` with test correct counts. Use
``--device cpu`` for CPU. CUDA runs the same PyTorch mathematics without dedicated SRM kernels.

Complete partitions and a third PSP model
-----------------------------------------

:func:`spikingjelly.timing_based.solver.first_spike_time` finds the earliest upward crossing. The caller must provide all stationary points of the weighted total response; the solver does not verify partition completeness.

.. code-block:: python

   import torch
   from spikingjelly.timing_based.solver import first_spike_time

   def response(time, piece, arrival, weight):
       return weight * (time - arrival) - 1, weight.expand_as(time)

   arrival = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
   weight = torch.tensor([[2.0]], dtype=torch.float32, requires_grad=True)
   # The residual is linear on this entire interval: the partition is complete.
   root = first_spike_time(response, torch.tensor([[0., 2.]], dtype=torch.float32),
                           (arrival, weight))
   root.sum().backward()  # root=0.6, arrival.grad=1, weight.grad=-0.25

:class:`spikingjelly.timing_based.neuron.CubicSRM` provides a third concrete model:

.. math::

   h(u)=\begin{cases}\frac{27}{4}z(1-z)^2,&0<z=u/L<1,\\0,&\text{otherwise},\end{cases}
   \qquad V_j(t)=\sum_e w_{j,i_e}h(t-t_e).

Each output emits at most once, at its earliest upcrossing of the threshold.
There is no output-history kernel or reset. Repeated inputs and signed weights
are supported. On intervals delimited by input arrivals and support endpoints,
the total response is cubic. Analytic quadratic derivative roots yield the
complete monotone partition, including multiple peaks caused by shifted inputs.
Local voltage/derivative prefix transitions avoid global powers of event times;
simultaneous updates are grouped while preserving individual time gradients.
The shared solver supplies numerical roots and implicit gradients. No uniform time grid is used; CUDA optionally
uses fused Triton kernels.
Tempotron retains its double-exponential solver, shunting and peak outputs;
ramp SRM retains its analytic multi-level scan and step reset. The general solver
does not replace these optimized paths.

.. code-block:: bash

   python -m spikingjelly.timing_based.examples.cubic_srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./cubic-srm-mnist \
       --hidden-features 64 --batch-size 128 --epochs 10 --seed 0

The example uses 784→hidden→10 CubicSRM layers, pixel times ``1-pixel``, window 4,
support 2 and threshold 1. Differentiable ``gain*softmax(weight)`` row constraints (hidden/output gains 8/2)
belong to the example. Earlier class spikes score higher: logits are
``50*(4-time)``; silence scores zero. Train with cross-entropy and Adam, using the
same validation-only checkpoint/early-stopping workflow as the ramp SRM example.
The complete 55000/5000/10000 split and source hashes are recorded. Default hidden
width is 128 and batch size 32; the command above uses width 64 and batch 128.
Stationary overshoots smaller than the solver's documented FP32 roundoff bound
are indistinguishable from grazing. Extreme support ratios remain limited by
floating-point resolution. No surrogate or silent-neuron recovery is built in.

The general solver and CubicSRM accept only FP32 floating tensors, with no
precision selector or FP64 path. Rebase/rescale extreme absolute times or support
ratios when intervals cannot be resolved.

Continuous-event spatial convolution
-------------------------------------

:class:`spikingjelly.timing_based.neuron.CubicConv2d` shares a cubic PSP kernel over
2D receptive fields. It accepts the same three CSR event tensors as CubicSRM,
plus an explicit ``(H,W)`` input size. Input indices flatten ``(Cin,H,W)``;
outputs flatten ``(Cout,Hout,Wout)`` and remain grouped into the original B trials.
Padding has no events. Multiple events per input location are retained, and both
spatial axes use integer stride/padding and a square kernel. There is no bias,
groups, dilation, reset history or cross-call state.

The implementation routes events into local trials, folds positions into batch,
and calls the existing FP32 cubic solver and optional Triton arithmetic. All
positions/channels are processed in tensors. Weight gradients accumulate across
positions, and overlapping fields accumulate gradients to their original input
times. The earliest-upcrossing, tangent, silence and first-order-only contracts
are inherited from CubicSRM; spatial convolution does not introduce time sampling.

CSR routing and unfold
~~~~~~~~~~~~~~~~~~~~~~

The public convolution accepts CSR events because one spatial position may be
silent or carry multiple spikes. Routing starts from each event, finds every
receptive field that covers its position, and records the corresponding local
``(channel, kernel_y, kernel_x)`` weight column. Output channels share this routed
event list. Each receptive field becomes a CubicSRM trial, and standard autograd
sums gradients into shared weights and input events used by overlapping fields.

For a dense single-spike time map, ``torch.nn.functional.unfold`` can extract
fixed-size time patches more cheaply. That representation needs an explicit mask
for silence, padded event slots for multiple spikes, and separate handling so that
zero padding does not become a time-zero event. Patch extraction does not change
the PSP, partition, or root solver. The public implementation accepts CSR events;
the MNIST benchmark uses unfold only as an equivalent reference because every
pixel emits once.

.. code-block:: python

    import torch
    from spikingjelly.timing_based.neuron import CubicConv2d

    layer = CubicConv2d(1, 2, 3, duration=4.0, stride=2, padding=1)
    with torch.no_grad():
        layer.weight.fill_(1.2)
    times = torch.tensor([0.1, 0.3])
    indices = torch.tensor([0, 0])  # repeated events at the same input pixel
    offsets = torch.tensor([0, 2])
    output_times, output_indices, output_offsets = layer(
        times, indices, offsets, (4, 4)
    )

The minimal MNIST network is ``1×28×28 → 8C5/stride2 → 8×12×12 → 1152→10``,
with two spiking layers and 11720 parameters. Pixels become ``1-pixel`` times;
all pixels emit, including the black background at time 1. Weights use the demo's
row-softmax gains 8/2, logits ``50*(4-time)`` (zero for silence), cross-entropy
and Adam. The shared trainer uses a 55000/5000/10000 split, validation-selected
checkpoints and one final test; ``--hidden-features`` means convolution channels.

.. code-block:: bash

    python -m spikingjelly.timing_based.examples.cubic_conv_mnist \
        --device cuda:0 --batch-size 128 --epochs 5 --hidden-features 8 \
        --dataset-dir ./data --output-dir ./cubic-conv-mnist

Convolution reduces local fan-in and adds output neurons, event routing, and sorting.
Temporary coefficients grow with positions, channels, and local event counts.
Performance comparisons must use the same network, events, and weights, and must
include backward and peak memory. Chunking limits some construction intermediates;
autograd still retains the tensors needed by every chunk. The following timings compare two event-routing methods for this network.

Reference results (commit `7341418a <https://github.com/fangwei123456/spikingjelly/commit/7341418a>`_, Tesla V100S):

With Torch 2.7.1+cu118 and Triton 3.3.1, seed 0 and batch size 128, validation
selected epoch 5 of five training epochs: validation accuracy was 92.24% and test
accuracy was 93.06% (9306/10000). These are single-seed results.

Three independent processes compared the same trained network, checkpoint,
MNIST first batch and cross-entropy forward+backward, excluding optimizer.
Three warmups and 10 alternating repeats per process gave the following medians
of process medians (peak memory is the maximum observed across processes):

.. list-table:: Equivalent routing comparison on dense single-spike inputs
   :header-rows: 1

   * - Batch
     - Multi-event CSR route (ms)
     - Single-spike unfold reference (ms)
     - CSR peak memory (MiB)
   * - 1
     - 40.90
     - 38.46
     - 10.10
   * - 8
     - 44.79
     - 42.83
     - 87.83
   * - 32
     - 53.37
     - 52.77
     - 361.71
   * - 128
     - 104.59
     - 104.00
     - 1583.22

The unfold reference processes one event per pixel with the same cubic solver
and output packing. The two routes have similar total latency on this dense workload. Isolated routing forward+backward
at B128 takes 2.14 ms (unfold 1.28 ms); this includes input-time gradients unlike the
image-input training step, so it is a scale comparison, not an exact subtraction
from total time. The existing partition/root operations dominate overall cost.
