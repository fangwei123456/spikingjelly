Timing-based SNNs with continuous events
========================================

``timing_based`` exposes explicit event times. TTFS constrains the code, while the
representation determines the implementation: a TTFS network that passes step-wise
binary activations belongs to ``activation_based``. The models here solve continuous
event responses without constructing a uniform time grid.

Start with the compact event format below. Then choose the section matching the
model you need:

* Tempotron: double-exponential PSP, one output spike, shunting and peak outputs.
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

Inputs are finite ``spike_times[E]``, int64 input-column ``input_indices[E]``, and
int64 CSR ``batch_offsets[B+1]``. Offsets start at 0, are nondecreasing and end at E.
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

   from spikingjelly.timing_based.encoding import GaussianTuning

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
backward uses implicit differentiation of the threshold equation, not bisection
branches. Peak values and times use local analytical derivatives. Shared-connection
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

   uv run python -m spikingjelly.timing_based.examples.tempotron_mnist \
       --device cpu --duration 100 --batch-size 64 --dataset-dir ./data

The example uses ``5*v_max/v_threshold`` as logits and adds the signed linear
peak-to-threshold deviation only for misfiring outputs, normalized by threshold.
The correct class should fire and other classes remain silent; suppressing false
premature spikes can restore late evidence. Prediction still uses
``v_max.argmax``. This task-level loss changes neither the neuron's threshold,
shunting nor internal autograd, and does not apply cross-entropy to ``t_spike``
values containing ``inf``.

The solver uses internal FP64 and eager execution. Bounded event blocks evaluate
states, stationary points and first-upcrossing candidates across the entire batch;
simultaneous inputs are aggregated once. Blocks contain up to 128 event groups on
CPU and 512 on CUDA. Local time-difference matrices replace tiny per-event
operations without a uniform time grid. Torch reference matrix memory scales with
batch size and squared block width: larger blocks trade memory for fewer launches.
Backward processes events across the batch, recomputing only required analytic
derivatives, with unchanged saved-tensor sizes. When Triton is installed, CUDA
automatically uses fused FP64 kernels for affine prefix-scan block states, threshold
roots and weight-only peak-voltage backward. The scan composes exact continuous
voltage/current transitions in chronological order and uses linear block storage,
avoiding the quadratic response matrices. It changes floating-point reduction
order, without introducing temporal sampling. Other analytic derivatives use
PyTorch. Without Triton, the same Torch reference runs on CUDA. Block-level synchronization remains;
this is not a fused full-model implementation or CUDA Graph execution. Historical
accuracy/speed do not describe the new dynamics.

The example defaults to a maximum of 40 epochs; use ``--epoch 100`` for an explicit
full reproduction. Two completed single-seed 100-epoch runs reached a test plateau
by 20–40 epochs and continued fitting training data afterward. The shorter cap is
an experiment budget, not a guarantee that every task converges by 40 epochs. Test
accuracy does not drive early stopping or checkpoint selection.

The new interface replaces integer T, ret_type, mse_loss, the old checkpoint
wrapper and GaussianTuning.encode. Use the interfaces shown above.

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
continues accepting inputs after firing. This differs from double-exponential
Tempotron dynamics and is not an equivalent acceleration. Spike k is the first
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

   uv run python -m spikingjelly.timing_based.examples.srm_mnist \
       --device cuda:0 --dataset-dir ./data --output-dir ./srm-mnist \
       --batch-size 128 --epochs 10 --seed 0

Use the complete MNIST dataset: a 55000/5000 train/validation split, validation-only
checkpoint selection and early stopping (default patience=3), followed by final
evaluation on all 10000 test examples. Save configuration, source fingerprints,
epoch metrics, ``best.pt`` and ``result.json`` with test correct counts. Use
``--device cpu`` for CPU. CUDA runs the same PyTorch mathematics without dedicated
SRM kernels; no general speed advantage is claimed.

Complete partitions and a third PSP model
-----------------------------------------

:func:`spikingjelly.timing_based.solver.first_spike_time` solves the earliest
upward zero crossing of a continuous total threshold residual. Its small
interface accepts a callable, complete monotone boundaries and explicit tensor
parameters. The callable receives FP32 query times and original piece indices,
and returns ``(F, partial_F/partial_t)``. Each leading index is an independent
time axis; the last dimension contains either all interval queries or one
selected query. Left/right local-extremum/plateau-end flags are separate for every interval,
so a knot can have different one-sided derivatives. The kernel must certify
window endpoint types too; horizontal inflections are not extrema.

A value/derivative callable alone cannot prove that no hidden peak crosses the
threshold. The caller must supply all stationary points of the weighted total
response, not just the union of individual PSP extrema. The solver trusts this
mathematical contract; format checks cannot certify a black-box partition.
It chooses the earliest valid interval, runs FP32 bracketed Newton/bisection, checks convergence
and residuals, and returns inf for silence. Grazing is excluded and an endpoint
upcrossing counts. Its first-order backward recomputes
``dt/dp=-F_p/F_t``; bisection is not differentiated. Differentiable previous output
times must be explicit parameters, so their chain rule is preserved. The solver
owns neither neuron history nor reset/shunting or event capacity. Higher-order
backward is unsupported; singular requested derivatives raise, without clipping.
Partitions select branches and receive no derivative.

With Triton installed, CUDA fuses general root updates and cubic polynomial
evaluation; otherwise it uses Torch. Partition ordering compares only the two
interior stationary points, preserving stable ties.

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

   uv run python -m spikingjelly.timing_based.examples.cubic_srm_mnist \
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
autograd still retains the tensors needed by every chunk. The measurements below
do not compare this implementation with cuDNN time-step convolution.

Validation on a Tesla V100S (Torch 2.7.1+cu118, Triton 3.3.1): seed 0, B128,
five full training epochs selected epoch 5, validation 92.24% and test **93.06%**
(9306/10000). Final routing/gradient-guard code replayed all 10000 logits exactly,
with identical finite B128 parameter gradients. This is a single-seed result.

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

The unfold reference is restricted to one event per pixel and uses the same
cubic solver and output packing; it is not a time-step model. The multi-event
CSR path is not faster on this dense workload. Isolated routing forward+backward
at B128 takes 2.14 ms (unfold 1.28 ms); this includes input-time gradients unlike the
image-input training step, so it is a scale comparison, not an exact subtraction
from total time. The existing partition/root operations dominate overall cost.
