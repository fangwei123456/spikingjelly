Continuous-event Tempotron
==========================

``timing_based`` exposes explicit event times. TTFS is a coding constraint, not
an implementation category: TTFS networks passing step-wise binary activations
still belong to ``activation_based``. This Tempotron solves input-event intervals
without constructing a uniform time grid.

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

This release uses eager PyTorch and internal FP64. Bounded event blocks evaluate
states, stationary points and first-upcrossing candidates across the entire batch;
simultaneous inputs are aggregated once. Blocks contain up to 128 event groups on
CPU and 512 on CUDA. Local time-difference matrices replace tiny per-event
operations without a uniform time grid. Temporary matrix memory scales with batch
size and squared block width: larger blocks trade memory for fewer launches.
Backward processes events across the batch, recomputing only required analytic
derivatives, with unchanged saved-tensor sizes. CUDA still has block-level synchronization and no
specialized GPU kernel. Full MNIST training requires its own evaluation; historical
accuracy/speed do not describe the new dynamics.

The new interface replaces integer T, ret_type, mse_loss, the old checkpoint
wrapper and GaussianTuning.encode. Use the interfaces shown above.

Reference
---------

Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based
decisions. Nature Neuroscience 9, 420–428 (2006).
`Paper <https://www.nature.com/articles/nn1643>`_.
