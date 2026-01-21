Triton backend and FlexSN
==============================

Author: `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版： :doc:`../cn/triton_flexsn`

SpikingJelly version ``0.0.0.1.0`` introduces the `Triton <https://github.com/triton-lang/triton>`_ backend as an alternative to PyTorch and CuPy. Compared with the CuPy backend, the Triton backend offers better readability, extensibility, and maintainability, makes it easier to achieve higher GPU utilization, and has the potential to be applied to `other hardware platforms <https://gitcode.com/Ascend/triton-ascend>`_.

Users can use the Triton backend of SpikingJelly in two ways:

* Predefined neuron kernels: set ``backend="triton"`` when constructing a neuron.
* ``FlexSN``: generate Triton kernels based on a user-defined PyTorch neuron dynamical function. This module is more flexible and powerful than the former ``auto_cuda`` module.

This tutorial will introduce the usage of the Triton backend from these two perspectives. The following preparations and prerequisites are required:

#. `Install Triton <https://triton-lang.org/main/getting-started/installation.html>`_. It is recommended to use ``triton >= 3.3.1``.
#. Be familiar with the SpikingJelly :doc:`./neuron` module.

Using Predefined Triton Kernels
++++++++++++++++++++++++++++++++++++++++++++++

Forward and Backward Propagation
---------------------------------------

The way to enable the Triton backend is similar to that of the CuPy backend. Taking ``LIFNode`` as an example:

.. code:: python

    import torch
    from spikingjelly.activation_based import neuron

    n = neuron.LIFNode(step_mode="m", backend="triton").to("cuda:0")
    x = torch.randn([16, 1, 3, 32, 32], device="cuda:0") # [T, B, C, H, W]

    s = n(x)
    print(s.device, s.shape, s.mean())
    # cuda:0 torch.Size([16, 1, 3, 32, 32]) tensor(0.0313, device='cuda:0')

Here, we construct an LIF neuron running in multi-step mode ``step_mode="m"`` and enable the Triton backend. After moving both the neuron and the input tensor to the ``"cuda:0"`` device, forward propagation can be performed. The Triton backend also supports backward propagation and produces (almost) identical results to other backends:

.. code:: python

    import torch
    import torch.nn.functional as F
    from spikingjelly.activation_based import neuron

    n_triton = neuron.LIFNode(
        step_mode="m", backend="triton", store_v_seq=True
    ).to("cuda:0")
    n_torch = neuron.LIFNode(
        step_mode="m", backend="torch", store_v_seq=True
    ).to("cuda:0")

    x = torch.randn([16, 1, 3, 32, 32], device="cuda:0") # [T, B, C, H, W]
    x_triton = x.clone().requires_grad_(True)
    x_torch = x.clone().requires_grad_(True)

    s_triton = n_triton(x_triton)
    s_torch = n_torch(x_torch)
    v_triton = n_triton.v_seq
    v_torch = n_torch.v_seq

    grad = torch.randn_like(s_triton)
    s_triton.backward(grad)
    s_torch.backward(grad)

    assert torch.allclose(s_triton, s_torch)
    print(s_triton.mean()) # tensor(0.0309, device='cuda:0', grad_fn=<MeanBackward0>)
    assert torch.allclose(v_triton, v_torch)
    print(v_triton.mean()) # tensor(-0.0702, device='cuda:0', grad_fn=<MeanBackward0>)
    assert torch.allclose(x_triton.grad, x_torch.grad, rtol=1e-6, atol=1e-6)
    print(
        F.cosine_similarity(x_triton.grad.flatten(), x_torch.grad.flatten(), dim=0)
    ) # tensor(1., device='cuda:0')

Performance Benchmark
------------------------

The Triton backend supports ``torch.float16``. Below, we use the performance benchmarking tools provided by Triton, namely ``triton.testing``, to compare the efficiency of different backends:

.. code:: python

    import torch
    import triton
    from spikingjelly.activation_based import neuron, functional

    DEVICE = "cuda:0"

    def forward_backward(net, x_seq):
        y_seq  = net(x_seq)
        y_seq.sum().backward()
        x_seq.grad = None
        functional.reset_net(net)


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["T"],
            x_vals=[4*i for i in range(1, 9)],
            line_arg="backend",
            line_vals=["torch", "cupy", "triton"],
            line_names=["torch", "cupy", "triton"],
            styles=[
                ('green', ':'), ('blue', '--'), ('red', '-.'),
            ],
            ylabel='Execution Time (ms)',
            plot_name='Performance-float16',
            args={"N": 64, "C": 64*32*32, 'dtype': torch.float16},
        )
    )
    def benchmark(T, N, C, dtype, backend):
        net = neuron.LIFNode(
            backend=backend,
            step_mode='m',
        ).to(device=DEVICE, dtype=dtype)
        x_seq = torch.rand(
            [T, N, C], device=DEVICE, dtype=dtype, requires_grad=True
        )
        results = triton.testing.do_bench(
            lambda: forward_backward(net, x_seq),
            quantiles=[0.5, 0.2, 0.8],
            grad_to_none=[x_seq]
        )
        return results

    benchmark.run(save_path="./logs", print_data=True, show_plots=True)

When running on a single GeForce RTX 4090, the results are as follows:

.. code:: text

    Performance-float16:
        T      torch      cupy    triton
    0   4.0   0.992784  0.667648  0.771072
    1   8.0   3.459072  1.338368  0.857088
    2  12.0   7.058432  1.988608  1.289216
    3  16.0  11.737088  2.630736  1.704896
    4  20.0  17.557505  3.263488  2.115584
    5  24.0  24.517120  3.902464  2.533376
    6  28.0  32.649216  4.535296  2.940928
    7  32.0  41.872896  5.189120  3.365888

.. image:: ../../_static/tutorials/triton_flexsn/Performance-float16.png
    :width: 100%

It can be observed that when both the data scale is large, the Triton backend exhibits a clear speed advantage over the CuPy and PyTorch backends.

.. admonition:: Warning
    :class: warning

    When using predefined Triton neuron kernels, please note the following:

    * Currently, only the most commonly used ``IFNode``, ``LIFNode``, and ``PLIFNode`` are equipped with Triton kernels. More Triton kernels will be added in future updates.
    * The Triton backend should be executed on a GPU.
    * The Triton backend only supports multi-step mode ``step_mode="m"``.

Using FlexSN to Customize Triton Neuron Kernels
+++++++++++++++++++++++++++++++++++++++++++++++++++

FlexSN can automatically generate high-performance multi-step Triton kernels from a user-defined **single-step neuronal dynamics function**, providing extremely high flexibility. With FlexSN, users can easily improve the efficiency of their customized neuron models.

Describing Neuronal Dynamics with Functions
----------------------------------------------

The discrete-time dynamics of most spiking neuron models can be described as:

.. math::

    Y_1[t], Y_2[t], \dots, V_1[t], V_2[t], \dots = f_s\left( X_1[t], X_2[t], \dots, V_1[t-1], V_2[t-1], \dots \right)

where :math:`Y_i` denotes outputs, :math:`V_i` denotes state variables, and :math:`X_i` denotes inputs. This equation can be described using a PyTorch function:

.. code:: python

    def single_step_inference(x1, x2, ..., v1, v2, ...):
        ...
        return y1, y2, ..., v1_updated, v2_updated, ...

Here, ``x1, x2, ...`` represent inputs, ``v1, v2, ...`` represent state variables, ``y1, y2, ...`` represent outputs, and ``v1_updated, v2_updated, ...`` represent the updated state variables (corresponding to ``v1, v2, ...``). For example, a soft-reset LIF neuron with non-decaying input (``tau=2``, ``v_th=1.0``, sigmoid surrogate function) can be described as:

.. code:: python

    from spikingjelly.activation_based import surrogate

    tau = 2.0 # time constant
    v_th = 1.0 # threshold
    spike_fn = surrogate.Sigmoid()

    def lif_single_step_inference(x, v):
        h = (1 - 1/tau) * v + x
        s = spike_fn(h - v_th)
        v = h - s * v_th
        return s, v

In this example, there is only one input, one output, and one state variable. For more complex neuron
models, however, there may be multiple inputs, outputs, and state variables. In addition, the model
hyperparameters ``tau``, ``v_th``, and ``spike_fn`` here are fixed global variables. To flexibly configure
hyperparameters, **function closures** can be used:

.. code:: python

    from spikingjelly.activation_based import surrogate

    def lif_single_step_inference_closure(tau=2., v_th=1., spike_fn=surrogate.Sigmoid()):
        def lif_single_step_inference(x, v):
            h = (1 - 1/tau) * v + x
            s = spike_fn(h - v_th)
            v = h - s * v_th
            return s, v
        return lif_single_step_inference

    f = lif_single_step_inference_closure(tau=99., v_th=0.5)

FlexSN Workflow
----------------------

Take the following customized spiking neuron as an example:

.. code:: python

    import torch
    from spikingjelly.activation_based import surrogate

    def complicated_lif_core_generator(beta: float, gamma: float, spike_fn=surrogate.ATan()):
        def complicated_lif_core(
    	    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, rho: torch.Tensor
        ):
            h = beta*v + x
            s1 = spike_fn(h - (rho+1.)) # spike, with threshold adaptation
            s2 = spike_fn(h - 1.)       # spike, without threshold adaptation
            rho = gamma*rho + s1        # adaptation variable update
            v1 = h * (1.-s1)            # hard reset
            v2 = h - s2                 # soft reset
            yy = torch.sigmoid(y)       # modulation factor
            v = v1*yy + v2 * (1.-yy)    # modulated reset
            return s1, s2, v, rho
        return complicated_lif_core

This model has two inputs ``x, y``, two outputs ``s1, s2``, and two state variables ``v, rho``. The dependency relationships among different variables are illustrated in the figure below:

.. image:: ../../_static/tutorials/triton_flexsn/neuron.png
    :width: 100%

To generate a multi-step Triton kernel, use :class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>` :

.. code:: python

    from spikingjelly.activation_based import neuron

    f = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="triton",
        store_state_seqs=True,
    )

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    s1, s2 = f(x, y)
    v, rho = f.state_seqs

    print(s1.mean()) # tensor(0.0821, device='cuda:0', grad_fn=<MeanBackward0>)
    print(s2.mean()) # tensor(0.1494, device='cuda:0', grad_fn=<MeanBackward0>)
    print(v.mean()) # tensor(-0.2750, device='cuda:0', grad_fn=<MeanBackward0>)
    print(rho.mean()) # tensor(0.4842, device='cuda:0', grad_fn=<MeanBackward0>)

The construction of :class:`FlexSN <spikingjelly.activation_based.neuron.flexsn.FlexSN>` requires the following arguments:

* ``core`` : a function that describes the single-step neuron dynamics, with the signature ``[*inputs, *states] -> [*outputs, *states]``.
* ``num_inputs, num_states, num_outputs`` : the numbers of inputs, state variables, and outputs, which should be consistent with the signature of ``core``.
* ``example_inputs`` : example arguments for ``core``. ``FlexSN`` will call ``core`` with these example inputs in order to capture the computation graph.
* ``requires_grad`` : whether the arguments of ``core`` require gradients. The default value is ``None``, which means that all arguments require gradients (i.e., equivalent to all ``True``).
* ``step_mode, backend`` : similar to other neuron modules, these two arguments determine the step mode and the backend. The ``triton`` backend is only valid when ``step_mode="m"``.
* ``store_state_seqs`` : similar to ``store_v_seq`` in other neuron modules, this argument determines whether state sequences are stored. If ``True``, the state sequences from the last run can be accessed via the ``state_seqs`` attribute. This attribute is a list, where each element corresponds to the sequence of a specific state variable.

``FlexSN`` also supports backward propagation, as shown in the following code block:

.. code:: python

    n_triton = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="triton",
        store_state_seqs=True,
    )

    n_torch = neuron.FlexSN(
        core=complicated_lif_core_generator(beta=0.5, gamma=0.9),
        num_inputs=2,
        num_states=2,
        num_outputs=2,
        example_inputs=(
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
            torch.zeros([1], device="cuda"), torch.zeros([1], device="cuda"),
        ),
        requires_grad=(True, True, True, True),
        step_mode="m",
        backend="torch",
        store_state_seqs=True,
    )

    x = torch.randn([16, 3, 32, 32], device="cuda")
    y = torch.randn([16, 3, 32, 32], device="cuda")
    x_triton = x.clone().requires_grad_(True)
    y_triton = y.clone().requires_grad_(True)
    x_torch = x.clone().requires_grad_(True)
    y_torch = y.clone().requires_grad_(True)

    s1_triton, s2_triton = n_triton(x_triton, y_triton)
    s1_torch, s2_torch = n_torch(x_torch, y_torch)
    grad = torch.randn_like(s1_triton)
    s1_triton.backward(grad)
    s1_torch.backward(grad)

    v_triton, rho_triton = n_triton.state_seqs
    v_torch, rho_torch = n_torch.state_seqs

    assert torch.allclose(s1_triton, s1_torch)
    assert torch.allclose(s2_triton, s2_torch)
    assert torch.allclose(x_triton.grad, x_torch.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y_triton.grad, y_torch.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_triton, v_torch, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rho_triton, rho_torch)
    print(s1_triton.mean())
    print(s2_triton.mean())
    print(x_triton.grad.mean())
    print(y_triton.grad.mean())
    print(v_triton.mean())
    print(rho_triton.mean())

All ``assert`` statements pass, and the outputs are shown below. This demonstrates that the Triton kernels
generated by ``FlexSN`` are equivalent to the original PyTorch function in both forward and backward
propagation.

.. code:: text

    tensor(0.0821, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.1494, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.0007, device='cuda:0')
    tensor(6.2995e-05, device='cuda:0')
    tensor(-0.2750, device='cuda:0', grad_fn=<MeanBackward0>)
    tensor(0.4842, device='cuda:0', grad_fn=<MeanBackward0>)

With the workflow described above, users can obtain Triton-accelerated neuron models with very little code. Compared with the former ``auto_cuda`` module (see :doc:`./cupy_neuron`), ``FlexSN`` is more flexible and
general.

.. admonition:: Note
    :class: note

    In the example above, the state variables ``v`` and ``rho`` use the default **zero initialization**.
    Users can override the ``init_states()`` method to change the state initialization rule. The original
    definition of this method is shown below, where ``*args`` represents the arguments of ``forward()``:

    .. code:: python

        class FlexSN(base.MemoryModule):

            ...

            @staticmethod
            def init_states(num_states: int, step_mode: str, *args) -> List[torch.Tensor]:
                if step_mode == "s":
                    return [torch.zeros_like(args[0]) for _ in range(num_states)]
                elif step_mode == "m":
                    return [torch.zeros_like(args[0][0]) for _ in range(num_states)]
                else:
                    raise ValueError(f"Unsupported step mode: {step_mode}")

    See :class:`FlexSN.init_states <spikingjelly.activation_based.neuron.flexsn.FlexSN.init_states>` for details.


.. admonition:: Warning
    :class: warning

    When using ``FlexSN``, please note the following:

    * It should be executed on a GPU.
    * ``FlexSN`` is not compatible with ``torch.jit``. Please set the environment variable ``PYTORCH_JIT=0`` before running your script.
    * The Triton backend only supports multi-step mode ``step_mode="m"``.
    * The PyTorch backend is implemented by repeatedly calling ``core``.
    * In the design of ``FlexSN``, compromises are made in efficiency in order to pursue generality. At present, ``IFNode``, ``LIFNode``, and ``PLIFNode`` are equipped with highly optimized predefined Triton kernels. Please use these predefined kernels instead of ``FlexSN`` whenever possible to obtain higher performance.
    * After completing a simulation trail with ``FlexSN``, ``reset()`` must be called to reset the neuron states.