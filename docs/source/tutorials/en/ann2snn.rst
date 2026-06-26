ANN2SNN
=======================================

Author: `DingJianhao <https://github.com/DingJianhao>`_, `fangwei123456 <https://github.com/fangwei123456>`_, `Lv Liuzhenghao <https://github.com/Lyu6PosHao>`_, `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版：:doc:`../cn/ann2snn`

.. admonition:: ANN2SNN tutorial versions

    The ANN2SNN public API has gone through three tutorial generations:

    #. :doc:`Older clock-driven-era ANN2SNN API <../../legacy_tutorials/en/5_ann2snn>`.
    #. :doc:`Legacy pre-Recipe Converter API <../../legacy_tutorials/en/ann2snn_converter_legacy>`, which used ``Converter(mode=..., dataloader=...)`` and ``convert_to_spiking_neurons(model)``.
    #. Current Recipe API, documented on this page: ``RateCodingRecipe`` or ``TransformerSpikeEquivalentRecipe`` defines the algorithm, and ``Converter.convert(model)`` executes it.

This tutorial focuses on ``spikingjelly.activation_based.ann2snn``. It shows how to convert a trained feedforward ANN to an SNN with the current Recipe API and simulate the converted model in SpikingJelly.

ANN2SNN API references are available `here <https://spikingjelly.readthedocs.io/zh_CN/latest/spikingjelly.activation_based.ann2snn.html>`_.

The current implementation is based on ``torch.fx``. ``torch.fx`` traces ``nn.Module`` instances into a graph representation, which is then transformed by ANN2SNN recipes.

Theoretical basis of ANN2SNN
----------------------------

SNNs communicate with discrete spikes, which enables efficient event-driven computation, but direct SNN training often requires more resources than ANN training. A practical route is to train an ANN first and then convert it to an SNN with similar behavior. This requires connecting ANN activations with SNN firing rates. For rate-coded SNNs, output classes are read from spike counts. The core question is whether the firing rate of a spiking neuron can approximate the activation of an ANN neuron.

ReLU activations in ANNs are strongly related to the firing rates of IF neurons with subtractive reset, where the membrane voltage is reset by subtracting :math:`V_{threshold}`. ``RateCodingRecipe`` uses this relationship for conversion. This neuron update method is the Soft reset method described in the `Neuron tutorial <https://spikingjelly.readthedocs.io/zh_CN/latest/tutorials/en/neuron.html>`_.

Experiment: Relationship between IF neuron spiking frequency and input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Give constant input to an IF neuron and observe its output spikes and firing rate. First import the relevant modules, create an IF neuron layer, and plot the input :math:`x_{i}` for each neuron:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import neuron
    from spikingjelly import visualizing
    from matplotlib import pyplot as plt
    import numpy as np

    plt.rcParams['figure.dpi'] = 200
    if_node = neuron.IFNode(v_reset=None)
    T = 128
    x = torch.arange(-0.2, 1.2, 0.04)
    plt.scatter(torch.arange(x.shape[0]), x)
    plt.title('Input $x_{i}$ to IF neurons')
    plt.xlabel('Neuron index $i$')
    plt.ylabel('Input $x_{i}$')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/0.*
    :width: 100%

Send the input to the IF neuron layer for ``T=128`` steps and observe the output spikes and firing rates:

.. code-block:: python

    s_list = []
    for t in range(T):
        s_list.append(if_node(x).unsqueeze(0))

    out_spikes = np.asarray(torch.cat(s_list))
    visualizing.plot_1d_spikes(out_spikes, 'IF neurons\' spikes and firing rates', 't', 'Neuron index $i$')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/1.*
    :width: 100%

Within a certain range, the firing frequency is proportional to the input :math:`x_{i}`.

Plot the firing rate against the input :math:`x_{i}` and compare it with :math:`\mathrm{ReLU}(x_{i})`:

.. code-block:: python

    plt.subplot(1, 2, 1)
    firing_rate = np.mean(out_spikes, axis=0)
    plt.plot(x, firing_rate)
    plt.title('Input $x_{i}$ and firing rate')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('Firing rate')
    plt.grid(linestyle='-.')

    plt.subplot(1, 2, 2)
    plt.plot(x, x.relu())
    plt.title('Input $x_{i}$ and ReLU($x_{i}$)')
    plt.xlabel('Input $x_{i}$')
    plt.ylabel('ReLU($x_{i}$)')
    plt.grid(linestyle='-.')
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/2.*
    :width: 100%

The two curves are nearly identical. However, the firing rate cannot exceed 1, so the IF neuron cannot approximate ReLU activations larger than 1.

Theoretical basis of ANN2SNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature [#f1]_ provides an analytical basis for ANN-to-SNN conversion. The theory shows that an IF neuron in an SNN is an unbiased estimator of the ReLU activation over time.

Consider the first layer. Let the firing rate of SNN neurons be :math:`r` and the corresponding ANN activation be :math:`a`. Assume constant input :math:`z \in [0,1]`.
For an IF neuron with subtractive reset, the membrane potential evolves as:

.. math::
	V_t=V_{t-1}+z-V_{threshold}\theta_t

where :math:`V_{threshold}` is the firing threshold (usually 1.0) and :math:`\theta_t` is the output spike. The average firing rate over :math:`T` steps can be derived by summing the membrane potential:

.. math::
	\sum_{t=1}^{T} V_t= \sum_{t=1}^{T} V_{t-1}+z T-V_{threshold} \sum_{t=1}^{T}\theta_t

Rearranging the :math:`V_t` terms and dividing by :math:`T`:

.. math::
	\frac{V_T-V_0}{T} = z - V_{threshold}  \frac{\sum_{t=1}^{T}\theta_t}{T} = z- V_{threshold}  \frac{N}{T}

where :math:`N` is the spike count in the time window :math:`T`, and :math:`\frac{N}{T}` is the firing rate :math:`r`. Substituting :math:`z = V_{threshold} a`:

.. math::
	r = a- \frac{ V_T-V_0 }{T V_{threshold}}

Therefore, when the simulation time step :math:`T` is infinite:

.. math::
	r = a (a>0)

For higher layers, [#f1]_ shows that the inter-layer firing rate satisfies:

.. math::
	r^l = W^l r^{l-1}+b^l- \frac{V^l_T}{T V_{threshold}}

See [#f1]_ for the full derivation. The methods in ann2snn are based on [#f1]_.

Converting to spiking neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversion mainly solves two problems:

1. ANNs use Batch Normalization for faster training and convergence. Batch normalization normalizes activations to zero mean, which conflicts with SNN properties. The BN parameters can be absorbed into the preceding parameter layers (Linear, Conv2d).

2. According to the conversion theory, each layer's inputs and outputs must lie within [0, 1], which requires scaling the parameters (model normalization)

◆ BatchNorm parameter absorption

Assume that the parameters of BatchNorm are: :math:`\gamma` (``BatchNorm.weight``), :math:`\beta` (``BatchNorm.bias``), :math:`\mu` (``BatchNorm.running_mean``) ,
:math:`\sigma` (``BatchNorm.running_var``, :math:`\sigma = \sqrt{\mathrm{running\_var}}`). For specific parameter definitions, see
`torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d>`_ .
A parameter module (e.g. Linear) has weight :math:`W` and bias :math:`b`. BatchNorm absorption folds the BatchNorm parameters into :math:`W` and :math:`b` so the new module produces the same output as the original module-plus-BatchNorm pair. The resulting :math:`\bar{W}` and :math:`\bar{b}` are:

.. math::
    \bar{W} = \frac{\gamma}{\sigma} W

.. math::
    \bar{b} = \frac{\gamma}{\sigma} (b - \mu) + \beta

◆ Model Normalization

For a parameter module with input tensor maximum :math:`\lambda_{pre}` and output tensor maximum :math:`\lambda`, the normalized weight :math:`\hat{W}` is:

.. math::
     \hat{W} = W * \frac{\lambda_{pre}}{\lambda}

The normalized bias :math:`\hat{b}` is:

.. math::
     \hat{b} = \frac{b}{\lambda}

Layer activations often contain large outliers that suppress the overall firing rate. Robust normalization replaces the tensor maximum with the p-quantile as the scaling factor. The recommended quantile is 99.9% [#f1]_.

BatchNorm fusion and model normalization are algebraic transformations before
spiking replacement. The rate-coding recipe then replaces ReLU activations with
IF neurons.
For average pooling in ANN, the converted model keeps spatial downsampling.
Because IF neurons approximate ReLU activations over time, adding another IF
neuron immediately after spatial downsampling usually has little effect on the
result.
There is no general max-pooling conversion rule in the current rate-coding
recipe. The literature uses a gating function based on momentum-accumulated
spikes to control pulse channels [#f1]_. This tutorial therefore recommends
``AvgPool2d`` for the example model.
During simulation, the converted SNN should receive a constant analog input
under this conversion theory. A Poisson encoder can introduce additional
accuracy loss.

Implementation and optional configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current ann2snn API separates **what algorithm to run** from **how to run
the conversion**:

* A recipe owns algorithm-specific options and graph transformations.
* ``Converter`` is the executor. It receives a recipe, traces the model with
  ``torch.fx``, and calls the recipe steps through ``convert(model)``.

For ReLU-to-IFNode rate-coding conversion, use ``RateCodingRecipe``. It needs a
calibration dataloader because it measures activation ranges before replacing
ReLU modules. The common normalization modes are:

* ``mode="max"``: MaxNorm, which uses maximum activation values [#f2]_.
* ``mode="99.9%"``: RobustNorm, which uses the 99.9% activation quantile [#f1]_.
* ``mode`` as a float in ``(0, 1]``: scale the maximum activation by this value.

``RateCodingRecipe`` also owns options such as ``fuse_flag``. When
``fuse_flag=True`` (the default), Conv-BatchNorm pairs are fused before
calibration.

The minimal rate-coding call is:

.. code-block:: python

    recipe = ann2snn.RateCodingRecipe(dataloader=train_loader, mode="max")
    snn = ann2snn.Converter(recipe=recipe).convert(ann)

After conversion, ReLU modules are removed. New modules needed by the SNN, such
as ``VoltageScaler`` and ``IFNode``, are created as ``spiking_*`` submodules
under the original parent module. Since the converted model is an
``fx.GraphModule``, you can use ``snn.graph.print_tabular()`` to inspect the
generated computation graph. More APIs are documented in `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_ .

.. note::

    Current versions require ``Converter(recipe=...)`` and
    ``Converter.convert(model)``. Older public algorithm methods, including
    ``convert_to_spiking_neurons()``, ``replace_by_td_operators()``, ``fuse()``,
    ``set_voltagehook()``, ``replace_by_neurons()``, and ``replace_by_ifnode()``,
    have been removed.


Classify MNIST
--------------

Build and load the ANN
^^^^^^^^^^^^^^^^^^^^^^

This section builds a simple convolutional network with ``ann2snn`` to classify the MNIST dataset.

The complete runnable example is ``spikingjelly.activation_based.ann2snn.examples.cnn_mnist``.
The network structure is defined in ``ann2snn.sample_models.mnist_cnn``:

.. code-block:: python

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(32, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            x = self.network(x)
            return x

Note: when the tensor needs to be flattened, define an ``nn.Flatten`` module in the network and call it in ``forward`` instead of using ``view``.

Set the runtime options:

.. code-block:: python

    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = "./data/mnist"
    batch_size = 100
    T = 50

``T`` is the number of SNN simulation steps used during inference. Create the MNIST dataloaders before conversion:

.. code-block:: python

    train_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    calibration_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset, batch_size=50, shuffle=True, drop_last=False
    )

The example script downloads a pretrained checkpoint. Load it and validate the ANN first:

.. code-block:: python

    ann2snn.download_url(
        "https://ndownloader.figshare.com/files/34960191",
        "./SJ-mnist-cnn_model-sample.pth",
    )
    model = mnist_cnn.CNN().to(device)
    model.load_state_dict(torch.load("SJ-mnist-cnn_model-sample.pth", map_location=device))
    acc = val(model, device, test_data_loader)
    print('ANN Validating Accuracy: %.4f' % (acc))

The ANN accuracy is:

.. code-block:: shell

    ANN Validating Accuracy: 0.9870


Make the conversion with the converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ANN is trained and validated. Select the rate-coding recipe, pass the
deterministic calibration dataloader, and run ``Converter`` to execute the
conversion:

.. code-block:: python

    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="max")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)

``snn_model`` is the converted SNN model. View its structure below. The
``BatchNorm2d`` modules disappear because the default rate-coding recipe fuses
BatchNorm parameters into the preceding Conv layers before calibration:

.. code-block:: python

    CNN(
      (network): Module(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (8): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        (11): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (12): Flatten(start_dim=1, end_dim=-1)
        (13): Linear(in_features=32, out_features=10, bias=True)
        (spiking_0): Module(
          (scaler0): VoltageScaler(0.193247)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(5.174733)
        )
        (spiking_1): Module(
          (scaler0): VoltageScaler(0.325697)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(3.070336)
        )
        (spiking_2): Module(
          (scaler0): VoltageScaler(0.121967)
          (if_node): IFNode(
            v_threshold=1.0, v_reset=None, detach_reset=False, step_mode=s, backend=torch
            (surrogate_function): Sigmoid(alpha=4.0, spiking=True)
          )
          (scaler1): VoltageScaler(8.198915)
        )
      )
    )

``snn_model`` is an ``fx.GraphModule``; see `GraphModule <https://pytorch.org/docs/stable/fx.html?highlight=graphmodule#torch.fx.GraphModule>`_.

Call ``GraphModule.graph.print_tabular()`` to inspect the computation graph in tabular form:

.. code-block:: shell

    # snn_model.graph.print_tabular()
    opcode       name                       target                     args                          kwargs
    -----------  -------------------------  -------------------------  ----------------------------  ------
    placeholder  x                          x                          ()                            {}
    call_module  network_0                  network.0                  (x,)                          {}
    call_module  network_spiking_0_scaler0  network.spiking_0.scaler0  (network_0,)                  {}
    call_module  network_spiking_0_if_node  network.spiking_0.if_node  (network_spiking_0_scaler0,)  {}
    call_module  network_spiking_0_scaler1  network.spiking_0.scaler1  (network_spiking_0_if_node,)  {}
    call_module  network_3                  network.3                  (network_spiking_0_scaler1,)  {}
    call_module  network_4                  network.4                  (network_3,)                  {}
    call_module  network_spiking_1_scaler0  network.spiking_1.scaler0  (network_4,)                  {}
    call_module  network_spiking_1_if_node  network.spiking_1.if_node  (network_spiking_1_scaler0,)  {}
    call_module  network_spiking_1_scaler1  network.spiking_1.scaler1  (network_spiking_1_if_node,)  {}
    call_module  network_7                  network.7                  (network_spiking_1_scaler1,)  {}
    call_module  network_8                  network.8                  (network_7,)                  {}
    call_module  network_spiking_2_scaler0  network.spiking_2.scaler0  (network_8,)                  {}
    call_module  network_spiking_2_if_node  network.spiking_2.if_node  (network_spiking_2_scaler0,)  {}
    call_module  network_spiking_2_scaler1  network.spiking_2.scaler1  (network_spiking_2_if_node,)  {}
    call_module  network_11                 network.11                 (network_spiking_2_scaler1,)  {}
    call_module  network_12                 network.12                 (network_11,)                 {}
    call_module  network_13                 network.13                 (network_12,)                 {}
    output       output                     output                     (network_13,)                 {}

Other recipes and custom recipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MNIST example above uses rate coding because it converts ReLU activations
to IF neurons. Transformer models can use
``TransformerSpikeEquivalentRecipe`` instead:

.. code-block:: python

    recipe = ann2snn.TransformerSpikeEquivalentRecipe()
    td_model = ann2snn.Converter(recipe=recipe).convert(transformer_ann)

This recipe does not need a dataloader, does not insert ``VoltageHook``, and
does not run rate-coding calibration. It replaces currently supported ANN
modules and attention calls with TD / spike-equivalent operators. It does not
claim fully spike-driven LLM conversion.

To add a new conversion algorithm, subclass ``ConversionRecipe`` and override
only the steps you need. Methods that are not overridden use the base no-op
implementation. A recipe is not an executor and should not provide
``convert()``, ``run()``, or ``__call__()``:

.. code-block:: python

    class MyRecipe(ann2snn.ConversionRecipe):
        def replace(self, converter, fx_model):
            # Implement the algorithm-specific graph rewrite here.
            return fx_model

The fixed step order is ``validate`` -> ``before_trace`` -> ``after_trace`` -> ``insert_observers`` -> ``calibrate`` -> ``replace`` -> ``finalize``.

Customizing conversion rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``RateCodingRecipe`` uses ``ReLURule`` to replace ``nn.ReLU``
modules with ``VoltageScaler(1 / s) -> IFNode -> VoltageScaler(s)``. The
calibration scale ``s`` is computed from ``VoltageHook`` by
``ThresholdOptimizer``.

Advanced users can pass explicit extension points to ``RateCodingRecipe``:

* ``rules`` match FX graph nodes, insert calibration hooks, find calibrated
  activation-hook pairs, and replace them with an SNN subgraph.
* ``NeuronFactory`` creates the neuron used during replacement. The default
  factory creates ``IFNode(v_threshold=1.0, v_reset=None)``.
* ``ThresholdOptimizer`` computes a finite positive scalar threshold from a
  calibrated ``VoltageHook``.

The following minimal rule shows the protocol shape without adding a new
conversion algorithm. It matches ``nn.Identity`` and replaces the calibrated
identity node with another ``nn.Identity`` module:

.. code-block:: python

    import torch
    import torch.nn as nn

    from spikingjelly.activation_based import ann2snn
    from spikingjelly.activation_based.ann2snn.modules import VoltageHook


    class IdentityRule:
        def match(self, node, modules):
            return node.op == "call_module" and type(modules[node.target]) is nn.Identity

        def insert_hooks(self, fx_model, node, hook_factory, hook_counts_per_prefix):
            target = f'{node.target}_voltage_hook'
            fx_model.add_submodule(target, hook_factory.create())
            with fx_model.graph.inserting_after(node):
                return fx_model.graph.call_module(target, args=(node,))

        def find_replacements(self, fx_model, modules):
            for hook_node in fx_model.graph.nodes:
                if hook_node.op == "call_module" and isinstance(
                    modules.get(hook_node.target), VoltageHook
                ):
                    yield hook_node.args[0], hook_node

        def replace_with_neurons(
            self, fx_model, activation_node, hook_node, neuron_factory, threshold_optimizer
        ):
            hook = fx_model.get_submodule(hook_node.target)
            threshold = threshold_optimizer.compute_threshold(hook)
            # IdentityRule does not use a spiking threshold, but real rules
            # should pass this scalar into their replacement subgraph.
            target = f'{activation_node.target}_spiking_identity'
            fx_model.add_submodule(target, nn.Identity())
            with fx_model.graph.inserting_after(hook_node):
                new_node = fx_model.graph.call_module(target, args=activation_node.args)
            hook_node.replace_all_uses_with(new_node)
            activation_node.replace_all_uses_with(new_node)
            fx_model.graph.erase_node(hook_node)
            fx_model.graph.erase_node(activation_node)


    recipe = ann2snn.RateCodingRecipe(
        dataloader=[torch.randn(2, 4)],
        rules=[IdentityRule()],
    )
    converter = ann2snn.Converter(recipe=recipe)

The built-in ``ReLURule`` path currently has defined semantics only for scalar
thresholds. Per-channel or tensor thresholds need explicit shape, broadcasting,
and ``VoltageScaler`` semantics, and are not part of this conversion path yet.

Comparison of different converting modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the same model, convert with each mode (``max``, ``99.9%``, ``1.0/2``, ``1.0/3``, ``1.0/4``, ``1.0/5``) and run inference for T steps to compare accuracy.

.. code-block:: python

    print('---------------------------------------------')
    print('Converting using MaxNorm')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="max")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_max_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))

    print('---------------------------------------------')
    print('Converting using RobustNorm')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="99.9%")
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_robust_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_robust_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/2 max(activation) as scales...')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 2)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_two_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_two_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/3 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 3)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_three_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_three_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/4 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 4)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_four_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_four_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/5 max(activation) as scales')
    recipe = ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 5)
    model_converter = ann2snn.Converter(recipe=recipe)
    snn_model = model_converter.convert(model)
    print('Simulating...')
    mode_five_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_five_accs[-1]))

The following output was measured with ``T=50`` on an NVIDIA GeForce RTX 4090:

.. code-block:: shell

    ANN Validating Accuracy: 0.9870
    ---------------------------------------------
    Converting using MaxNorm
    Calibration: 3.76s
    Simulating...
    Simulation: 7.61s
    SNN accuracy (simulation 50 time-steps): 0.9771
    ---------------------------------------------
    Converting using RobustNorm
    Calibration: 12.08s
    Simulating...
    Simulation: 7.31s
    SNN accuracy (simulation 50 time-steps): 0.9848
    ---------------------------------------------
    Converting using 1/2 max(activation) as scales
    Calibration: 3.73s
    Simulating...
    Simulation: 7.28s
    SNN accuracy (simulation 50 time-steps): 0.9846
    ---------------------------------------------
    Converting using 1/3 max(activation) as scales
    Calibration: 3.72s
    Simulating...
    Simulation: 7.29s
    SNN accuracy (simulation 50 time-steps): 0.9825
    ---------------------------------------------
    Converting using 1/4 max(activation) as scales
    Calibration: 3.75s
    Simulating...
    Simulation: 7.27s
    SNN accuracy (simulation 50 time-steps): 0.9734
    ---------------------------------------------
    Converting using 1/5 max(activation) as scales
    Calibration: 3.70s
    Simulating...
    Simulation: 7.26s
    SNN accuracy (simulation 50 time-steps): 0.9472

RobustNorm and moderate scaling usually converge faster than MaxNorm in early
SNN timesteps on this example. Based on the time-varying accuracy of the model
output, we can plot the accuracy for different settings.

.. code-block:: python

    fig = plt.figure()
    plt.plot(np.arange(0, T), mode_max_accs, label="mode: max")
    plt.plot(np.arange(0, T), mode_robust_accs, label="mode: 99.9%")
    plt.plot(np.arange(0, T), mode_two_accs, label="mode: 1.0/2")
    plt.plot(np.arange(0, T), mode_three_accs, label="mode: 1.0/3")
    plt.plot(np.arange(0, T), mode_four_accs, label="mode: 1.0/4")
    plt.plot(np.arange(0, T), mode_five_accs, label="mode: 1.0/5")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Acc")
    plt.show()

.. image:: ../../_static/tutorials/5_ann2snn/accuracy_mode_recipe_api.png

Different settings trade off early-timestep convergence and final accuracy.
Users can choose the normalization mode according to latency and accuracy
requirements.

.. [#f1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
.. [#f2] Diehl, Peter U. , et al. Fast classifying, high-accuracy spiking deep networks through weight and threshold balancing. Neural Networks (IJCNN), 2015 International Joint Conference on IEEE, 2015.

Additional references:

* Rueckauer, B., Lungu, I. A., Hu, Y., & Pfeiffer, M. (2016). Theory and tools for the conversion of analog to spiking convolutional neural networks. arXiv preprint arXiv:1612.04052.
* Sengupta, A., Ye, Y., Wang, R., Liu, C., & Roy, K. (2019). Going deeper in spiking neural networks: Vgg and residual architectures. Frontiers in neuroscience, 13, 95.
