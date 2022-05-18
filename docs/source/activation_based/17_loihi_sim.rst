Loihi仿真
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

LAVA-DL框架中Block的行为
-----------------------------------------------------

`lava.lib.dl.slayer.block` 封装突触和神经元到单个Block，可以通过如下流程使用Block来进行Loihi仿真：

1.使用Block导出hdf5定义的网络
2.加载网络，转换为LAVA中的Process
3.使用LAVA提供的Loihi仿真器仿真Process

Block是为Loihi仿真而生，它并不是像 `nn.Sequential` 这样简单的把两个模块包装一下，而是有更复杂的行为。

根据对源代码的分析，我们的结论是：

在 `slayer.block` 中：

- `p_scale = 1 << 12`

- `w_scale = scale`

- `s_scale = scale * (1 << 6)`

- 若不指定 `pre_hook_fx = None` 或其他特定的函数，则 `self.synapse.weight` 会被量化，然后限幅，最终取值范围是 `2k / w_scale, k = -128, -127, ..., 127`，共有256种取值

- `p_scale = 1 << 12, self.neuron.current_decay = int(p_scale * current_decay), self.neuron.voltage_decay = int(p_scale * voltage_decay)`，
  但在计算衰减时，衰减后的值会通过 `right_shift_to_zero(x, bits=12)` 还原

- `self.threshold = int(threshold * w_scale) / w_scale`

- 计算神经动态时， `x, self.current_state, self.voltage_state, self.threshold` 都会先乘上 `s_scale` 进行计算，最后的输出再除以 `s_scale` 进行还原


下面的内容是源代码的分析过程，不感兴趣的读者可以跳过。

以 `slayer.block.Dense` 为例，对其行为进行介绍。


`slayer.block.Dense` 的参数说明如下：

    - neuron_params (dict, optional) –- a dictionary of CUBA LIF neuron parameter. Defaults to None.

    - in_neurons (int) –- number of input neurons.

    - out_neurons (int) –- number of output neurons.

    - weight_scale (int, optional) –- weight initialization scaling. Defaults to 1.

    - weight_norm (bool, optional) –- flag to enable weight normalization. Defaults to False.

    - pre_hook_fx (optional) –- a function pointer or lambda that is applied to synaptic weights before synaptic operation. None means no transformation. Defaults to None.

    - delay (bool, optional) -– flag to enable axonal delay. Defaults to False.

    - delay_shift (bool, optional) –- flag to simulate spike propagation delay from one layer to next. Defaults to True.

    - mask (bool array, optional) -– boolean synapse mask that only enables relevant synapses. None means no masking is applied. Defaults to None.

    - count_log (bool, optional) -– flag to return event count log. If True, an additional value of average event rate is returned. Defaults to False.

`slayer.block.Dense` 前向传播的流程为：

`x` -> `synapse` -> `neuron`

突触的量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 `synapse` 的前向传播中，在进行计算前，会对自身的权重做一次变换：

.. code-block:: python

    # lava\lib\dl\slayer\synapse\layer.py
    class Dense(torch.torch.nn.Conv3d, GenericLayer):
        def forward(self, input):
            # ...
            if self._pre_hook_fx is None:
                weight = self.weight
            else:
                weight = self._pre_hook_fx(self.weight)
            # ...

根据 `slayer.block.Dense` 的构造函数：

.. code-block:: python

    # lava\lib\dl\slayer\block\cuba.py
    class Dense(AbstractCuba, base.AbstractDense):
        def __init__(self, *args, **kwargs):
            super(Dense, self).__init__(*args, **kwargs)
            self.synapse = synapse.Dense(**self.synapse_params)
            if 'pre_hook_fx' not in kwargs.keys():
                self.synapse.pre_hook_fx = self.neuron.quantize_8bit
            del self.synapse_params

可以发现，在不专门指定 'pre_hook_fx' 的情况下，`self.synapse.pre_hook_fx = self.neuron.quantize_8bit`。
因此，`slayer.block.Dense` 中的突触，默认是进行了量化。

我们查看量化函数的具体做法：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\base.py
    class Neuron(torch.nn.Module):
        def quantize_8bit(self, weight, descale=False):
            if descale is False:
                return quantize(
                    weight, step=2 / self.w_scale
                ).clamp(-256 / self.w_scale, 255 / self.w_scale)
            else:
                return quantize(
                    weight, step=2 / self.w_scale
                ).clamp(-256 / self.w_scale, 255 / self.w_scale) * self.w_scale

    # lava\lib\dl\slayer\utils\quantize.py
    class _quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, step=1):
            return torch.round(input / step) * step

        @staticmethod
        def backward(ctx, gradOutput):
            return gradOutput, None
    
    def quantize(input, step=1):
        return _quantize.apply(input, step)


在 `spikingjelly.clock_driven.lava_exchange.step_quantize <https://spikingjelly.readthedocs.io/zh_CN/latest/spikingjelly.clock_driven.lava_exchange.html#spikingjelly.clock_driven.lava_exchange.step_quantize>`_
中提供了一个量化函数的示意图：

.. image:: ../_static/API/clock_driven/lava_exchange/step_quantize.*
        :width: 100%

可以看出，`self.synapse.weight` 被进行 `step = 2 / self.neuron.w_scale` 的量化，然后再被限幅到 `[-256 / self.neuron.w_scale, 255 / self.neuron.w_scale]`。
因此，`self.synapse.weight` 量化后的取值范围为 `2k / self.neuron.w_scale, k = -128, -127, ..., 127`，共有256个取值，因而是8比特量化。


神经动态的量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 `neuron` 的前向传播中，首先进行神经动态（LAVA的重置过程被融合进了神经动态），然后进行放电：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    class Neuron(base.Neuron):
            def forward(self, input):
                _, voltage = self.dynamics(input)
                return self.spike(voltage)

神经动态主要包括电流和电压的计算。电流和电压的衰减系数分别是 `self.current_decay` 和 `self.voltage_decay`，它们在初始化时被缩放了一次：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    class Neuron(base.Neuron):
        def __init__(
        self, threshold, current_decay, voltage_decay,
        tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False,
        graded_spike=False
    ):
        super(Neuron, self).__init__(
            threshold=threshold,
            tau_grad=tau_grad,
            scale_grad=scale_grad,
            p_scale=1 << 12,
            w_scale=scale,
            s_scale=scale * (1 << 6),
            norm=norm,
            dropout=dropout,
            persistent_state=persistent_state,
            shared_param=shared_param,
            requires_grad=requires_grad
        )
        # ...
        self.register_parameter(
                'current_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * current_decay]),
                    requires_grad=self.requires_grad,
                )
            )
        self.register_parameter(
                'voltage_decay',
                torch.nn.Parameter(
                    torch.FloatTensor([self.p_scale * voltage_decay]),
                    requires_grad=self.requires_grad,
                )
            )
        # ...

因此，它们实际的值并不是在构造时给定的 `current_decay` 和 `voltage_decay`，而是乘上了 `self.p_scale`，也就是 `1 << 12`。

它们在神经动态中进行计算时，又被 `quantize` 函数量化了一次：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    class Neuron(base.Neuron):
        def dynamics(self, input):
            # ...
            # clamp the values only when learning is enabled
            # This means we don't need to clamp the values after gradient update.
            # It is done in runtime now. Might be slow, but overhead is negligible.
            if self.requires_grad is True:
                self.clamp()

            current = leaky_integrator.dynamics(
            input,
            quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
            debug=self.debug
            )

            voltage = leaky_integrator.dynamics(
            current,  # bias can be enabled by adding it here
            quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
            self.threshold,
            debug=self.debug
            )
            # ...

在训练时，每次前向传播前都会调用 `self.clamp()` 进行限幅：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    def clamp(self):
        """A function to clamp the sin decay and cosine decay parameters to be
        within valid range. The user will generally not need to call this
        function.
        """
        with torch.no_grad():
            self.current_decay.data.clamp_(0, self.p_scale)
            self.voltage_decay.data.clamp_(0, self.p_scale)



结合限幅和量化过程，我们可以得知，在进行神经动态计算电流和电压衰减时：

-- 真正的衰减系数是 `quantize(self.current_decay)` 和 `quantize(self.voltage_decay)`

-- 衰减系数的取值是量化的，取值范围为 `0, 1, 2, ..., self.p_scale`


接下来我们关注状态和阈值的量化。

收件根据构造函数，我们回顾一下几个系数之间的关系：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    class Neuron(base.Neuron):
        def __init__(
        self, threshold, current_decay, voltage_decay,
        tau_grad=1, scale_grad=1, scale=1 << 6,
        norm=None, dropout=None,
        shared_param=True, persistent_state=False, requires_grad=False,
        graded_spike=False
    ):
        super(Neuron, self).__init__(
        # ...
        p_scale=1 << 12,
        w_scale=scale,
        s_scale=scale * (1 << 6),
        # ...

根据 `base.Neuron` 的构造函数：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\base.py
    class Neuron(torch.nn.Module):
        def __init__(
        self, threshold,
        tau_grad=1, scale_grad=1,
        p_scale=1, w_scale=1, s_scale=1,
        norm=None, dropout=None,
        persistent_state=False, shared_param=True,
        requires_grad=True,
        complex=False
        ):
        # ...
        self.p_scale = p_scale
        self.w_scale = int(w_scale)
        self.s_scale = int(s_scale)
        # quantize to proper value
        self._threshold = int(threshold * self.w_scale) / self.w_scale
        # ...

可以发现阈值实际上是做了一个 `step = self.w_scale` 的量化。

最后，我们看一下 `self.s_scale` 在 `leaky_integrator.dynamics` 中的作用。查看源码：

.. code-block:: python

    # lava\lib\dl\slayer\neuron\cuba.py
    class Neuron(base.Neuron):
        def dynamics(self, input):
            # ...
            current = leaky_integrator.dynamics(
            input,
            quantize(self.current_decay),
            self.current_state.contiguous(),
            self.s_scale,
            debug=self.debug
            )

            voltage = leaky_integrator.dynamics(
            current,  # bias can be enabled by adding it here
            quantize(self.voltage_decay),
            self.voltage_state.contiguous(),
            self.s_scale,
            self.threshold,
            debug=self.debug
            )
            # ...

    # lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.py
    def _li_dynamics_fwd(
    input, decay, state, threshold, w_scale, dtype=torch.int32
    ):
        output_old = (state * w_scale).clone().detach().to(dtype).to(input.device)
        decay_int = (1 << 12) - decay.clone().detach().to(dtype).to(input.device)
        output = torch.zeros_like(input)

        threshold *= w_scale

        for n in range(input.shape[-1]):
            output_new = right_shift_to_zero(output_old * decay_int, 12) + \
                (w_scale * input[..., n]).to(dtype)
            if threshold > 0:
                spike_new = (output_new >= threshold)
                output_old = output_new * (spike_new < 0.5)
            else:
                output_old = output_new

            output[..., n] = output_new / w_scale

        return output

    # lava\lib\dl\slayer\utils\int_utils.py
    def right_shift_to_zero(x, bits):
        """Right shift with quantization towards zero implementation.

        Parameters
        ----------
        x : torch.int32 or torch.int64
            input tensor.
        bits : int
            number of bits to shift.

        Returns
        -------
        torch.int32 or torch.int64
            right shift to zero result.

        """
        # ...


可以发现，`input, state, threshold` 都会先乘上 `w_scale` 进行计算，最后再除以 `w_scale` 进行还原。`p_scale = 1 << 12`，因而 `right_shift_to_zero(x, bits=12)`。

最后的结论是，在 `slayer.block` 中：

- `p_scale = 1 << 12`

- `w_scale = scale`

- `s_scale = scale * (1 << 6)`

- 若不指定 `pre_hook_fx = None` 或其他特定的函数，则 `self.synapse.weight` 会被量化，然后限幅，最终取值范围是 `2k / w_scale, k = -128, -127, ..., 127`，共有256种取值

- `p_scale = 1 << 12, self.neuron.current_decay = int(p_scale * current_decay), self.neuron.voltage_decay = int(p_scale * voltage_decay)`，
  但在计算衰减时，最终的输出会通过 `right_shift_to_zero(x, bits=12)` 还原

- `self.threshold = int(threshold * w_scale) / w_scale`

- 计算神经动态时， `x, self.current_state, self.voltage_state, self.threshold` 都会先乘上 `s_scale` 进行计算，最后的输出再除以 `s_scale` 进行还原