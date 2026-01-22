from typing import Callable, Optional, Tuple, List
import os
import logging

import torch

from .. import base

try:
    from .. import triton_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    triton_kernel = None


__all__ = ["FlexSNKernel", "FlexSN"]


class FlexSNKernel:
    def __init__(
        self, core: Callable, num_inputs: int, num_states: int, num_outputs: int,
        example_inputs: Optional[Tuple[torch.Tensor]] = None,
        requires_grad: Optional[Tuple[bool]] = None,
    ):
        """
        **API Language:**
        :ref:`中文 <FlexSNKernel.__init__-cn>` | :ref:`English <FlexSNKernel.__init__-en>`

        ----

        .. _FlexSNKernel.__init__-cn:

        * **中文**

        ``FlexSNKernel`` 可以根据自定义的 PyTorch 单步函数 ``core`` 生成 Triton 多步脉冲神经元核。
        不同于 :class:`FlexSN` ， ``FlexSNKernel`` 只是对 autograd function
        :class:`FlexSNFunction <spikingjelly.activation_based.triton_kernel.flexsn.wrapper.FlexSNFunction>`
        的简单封装，实例化后得到一个可调用对象 ( ``Callable`` object) 。

        实例化后， ``FlexSNKernel`` 对象接受的输入参数为 ``[*input_seqs, *states]`` ，其中 ``input_seqs`` 是
        ``num_inputs`` 个输入序列，``states`` 是 ``num_states`` 个初始状态；返回值为 ``[*output_seqs, *state_seqs]`` ，
        其中 ``output_seqs`` 是 ``num_outputs`` 个输出序列，``state_seqs`` 是 ``num_states`` 个状态序列。

        .. admonition:: 警告
            :class: warning

            使用 ``FlexSNKernel`` 需要禁用 ``torch.jit``。请在运行脚本前设置环境变量 ``PYTORCH_JIT=0``。
            参见: https://docs.pytorch.org/docs/2.8/jit.html#disable-jit-for-debugging 。

        阅读 :class:`FlexSN` 文档以获取参数的详细信息。

        ----

        .. _FlexSNKernel.__init__-en:

        * **English**

        ``FlexSNKernel`` can generate Triton multi-step spiking neuron kernels from
        a customized PyTorch single-step function ``core``. It is a simple ``Callable`` wrapper
        for the autograd function
        :class:`FlexSNFunction <spikingjelly.activation_based.triton_kernel.flexsn.wrapper.FlexSNFunction>`.

        The input arguments of a ``FlexSNKernel`` object is ``[*input_seqs, *states]`` , where ``input_seqs`` is
        a list of input sequences, ``states`` is a list of initial states; the return value is
        ``[*output_seqs, *state_seqs]`` , where ``output_seqs`` is a list of output sequences, and
        ``state_seqs`` is a list of state sequences.

        .. admonition:: Warning
            :class: warning

            ``FlexSNKernel`` requires ``torch.jit`` to be disabled. Please set the env var
            ``PYTORCH_JIT=0``. See https://docs.pytorch.org/docs/2.8/jit.html#disable-jit-for-debugging .

        For detailed information about arguments, refer to :class:`FlexSN`.
        """
        super().__init__()
        jit_disabled = os.environ.get("PYTORCH_JIT", "1") == "0"
        if not jit_disabled:
            raise RuntimeError(
                "FlexSNKernel requires torch.jit to be disabled. "
                "Please set the env var PYTORCH_JIT=0 before running your script. "
                "See https://docs.pytorch.org/docs/2.8/jit.html#disable-jit-for-debugging ."
            )

        if example_inputs is None:
            example_inputs = [
                torch.randn([1], device="cuda") for _ in range(num_inputs + num_states)
            ]
            example_inputs = tuple(example_inputs)
        example_inputs = tuple(x.to("cuda") for x in example_inputs)

        self.inf_graph = triton_kernel.torch2triton.generate_inference_graph(
            core, example_inputs
        )
        self.fwd_graph, self.bwd_graph = (
            triton_kernel.torch2triton.generate_forward_and_backward_graph(
                core, example_inputs, requires_grad=requires_grad
            )
        )
        self.info = triton_kernel.flexsn.extract_info(
            self.fwd_graph, num_inputs, num_states, num_outputs
        )

        core_str, core_name = triton_kernel.torch2triton.generate_triton_code_str(
            self.inf_graph, core.__name__ + "_inference"
        )
        self.f_inf = triton_kernel.flexsn.get_flexsn_inference_kernel(
            core_str, core_name, info=self.info
        )

        core_str, core_name = triton_kernel.torch2triton.generate_triton_code_str(
            self.fwd_graph, core.__name__ + "_forward"
        )
        self.f_fwd = triton_kernel.flexsn.get_flexsn_forward_kernel(
            core_str, core_name, info=self.info
        )

        core_str, core_name = triton_kernel.torch2triton.generate_triton_code_str(
            self.bwd_graph, core.__name__ + "_backward"
        )
        self.f_bwd = triton_kernel.flexsn.get_flexsn_backward_kernel(
            core_str, core_name, info=self.info
        )

    def __call__(self, *args): # args: [*input_seqs, *states]
        return triton_kernel.flexsn.FlexSNFunction.apply(
            self.f_inf, self.f_fwd, self.f_bwd, self.info, *args
        ) # [*output_seqs, *state_seqs]


class FlexSN(base.MemoryModule):
    def __init__(
        self,
        core: Callable,
        num_inputs: int,
        num_states: int,
        num_outputs: int,
        example_inputs: Optional[Tuple[torch.Tensor]] = None,
        requires_grad: Optional[Tuple[bool]] = None,
        step_mode: str = "m",
        backend: str = "triton",
        store_state_seqs: bool = False,
    ):
        """
        **API Language:**
        :ref:`中文 <FlexSN.__init__-cn>` | :ref:`English <FlexSN.__init__-en>`

        ----

        .. _FlexSN.__init__-cn:

        * **中文**

        ``FlexSN`` 可以根据自定义的 PyTorch 单步函数 ``core`` 生成 Triton 多步脉冲神经元。
        ``FlexSN`` 在 :class:`FlexSNKernel` 的基础上，进一步实现了其他 SpikingJelly 神经元的功能。
        实例化后，``FlexSN`` 对象输入和输出的语义与其他 SpikingJelly 神经元一致，取决于步进模式 ``step_mode`` 。

        .. admonition:: 警告
            :class: warning

            使用 FlexSN 需要禁用 ``torch.jit``。请在运行脚本前设置环境变量 ``PYTORCH_JIT=0``。
            参见: https://docs.pytorch.org/docs/2.8/jit.html#disable-jit-for-debugging 。

        :param core: 描述单步前向推理的函数，签名应为 ``[*inputs, *states] -> [*outputs, *updated_states]``，
            其中输入与输出均为张量。“inputs”和“outputs”的数量任意，需用 ``num_inputs`` 和 ``num_outputs`` 指明。
            “states”的数量任意，与“updated_states”数量一致，且需用 ``num_states`` 指明。
        :type core: Callable

        :param num_inputs: 输入的数量，应严格对应 ``core`` 参数及 ``example_inputs`` 中“inputs”的数量。
        :type num_inputs: int

        :param num_states: 状态的数量，应严格对应 ``core`` 的参数、返回值及 ``example_inputs`` 中的“states”的数量。
        :type num_states: int

        :param num_outputs: 输出的数量，应严格对应 ``core`` 返回值中“outputs”的数量。
        :type num_outputs: int

        :param example_inputs: 提供给 ``core`` 的示例输入，形式为 ``[*inputs, *states]``。
            这些张量都会自动被放置到 ``cuda`` 设备上。若为 ``None`` ，则自动生成
            ``num_inputs + num_states`` 个仅含一个元素的张量。默认为 ``None`` 。
        :type example_inputs: Optional[Tuple[torch.Tensor]]

        :param requires_grad: 指示 ``core`` 的参数 (即 ``[*inputs, *states]``) 是否需要梯度。
            用于生成前向和反向计算图。长度应与 ``core`` 的参数及 ``example_inputs`` 对应。
            若为 ``None``，则所有参数均需梯度。默认 ``None``。
        :type requires_grad: Optional[Tuple[bool]]

        :param step_mode: 步进模式。Triton 内核仅在 ``"m"`` 模式下可用。默认 ``"m"``。
        :type step_mode: str

        :param backend: 使用的后端。``"triton"`` 仅在 ``step_mode="m"`` 时可用；``"torch"`` 始终可用。
            默认 ``"triton"``。
        :type backend: str

        :param store_state_seqs: 是否保存状态序列。如果为 ``True``，用户可以通过 ``state_seqs`` 属性访问。
            ``state_seqs`` 是个列表，每个元素是形状为 ``[T, ...]`` 的张量。默认 ``False``。
        :type store_state_seqs: bool

        ----

        .. _FlexSN.__init__-en:

        * **English**

        ``FlexSN`` can generate Triton multi-step spiking neuron from a customized PyTorch
        single-step function ``core`` . ``FlexSN`` is built upon :class:`FlexSNKernel`
        and further implements other features of SpikingJelly neurons. The input / output
        semantics of a ``FlexSN`` object is similar to those of other SpikingJelly neurons,
        depending on ``step_mode`` .

        .. admonition:: Warning
            :class: warning

            FlexSN requires ``torch.jit`` to be disabled. Please set the env var
            ``PYTORCH_JIT=0``. See https://docs.pytorch.org/docs/2.8/jit.html#disable-jit-for-debugging .

        :param core: a function describing the single-step inference dynamics of
            the spiking neuron. Its signature should be ``[*inputs, *states] -> [*outputs, *updated_states]``,
            and the arguments and return values should all be tensors. There can
            be arbitrary number of inputs and outputs (specified by ``num_inputs`` and ``num_outputs``).
            There can be arbitrary number of states (specified by ``num_states``),
            and the number of updated states should match the number of states.
        :type core: Callable

        :param num_inputs: number of inputs. It should strictly match the
            number of inputs" in ``core``'s arguments and ``example_inputs``.
        :type num_inputs: int

        :param num_states: number of states. It should strictly match the
            number of "states" in ``core``'s arguments, ``core``'s return values,
            and ``example_inputs``.
        :type num_states: int

        :param num_outputs: number of outputs. It should strictly match the
            number of "outputs" in ``core``'s return values.
        :type num_outputs: int

        :param example_inputs: example inputs to ``core`` with the form of ``[*inputs, *states]``.
            These tensors will be moved to ``cuda`` device. If None, ``example_inputs`` will be
            ``num_inputs + num_states`` tensors with single element. Defaults to ``None``.
        :type example_inputs: Optional[Tuple[torch.Tensor]]

        :param requires_grad: whether the core's arguments (i.e.
            ``[*inputs, *states]``) requires gradients. This info is used to
            generate the forward and backward graphs. Its length should match
            the number of ``core``'s arguments and the length of ``example_inputs``.
            If None, all argument tensors require grad. Defaults to ``None``.
        :type requires_grad: Optional[Tuple[bool]]

        :param step_mode: step mode. Triton kernel is available only in "m"
            mode. Defaults to ``"m"``.
        :type step_mode: str

        :param backend: backend to use. ``"triton"`` is available only when
            ``step_mode="m"``. ``"torch"`` is always available. Defaults to ``"triton"``.
        :type backend: str

        :param store_state_seqs: whether to store the state sequences. If ``True``,
            users can access the state sequences via ``state_seqs`` property.
            ``state_seqs`` is a list of tensors with shape ``[T, ...]``. Defaults
            to ``False``.
        :type store_state_seqs: bool
        """
        super().__init__()
        self.core = core
        self.num_inputs = num_inputs
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.step_mode = step_mode
        self.backend = backend
        self.store_state_seqs = store_state_seqs

        self.kernel = FlexSNKernel(core, num_inputs, num_states, num_outputs, example_inputs, requires_grad)
        # register states as memory buffers
        self.register_memory("states", None)

    @property
    def supported_backends(self):
        return ("triton", "torch")

    @property
    def store_state_seqs(self):
        return self._store_state_seqs

    @store_state_seqs.setter
    def store_state_seqs(self, value: bool):
        self._store_state_seqs = value
        if value:
            if not hasattr(self, "state_seqs"):
                self.register_memory("state_seqs", None)

    @staticmethod
    def init_states(num_states: int, step_mode: str, *args) -> List[torch.Tensor]:
        """
        **API Language:**
        :ref:`中文 <FlexSN.init_states-cn>` | :ref:`English <FlexSN.init_states-en>`

        ----

        .. _FlexSN.init_states-cn:

        * **中文**

        初始化神经元的状态张量。用户可以通过重写此方法来自定义状态初始化规则。默认情况下，所有
        状态均被初始化为零张量。

        :param num_states: 状态变量的数量。应与 ``core`` 的“states”部分中的张量数量一致。
        :type num_states: int

        :param step_mode: 本模块当前所处的步进模式。可选值为 ``"s"`` 或 ``"m"`` 。
        :type step_mode: str

        :param args: ``forward`` 的输入，即 ``[*inputs]``。用户应根据 ``args``
            和 ``FlexSN`` 的 ``step_mode`` 等信息来决定状态张量的初始化方式。
        :type args: Sequence[torch.Tensor]

        :return: 初始化后的状态张量列表，顺序对应了 ``core`` 参数的“states”部分。
        :rtype: List[torch.Tensor]

        ----

        .. _FlexSN.init_states-en:

        * **English**

        Initialize the neuron state tensors. Users can override this method to
        customize the state initialization rules. By default, all state tensors
        are initialized to zero.

        The state tensors are stored in the ``states`` attribute, which is a list
        of tensors, whose order corresponds to the "states" part of ``core``.

        :param num_states: number of states. It should strictly match the
            number of "states" in ``core``'s return values.
        :type num_states: int

        :param step_mode: the current step mode of this module. It can be ``"s"`` or ``"m"`` .
        :type step_mode: str

        :param args: the input of ``forward``, i.e., ``[*inputs]``.
            Users should initialize state tensors based on ``args`` and ``step_mode``.
        :type args: Sequence[torch.Tensor]

        :return: the list of initialized state tensors, whose order corresponds to
            the "states" part of ``core``.
        :rtype: List[torch.Tensor]

        """

        if step_mode == "s":
            return [torch.zeros_like(args[0]) for _ in range(num_states)]
        elif step_mode == "m":
            return [torch.zeros_like(args[0][0]) for _ in range(num_states)]
        else:
            raise ValueError(f"Unsupported step mode: {step_mode}")

    def single_step_forward(self, *args):
        # only torch backend is supported for single-step forward
        results = self.core(*args, *self.states)  # [*outputs, *states]
        self.states = results[self.num_outputs :]
        return results[: self.num_outputs]

    def multi_step_forward(self, *args):
        if self.backend == "torch":
            T = args[0].shape[0]
            output_seqs = [[] for _ in range(self.num_outputs)]
            if self.store_state_seqs:
                state_seqs = [[] for _ in range(self.num_states)]

            for t in range(T):
                outputs = self.single_step_forward(*[arg[t] for arg in args])
                for i in range(self.num_outputs):
                    output_seqs[i].append(outputs[i])
                if self.store_state_seqs:
                    for i in range(self.num_states):
                        state_seqs[i].append(self.states[i])

            if self.store_state_seqs:
                self.state_seqs = [torch.stack(v, dim=0) for v in state_seqs]

            return [torch.stack(y, dim=0) for y in output_seqs]

        elif self.backend == "triton":
            result_seqs = self.kernel(*args, *self.states)
            output_seqs = result_seqs[: self.num_outputs]
            state_seqs = result_seqs[self.num_outputs :]
            self.states = [v[-1] for v in state_seqs]
            if self.store_state_seqs:
                self.state_seqs = state_seqs
            return output_seqs

    def forward(self, *args):
        if self.states is None:
            self.states = self.init_states(self.num_states, self.step_mode, *args)
        output = super().forward(*args)
        return output[0] if len(output) == 1 else output

    def extra_repr(self):
        return (
            f"core={self.core.__name__}, "
            f"num_inputs={self.num_inputs}, "
            f"num_states={self.num_states}, "
            f"num_outputs={self.num_outputs}, "
        )
