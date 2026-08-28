from __future__ import annotations

import copy
import functools
from typing import Callable

import torch

from .. import base

__all__ = ["FlexSN"]


def _as_tuple(value) -> tuple:
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def _contains_tensor(value) -> bool:
    if isinstance(value, (torch.Tensor, torch.nn.Module)):
        return True
    if isinstance(value, functools.partial):
        return (
            _contains_tensor(value.func)
            or _contains_tensor(value.args)
            or _contains_tensor(value.keywords)
        )
    if isinstance(value, dict):
        return any(_contains_tensor(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(v) for v in value)
    return False


def _reject_captured_tensors(core: Callable) -> None:
    if isinstance(core, torch.nn.Module):
        raise TypeError(
            "FlexSN core must be a pure callable. Pass tensor parameters through "
            "static_inputs instead of capturing an nn.Module."
        )
    if isinstance(core, functools.partial):
        if _contains_tensor(core.args) or _contains_tensor(core.keywords):
            raise TypeError(
                "FlexSN core cannot capture tensor-valued partial arguments; "
                "pass them through static_inputs."
            )
        core = core.func
    closure = getattr(core, "__closure__", None)
    if closure is not None:
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if _contains_tensor(value):
                raise TypeError(
                    "FlexSN core cannot capture tensors or modules; pass tensor "
                    "values through static_inputs."
                )


class FlexSN(base.MemoryModule):
    def __init__(
        self,
        core: Callable,
        num_states: int,
        static_inputs: tuple[torch.Tensor, ...] = (),
        step_mode: str = "m",
        backend: str = "triton",
        store_state_seqs: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <FlexSN.__init__-cn>` | :ref:`English <FlexSN.__init__-en>`

        ----

        .. _FlexSN.__init__-cn:

        * **中文**

        根据纯 PyTorch 单步函数构造有状态脉冲神经元。``core`` 的签名为
        ``core(*step_inputs, *states, *static_inputs)``，返回
        ``(*outputs, *updated_states)``。输入和输出数量在第一次调用时推导，
        ``num_states`` 个更新状态始终位于返回值末尾。

        :param core: 不捕获 Tensor 或 ``nn.Module`` 的纯单步函数。
        :type core: Callable
        :param num_states: 状态张量数量。
        :type num_states: int
        :param static_inputs: 每个时间步复用的 Tensor。Parameter 注册为参数，
            其他 Tensor 注册为 buffer。
        :type static_inputs: tuple[torch.Tensor, ...]
        :param step_mode: ``"s"`` 或 ``"m"``。HOP/Triton 仅支持 ``"m"``。
        :type step_mode: str
        :param backend: ``"torch"``、``"hop"`` 或 ``"triton"``。
        :type backend: str
        :param store_state_seqs: managed 多步调用是否保存完整状态序列。
        :type store_state_seqs: bool
        :raises TypeError: ``core`` 捕获 Tensor/模块，或 static input 不是 Tensor。
        :raises ValueError: 参数或 backend/step-mode 组合无效。

        ----

        .. _FlexSN.__init__-en:

        * **English**

        Build a stateful spiking neuron from a pure PyTorch single-step
        callable. ``core`` has signature
        ``core(*step_inputs, *states, *static_inputs)`` and returns
        ``(*outputs, *updated_states)``. Input and output arities are inferred
        on the first call; the final ``num_states`` returns are the updated
        states.

        :param core: Pure single-step callable that captures no Tensor or module.
        :type core: Callable
        :param num_states: Number of state tensors.
        :type num_states: int
        :param static_inputs: Tensors reused at every step. Parameters are
            registered as parameters and other tensors as buffers.
        :type static_inputs: tuple[torch.Tensor, ...]
        :param step_mode: ``"s"`` or ``"m"``; HOP/Triton only support ``"m"``.
        :type step_mode: str
        :param backend: ``"torch"``, ``"hop"``, or ``"triton"``.
        :type backend: str
        :param store_state_seqs: Save full state sequences for managed multi-step calls.
        :type store_state_seqs: bool
        :raises TypeError: If ``core`` captures tensors/modules or a static input is not a Tensor.
        :raises ValueError: If an argument or backend/step-mode combination is invalid.
        """
        super().__init__()
        if not callable(core):
            raise TypeError("FlexSN core must be callable.")
        if not isinstance(num_states, int) or num_states < 0:
            raise ValueError("FlexSN num_states must be a non-negative integer.")
        _reject_captured_tensors(core)

        self.core = core
        self.num_states = num_states
        self._static_input_names = tuple(
            f"_static_input_{i}" for i in range(len(static_inputs))
        )
        for name, tensor in zip(self._static_input_names, static_inputs, strict=True):
            if isinstance(tensor, torch.nn.Parameter):
                self.register_parameter(name, tensor)
            elif isinstance(tensor, torch.Tensor):
                self.register_buffer(name, tensor)
            else:
                raise TypeError(
                    "FlexSN static_inputs must contain tensors, but got "
                    f"{type(tensor).__name__}."
                )

        self._state_names = tuple(f"_state_{i}" for i in range(num_states))
        for name in self._state_names:
            self.register_memory(name, None)

        self._num_inputs: int | None = None
        self._num_outputs: int | None = None
        self._triton_handle: int | None = None
        self._triton_handle_finalizer = None
        self.state_seqs: tuple[torch.Tensor, ...] | None = None
        self._store_state_seqs = bool(store_state_seqs)

        self.step_mode = step_mode
        self.backend = backend

    @property
    def static_inputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, name) for name in self._static_input_names)

    @property
    def states(self) -> tuple[torch.Tensor | None, ...]:
        return tuple(self._memories[name] for name in self._state_names)

    @states.setter
    def states(self, values: tuple[torch.Tensor, ...]) -> None:
        values = tuple(values)
        if len(values) != self.num_states:
            raise ValueError(
                f"FlexSN expected {self.num_states} states, but got {len(values)}."
            )
        for name, value in zip(self._state_names, values, strict=True):
            if not isinstance(value, torch.Tensor):
                raise TypeError("FlexSN states must be tensors.")
            self._memories[name] = value

    @property
    def store_state_seqs(self) -> bool:
        return self._store_state_seqs

    @store_state_seqs.setter
    def store_state_seqs(self, value: bool) -> None:
        self._store_state_seqs = bool(value)
        self.state_seqs = None

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch", "hop", "triton")

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value not in self.supported_backends:
            raise NotImplementedError(f"Unsupported FlexSN backend: {value!r}.")
        if value != "torch" and self.step_mode != "m":
            raise RuntimeError(f"FlexSN backend={value!r} requires step_mode='m'.")
        self._backend = value
        if hasattr(self, "state_seqs"):
            self.state_seqs = None

    @property
    def step_mode(self) -> str:
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str) -> None:
        if value not in ("s", "m"):
            raise ValueError(f"Unsupported FlexSN step_mode: {value!r}.")
        backend = getattr(self, "_backend", "torch")
        if value == "s" and backend != "torch":
            raise RuntimeError(
                f"FlexSN backend={backend!r} does not support step_mode='s'."
            )
        self._step_mode = value
        if hasattr(self, "state_seqs"):
            self.state_seqs = None

    def reset(self) -> None:
        super().reset()
        self.state_seqs = None

    @staticmethod
    def init_states(
        num_states: int, step_mode: str, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if not inputs:
            raise ValueError("FlexSN requires at least one input tensor.")
        reference = inputs[0] if step_mode == "s" else inputs[0][0]
        return tuple(torch.zeros_like(reference) for _ in range(num_states))

    def materialize_states(
        self,
        inputs: tuple[object, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        if len(states) != self.num_states:
            raise ValueError(
                f"FlexSN expected {self.num_states} states, but got {len(states)}."
            )
        if any(state is None for state in states):
            return self.init_states(self.num_states, step_mode, *inputs)
        return states

    def _validate_operands(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        step_mode: str,
    ) -> torch.Tensor:
        if not inputs:
            raise ValueError("FlexSN requires at least one input tensor.")
        if len(states) != self.num_states:
            raise ValueError(
                f"FlexSN expected {self.num_states} states, but got {len(states)}."
            )
        if len(static_inputs) != len(self._static_input_names):
            raise ValueError(
                f"FlexSN expected {len(self._static_input_names)} static inputs, "
                f"but got {len(static_inputs)}."
            )
        operands = (*inputs, *states, *static_inputs)
        if any(not isinstance(tensor, torch.Tensor) for tensor in operands):
            raise TypeError("FlexSN inputs, states, and static_inputs must be tensors.")
        if step_mode == "m":
            T = inputs[0].shape[0]
            if T == 0:
                raise ValueError("FlexSN does not support empty multi-step inputs.")
            if any(tensor.shape[0] != T for tensor in inputs):
                raise ValueError("FlexSN input sequences must share the same T.")
            step_inputs = tuple(tensor[0] for tensor in inputs)
            reference = step_inputs[0]
        else:
            step_inputs = inputs
            reference = inputs[0]
        for tensor in (*step_inputs, *states):
            if (
                tensor.numel() != reference.numel()
                or tensor.dtype != reference.dtype
                or tensor.device != reference.device
            ):
                raise ValueError(
                    "FlexSN inputs and states must share per-step numel, dtype, "
                    "and device."
                )
        for tensor in static_inputs:
            if (
                tensor.numel() not in (1, reference.numel())
                or tensor.dtype != reference.dtype
                or tensor.device != reference.device
            ):
                raise ValueError(
                    "FlexSN static inputs must be scalar or match the per-step "
                    "numel, dtype, and device."
                )
        if self._num_inputs is None:
            self._num_inputs = len(inputs)
        elif len(inputs) != self._num_inputs:
            raise ValueError(
                f"FlexSN core interface expects {self._num_inputs} inputs, "
                f"but got {len(inputs)}."
            )
        return reference

    def _split_returns(
        self, returns, reference: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        returns = _as_tuple(returns)
        if len(returns) < self.num_states:
            raise ValueError(
                f"FlexSN core returned {len(returns)} tensors, fewer than "
                f"num_states={self.num_states}."
            )
        num_outputs = len(returns) - self.num_states
        if self._num_outputs is None:
            self._num_outputs = num_outputs
        elif num_outputs != self._num_outputs:
            raise ValueError(
                f"FlexSN core interface expects {self._num_outputs} outputs, "
                f"but returned {num_outputs}."
            )
        if any(not isinstance(tensor, torch.Tensor) for tensor in returns):
            raise TypeError("FlexSN core must return tensors.")
        for tensor in returns:
            if (
                tensor.numel() != reference.numel()
                or tensor.dtype != reference.dtype
                or tensor.device != reference.device
            ):
                raise ValueError(
                    "FlexSN outputs and updated states must match the per-step "
                    "numel, dtype, and device."
                )
        return returns[:num_outputs], returns[num_outputs:]

    def _core_step(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        reference = self._validate_operands(inputs, states, static_inputs, "s")
        return self._split_returns(
            self.core(*inputs, *states, *static_inputs), reference
        )

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        *,
        static_inputs: tuple[torch.Tensor, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        return self._core_step(inputs, states, tuple(static_inputs))

    def _torch_scan(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        store_state_seqs: bool,
    ):
        output_steps = None
        state_steps = None
        for t in range(inputs[0].shape[0]):
            outputs, states = self._core_step(
                tuple(tensor[t] for tensor in inputs), states, static_inputs
            )
            if output_steps is None:
                output_steps = [[] for _ in outputs]
                state_steps = [[] for _ in states] if store_state_seqs else None
            for values, value in zip(output_steps, outputs, strict=True):
                values.append(value)
            if state_steps is not None:
                for values, value in zip(state_steps, states, strict=True):
                    values.append(value)
        outputs = tuple(torch.stack(values) for values in (output_steps or ()))
        state_seqs = (
            tuple(torch.stack(values) for values in state_steps)
            if state_steps is not None
            else None
        )
        return outputs, states, state_seqs

    def _expanded_static_inputs(
        self, reference: torch.Tensor, static_inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.expand_as(reference)
            if tensor.numel() == 1
            else tensor.reshape_as(reference)
            for tensor in static_inputs
        )

    def _wrapped_core(self, num_inputs: int, num_outputs: int):
        core = self.core
        num_states = self.num_states
        num_static = len(self._static_input_names)

        def wrapped(*args):
            inputs = args[:num_inputs]
            carried = args[num_inputs:]
            states = carried[:num_states]
            static_inputs = carried[num_states : num_states + num_static]
            returns = _as_tuple(core(*inputs, *states, *static_inputs))
            outputs = returns[:num_outputs]
            next_states = returns[num_outputs : num_outputs + num_states]
            return (*outputs, *next_states, *static_inputs)

        return wrapped

    def _hop_scan(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        store_state_seqs: bool,
    ):
        from ..triton_kernel.flexsn.hop import flex_sn_scan

        reference = self._validate_operands(inputs, states, static_inputs, "m")
        probe_outputs, _ = self._split_returns(
            self.core(*(tensor[0] for tensor in inputs), *states, *static_inputs),
            reference,
        )
        num_outputs = len(probe_outputs)
        expanded_static = self._expanded_static_inputs(reference, static_inputs)
        wrapped = self._wrapped_core(len(inputs), num_outputs)
        flat_args = (*inputs, *states, *expanded_static)
        total_states = self.num_states + len(expanded_static)
        results = flex_sn_scan(
            wrapped, len(inputs), total_states, num_outputs, *flat_args
        )
        outputs = tuple(results[:num_outputs])
        all_state_seqs = tuple(results[num_outputs : num_outputs + self.num_states])
        states = tuple(sequence[-1] for sequence in all_state_seqs)
        state_seqs = all_state_seqs if store_state_seqs else None
        return outputs, states, state_seqs

    def _ensure_triton_runtime(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        num_outputs: int,
    ) -> None:
        if self._triton_handle is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("FlexSN backend='triton' requires CUDA.")

        from ..triton_kernel.flexsn.custom_ops import (
            attach_flexsn_handle_finalizer,
            register_flexsn_kernel_handle,
        )
        from ..triton_kernel.flexsn.kernel import (
            build_inference_final_state_kernel,
            build_inference_kernel,
            build_training_kernels,
        )

        reference = inputs[0][0]
        expanded_static = self._expanded_static_inputs(reference, static_inputs)
        examples = tuple(tensor[0] for tensor in inputs) + states + expanded_static
        wrapped = self._wrapped_core(len(inputs), num_outputs)
        total_states = self.num_states + len(expanded_static)
        inference_kernel, inference_info = build_inference_kernel(
            wrapped, len(inputs), total_states, num_outputs, examples
        )
        final_kernel, final_info = build_inference_final_state_kernel(
            wrapped, len(inputs), total_states, num_outputs, examples
        )
        forward_kernel, backward_kernel, training_info = build_training_kernels(
            wrapped, len(inputs), total_states, num_outputs, examples
        )
        self._triton_handle = register_flexsn_kernel_handle(
            inference_kernel=inference_kernel,
            inference_info=inference_info,
            inference_final_state_kernel=final_kernel,
            inference_final_state_info=final_info,
            forward_kernel=forward_kernel,
            backward_kernel=backward_kernel,
            training_info=training_info,
        )
        self._triton_handle_finalizer = attach_flexsn_handle_finalizer(
            self, self._triton_handle
        )

    def _triton_scan(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        store_state_seqs: bool,
    ):
        from ..triton_kernel.flexsn.custom_ops import (
            flexsn_triton_inference,
            flexsn_triton_inference_final_state,
            flexsn_triton_training,
            flexsn_triton_training_final_state,
        )

        reference = self._validate_operands(inputs, states, static_inputs, "m")
        probe_outputs, _ = self._split_returns(
            self.core(*(tensor[0] for tensor in inputs), *states, *static_inputs),
            reference,
        )
        num_outputs = len(probe_outputs)
        self._ensure_triton_runtime(inputs, states, static_inputs, num_outputs)
        expanded_static = self._expanded_static_inputs(reference, static_inputs)
        flat_args = [*inputs, *states, *expanded_static]
        use_training = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in flat_args
        )
        if use_training:
            op = (
                flexsn_triton_training
                if store_state_seqs
                else flexsn_triton_training_final_state
            )
        else:
            op = (
                flexsn_triton_inference
                if store_state_seqs
                else flexsn_triton_inference_final_state
            )
        results = op(self._triton_handle, flat_args)
        outputs = tuple(results[:num_outputs])
        state_results = tuple(results[num_outputs : num_outputs + self.num_states])
        if store_state_seqs:
            state_seqs = state_results
            states = tuple(sequence[-1] for sequence in state_seqs)
        else:
            states = state_results
            state_seqs = None
        return outputs, states, state_seqs

    def _multi_step(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
        static_inputs: tuple[torch.Tensor, ...],
        store_state_seqs: bool,
    ):
        self._validate_operands(inputs, states, static_inputs, "m")
        if self.backend == "torch":
            return self._torch_scan(inputs, states, static_inputs, store_state_seqs)
        if self.backend == "hop":
            return self._hop_scan(inputs, states, static_inputs, store_state_seqs)
        if self.backend == "triton":
            return self._triton_scan(inputs, states, static_inputs, store_state_seqs)
        raise ValueError(self.backend)

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        *,
        static_inputs: tuple[torch.Tensor, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        outputs, states, _ = self._multi_step(
            inputs, states, tuple(static_inputs), False
        )
        return outputs, states

    def forward(self, *inputs: torch.Tensor):
        static_inputs = self.static_inputs
        if self.step_mode == "m" and inputs and inputs[0].shape[0] == 0:
            raise ValueError("FlexSN does not support empty multi-step inputs.")
        states = self.materialize_states(inputs, self.states, self.step_mode)
        if self.step_mode == "s":
            outputs, states = self.single_step_functional_forward(
                inputs, states, static_inputs=static_inputs
            )
            self.state_seqs = None
        else:
            outputs, states, self.state_seqs = self._multi_step(
                inputs, states, static_inputs, self.store_state_seqs
            )
        self.states = states
        return outputs[0] if len(outputs) == 1 else outputs

    def _release_triton_runtime(self) -> None:
        finalizer = self._triton_handle_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()
        self._triton_handle = None
        self._triton_handle_finalizer = None

    def _apply(self, fn, recurse=True):
        self._release_triton_runtime()
        return super()._apply(fn, recurse=recurse)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key in {"_triton_handle", "_triton_handle_finalizer"}:
                continue
            result.__dict__[key] = copy.deepcopy(value, memo)
        result._triton_handle = None
        result._triton_handle_finalizer = None
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_triton_handle"] = None
        state["_triton_handle_finalizer"] = None
        return state

    def extra_repr(self) -> str:
        core_name = getattr(self.core, "__name__", type(self.core).__name__)
        return (
            f"core={core_name}, num_states={self.num_states}, "
            f"num_static_inputs={len(self._static_input_names)}, "
            f"step_mode={self.step_mode!r}, backend={self.backend!r}"
        )
