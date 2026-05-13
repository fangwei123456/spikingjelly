import contextlib
import copy
import functools
import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .. import base
from ..triton_kernel.flexsn.custom_ops import (
    flexsn_inductor_inference,
    flexsn_inductor_inference_final_state,
    flexsn_inductor_training,
    flexsn_inductor_training_final_state,
)

try:
    from ..triton_kernel.flexsn import (
        dynamo_hop_available as _flexsn_dynamo_hop_available,
    )
    from ..triton_kernel.flexsn import eager_scan as _flexsn_eager_scan
    from ..triton_kernel.flexsn import (
        eager_scan_final_state as _flexsn_eager_scan_final_state,
    )
    from ..triton_kernel.flexsn import flex_sn_scan as _flexsn_hop_scan
    from ..triton_kernel.flexsn import lowerable_scan as _flexsn_lowerable_scan
    from ..triton_kernel.flexsn import (
        lowerable_scan_available as _flexsn_lowerable_scan_available,
    )
    from ..triton_kernel.flexsn import (
        lowerable_scan_final_state as _flexsn_lowerable_scan_final_state,
    )
    from ..triton_kernel.flexsn import (
        lowerable_while_loop_available as _flexsn_lowerable_while_loop_available,
    )
    from ..triton_kernel.flexsn import (
        lowerable_while_loop_scan as _flexsn_lowerable_while_loop_scan,
    )
    from ..triton_kernel.flexsn import (
        lowerable_while_loop_scan_final_state as _flexsn_lowerable_while_loop_scan_final_state,
    )
except (ImportError, AttributeError) as e:
    logging.info(f"spikingjelly.activation_based.neuron.flexsn: {e}")
    _flexsn_eager_scan = None
    _flexsn_eager_scan_final_state = None
    _flexsn_hop_scan = None
    _flexsn_lowerable_scan = None
    _flexsn_lowerable_scan_final_state = None
    _flexsn_lowerable_scan_available = None
    _flexsn_dynamo_hop_available = None
    _flexsn_lowerable_while_loop_scan = None
    _flexsn_lowerable_while_loop_scan_final_state = None
    _flexsn_lowerable_while_loop_available = None


__all__ = ["FlexSNKernel", "FlexSN"]


def _is_flexsn_triton_backend(backend: str) -> bool:
    return backend in ("triton", "inductor")


def _warmup_inductor_inference_final_state_kernel(module: "FlexSN") -> None:
    if (
        module._inductor_scan_final_state_kernel is None
        or module._inductor_scan_final_state_info is None
    ):
        return

    from ..triton_kernel.flexsn.wrapper import flexsn_inference_final_state

    info = module._inductor_scan_final_state_info
    device = getattr(module, "_inductor_scan_final_state_device", None)
    if device is not None and device.type != "cuda":
        device = None
    if isinstance(module.core, torch.nn.Module):
        if device is None:
            for tensor in [*module.core.parameters(), *module.core.buffers()]:
                if tensor.device.type == "cuda":
                    device = tensor.device
                    break
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    warm_args = _make_inductor_final_state_warmup_args(
        info,
        device,
        getattr(module, "_inductor_scan_final_state_warmup_specs", None),
    )

    device_guard = (
        torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    )
    with torch.no_grad(), device_guard:
        flexsn_inference_final_state(
            module._inductor_scan_final_state_kernel,
            info,
            *warm_args,
        )


def _make_inductor_final_state_warmup_specs(
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
    expected: int,
):
    if example_inputs is None or len(example_inputs) < expected:
        return None
    return tuple(
        (tuple(tensor.shape), tensor.dtype) for tensor in example_inputs[:expected]
    )


def _make_inductor_final_state_warmup_args(info, device, warmup_specs):
    expected = info.num_inputs + info.num_states
    if warmup_specs is not None and len(warmup_specs) >= expected:
        warm_args = [
            torch.zeros((1, *shape), dtype=dtype, device=device)
            for shape, dtype in warmup_specs[: info.num_inputs]
        ]
        warm_args.extend(
            torch.zeros(shape, dtype=dtype, device=device)
            for shape, dtype in warmup_specs[info.num_inputs : expected]
        )
        return warm_args

    seq_template = torch.zeros((1, 1), device=device)
    state_template = seq_template[0].clone()
    warm_args = [seq_template.clone() for _ in range(info.num_inputs)]
    warm_args.extend(state_template.clone() for _ in range(info.num_states))
    return warm_args


def _first_cuda_device(tensors):
    if tensors is not None:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                return tensor.device
    return None


def _as_tuple(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


def _flat_args_on_single_cuda_device(
    flat_args: Sequence[torch.Tensor],
) -> Tuple[bool, bool]:
    if len(flat_args) == 0:
        return False, False
    first = flat_args[0]
    all_cuda = all(t.is_cuda for t in flat_args)
    same_device = all(t.device == first.device for t in flat_args)
    return all_cuda, same_device


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling):
        return is_compiling()

    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    if callable(is_compiling):
        return is_compiling()

    return False


def _run_hop_scan(
    core: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    store_state_seqs: bool,
    *flat_args: torch.Tensor,
    output_template_specs=None,
):
    enable_lowerable_while_loop = (
        os.environ.get("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_WHILE_LOOP", "0") == "1"
    )
    enable_lowerable_scan = (
        os.environ.get("SJ_ENABLE_EXPERIMENTAL_LOWERABLE_SCAN", "0") == "1"
    )
    is_compiling = _is_compiling()
    # flexsn imports scan helpers as an all-or-none group and sets all
    # of them to None on failure, so _flexsn_eager_scan is the availability
    # sentinel for this backend family.
    if _flexsn_eager_scan is None:
        raise RuntimeError(
            "FlexSN HOP backend is unavailable: eager_scan failed to import. "
            "See logs from "
            "spikingjelly.activation_based.triton_kernel.flexsn."
        )

    lowerable_while_loop_impl = (
        _flexsn_lowerable_while_loop_scan
        if store_state_seqs
        else _flexsn_lowerable_while_loop_scan_final_state
    )
    lowerable_scan_impl = (
        _flexsn_lowerable_scan
        if store_state_seqs
        else _flexsn_lowerable_scan_final_state
    )
    use_lowerable_while_loop = (
        is_compiling
        and lowerable_while_loop_impl is not None
        and callable(_flexsn_lowerable_while_loop_available)
        and _flexsn_lowerable_while_loop_available()
        and (not torch.is_grad_enabled())
        and enable_lowerable_while_loop
    )
    use_lowerable_scan = (
        is_compiling
        and lowerable_scan_impl is not None
        and callable(_flexsn_lowerable_scan_available)
        and _flexsn_lowerable_scan_available()
        and (not torch.is_grad_enabled())
        and enable_lowerable_scan
    )

    if use_lowerable_while_loop:
        scan_impl = lowerable_while_loop_impl
    elif use_lowerable_scan:
        scan_impl = lowerable_scan_impl
    elif (
        _flexsn_hop_scan is not None
        and store_state_seqs
        and (
            not is_compiling
            or (
                callable(_flexsn_dynamo_hop_available)
                and _flexsn_dynamo_hop_available()
            )
        )
    ):
        scan_impl = _flexsn_hop_scan
    elif not store_state_seqs:
        scan_impl = _flexsn_eager_scan_final_state
    else:
        scan_impl = _flexsn_eager_scan
    if scan_impl is None:
        raise RuntimeError("FlexSN HOP backend has no available scan implementation.")

    template_kwargs = (
        {}
        if output_template_specs is None
        else {"output_template_specs": output_template_specs}
    )
    return scan_impl(
        core,
        num_inputs,
        num_states,
        num_outputs,
        *flat_args,
        **template_kwargs,
    )


def _can_elide_zero_state_inputs(module: "FlexSN") -> bool:
    return (
        _is_flexsn_triton_backend(module.backend)
        and module._inductor_handle is not None
        and module._memories_rv.get("states") is None
        and module.__class__.init_states is FlexSN.init_states
    )


def _last_state_or_current(
    state_seq: torch.Tensor,
    current_state: torch.Tensor,
) -> torch.Tensor:
    return current_state if state_seq.shape[0] == 0 else state_seq[-1]


def _empty_multistep_outputs(
    args: Tuple[torch.Tensor, ...],
    states: List[torch.Tensor],
    num_outputs: int,
    output_template_specs: Optional[Tuple[Tuple, ...]] = None,
    *,
    use_template_device: bool = True,
) -> List[torch.Tensor]:
    def _empty_output(i: int) -> torch.Tensor:
        if output_template_specs is not None and i < len(output_template_specs):
            spec = output_template_specs[i]
            if len(spec) == 2:
                shape, dtype = spec
                device = args[0].device
            else:
                shape, dtype, template_device = spec
                device = template_device if use_template_device else args[0].device
            return torch.empty((0, *shape), dtype=dtype, device=device)
        if args:
            ref = args[0].new_empty(args[0].shape[1:])
        elif states:
            ref = states[-1]
        else:
            raise ValueError(
                "FlexSN empty output fallback requires at least one input or state."
            )
        return ref.new_empty((0, *ref.shape))

    return [_empty_output(i) for i in range(num_outputs)]


def _make_output_template_specs_from_outputs(
    num_outputs: int,
    example_outputs: Optional[Tuple[torch.Tensor, ...]],
) -> Optional[Tuple[Tuple, ...]]:
    if example_outputs is None:
        return None
    if len(example_outputs) != num_outputs:
        raise ValueError(
            f"FlexSN expected {num_outputs} example output tensors, but got "
            f"{len(example_outputs)}."
        )
    specs = []
    for i, tensor in enumerate(example_outputs):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"FlexSN example output #{i} is {type(tensor)!r}; expected a tensor."
            )
        specs.append((tuple(tensor.shape), tensor.dtype, tensor.device))
    return tuple(specs)


def _runtime_output_template_specs(
    output_template_specs: Optional[Tuple[Tuple, ...]],
) -> Optional[Tuple[Tuple, ...]]:
    if output_template_specs is None:
        return None
    return tuple((tuple(shape), dtype) for shape, dtype, *_ in output_template_specs)


def _validate_scan_backend_output_template_specs(
    output_template_specs: Optional[Tuple[Tuple, ...]],
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
) -> None:
    if output_template_specs is None or example_inputs is None:
        return
    seq_template = example_inputs[0]
    expected_shape = tuple(seq_template.shape)
    expected_dtype = seq_template.dtype
    for i, spec in enumerate(output_template_specs):
        shape, dtype = spec[:2]
        if tuple(shape) != expected_shape or dtype != expected_dtype:
            raise ValueError(
                "FlexSN Triton path (backend='triton'/'inductor') requires example_outputs "
                "to match the first example input's per-step shape and dtype "
                f"({expected_shape}, {expected_dtype}), but example output "
                f"#{i} is ({tuple(shape)}, {dtype})."
            )


def _core_requires_grad(core: Callable) -> bool:
    if isinstance(core, functools.partial):
        return (
            _core_requires_grad(core.func)
            or _value_requires_grad(core.args)
            or _value_requires_grad(core.keywords)
        )

    bound_self = getattr(core, "__self__", None)
    if bound_self is not None and _value_requires_grad(bound_self):
        return True
    if bound_self is not None:
        bound_self_dict = getattr(bound_self, "__dict__", None)
        if bound_self_dict is not None:
            values = bound_self_dict.values()
            if isinstance(bound_self, torch.nn.Module):
                values = (
                    value
                    for key, value in bound_self_dict.items()
                    if key not in {"_parameters", "_buffers", "_modules"}
                )
            if any(_value_requires_grad(value) for value in values):
                return True

    if isinstance(core, torch.nn.Module):
        for tensor in [*core.parameters(), *core.buffers()]:
            if tensor.requires_grad:
                return True

    if callable(core) and not hasattr(core, "__code__"):
        core_dict = getattr(core, "__dict__", None)
        if core_dict is not None:
            for value in core_dict.values():
                if _value_requires_grad(value):
                    return True

    closure = getattr(core, "__closure__", None)
    if closure is None:
        return False
    for cell in closure:
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        if _value_requires_grad(cell_value):
            return True
    return False


def _value_requires_grad(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, torch.nn.Module):
        return any(
            tensor.requires_grad for tensor in [*value.parameters(), *value.buffers()]
        )
    if isinstance(value, functools.partial):
        return (
            _core_requires_grad(value.func)
            or _value_requires_grad(value.args)
            or _value_requires_grad(value.keywords)
        )
    if isinstance(value, dict):
        return any(_value_requires_grad(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_value_requires_grad(v) for v in value)
    return False


def _validate_scan_backend_contract(
    core: Callable,
    num_inputs: int,
    num_states: int,
    num_outputs: int,
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
):
    if num_inputs + num_states == 0:
        raise ValueError("FlexSN requires at least one input or state tensor.")
    if num_inputs == 0:
        raise ValueError(
            "FlexSN Triton path (backend='triton'/'inductor') requires at least one input "
            "sequence to derive T."
        )

    if example_inputs is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        example_inputs = tuple(
            torch.zeros(1, device=device) for _ in range(num_inputs + num_states)
        )
    else:
        device = example_inputs[0].device
        example_inputs = tuple(x.detach().to(device).clone() for x in example_inputs)

    seq_template = example_inputs[0]
    for i, tensor in enumerate(example_inputs[1:], start=1):
        if tensor.numel() != seq_template.numel():
            raise ValueError(
                "FlexSN Triton path (backend='triton'/'inductor') currently requires every "
                "example input and state tensor to have the same number of "
                f"elements as the first example tensor ({seq_template.numel()}), "
                f"but example #{i} has {tensor.numel()} elements."
            )
        if tensor.dtype != seq_template.dtype:
            raise ValueError(
                "FlexSN Triton path (backend='triton'/'inductor') currently requires every "
                "example input and state tensor to match the first example "
                f"tensor's dtype ({seq_template.dtype}), but example #{i} has "
                f"dtype {tensor.dtype}."
            )

    with torch.no_grad():
        returns = _as_tuple(core(*example_inputs))

    expected = num_outputs + num_states
    if len(returns) != expected:
        raise ValueError(
            f"FlexSN core returned {len(returns)} values, but expected "
            f"{expected} (= num_outputs + num_states)."
        )

    for i, tensor in enumerate(returns):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"FlexSN core return #{i} is {type(tensor)!r}; "
                "scan backends require tensor outputs."
            )
        if tensor.shape != seq_template.shape:
            raise ValueError(
                "FlexSN Triton path (backend='triton'/'inductor') currently requires every "
                "per-step output and updated state to have the same shape as "
                f"the first example tensor {tuple(seq_template.shape)}, but "
                f"return #{i} has shape {tuple(tensor.shape)}."
            )
        if tensor.dtype != seq_template.dtype or tensor.device != seq_template.device:
            raise ValueError(
                "FlexSN Triton path (backend='triton'/'inductor') currently requires every "
                "per-step output and updated state to match the first example "
                f"tensor's dtype/device ({seq_template.dtype}, "
                f"{seq_template.device}), but return #{i} is "
                f"({tensor.dtype}, {tensor.device})."
            )
    return example_inputs


class FlexSNKernel:
    def __init__(
        self,
        core: Callable,
        num_inputs: int,
        num_states: int,
        num_outputs: int,
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
        不同于 :class:`FlexSN` ， ``FlexSNKernel`` 是对底层 custom op 调度的轻量 ``Callable`` 封装。

        实例化后， ``FlexSNKernel`` 对象接受的输入参数为 ``[*input_seqs, *states]`` ，其中 ``input_seqs`` 是
        ``num_inputs`` 个输入序列，``states`` 是 ``num_states`` 个初始状态；返回值为 ``[*output_seqs, *state_seqs]`` ，
        其中 ``output_seqs`` 是 ``num_outputs`` 个输出序列，``state_seqs`` 是 ``num_states`` 个状态序列。

        :param core: 描述单步前向推理的函数，签名应为
            ``[*inputs, *states] -> [*outputs, *updated_states]``，其中输入、输出和状态均为张量。
        :type core: Callable

        :param num_inputs: 输入序列的数量。
        :type num_inputs: int

        :param num_states: 初始状态张量的数量，同时也应与返回的更新后状态数量一致。
        :type num_states: int

        :param num_outputs: 输出序列的数量。
        :type num_outputs: int

        :param example_inputs: 传给 ``core`` 的示例张量，形式为 ``[*inputs, *states]``，用于辅助构建推理与训练 kernel。
            若为 ``None``，则由底层构建器使用默认示例张量。
        :type example_inputs: Optional[Tuple[torch.Tensor]]

        :param requires_grad: 指示 ``core`` 各个输入参数是否需要梯度的布尔元组，仅用于训练 kernel 的构建。
            若为 ``None``，则由底层构建器采用默认行为。
        :type requires_grad: Optional[Tuple[bool]]

        :raises RuntimeError: 当前环境未启用 CUDA 时抛出，因为 ``FlexSNKernel`` 仅支持运行在 CUDA 设备上的 Triton 内核。

        ----

        .. _FlexSNKernel.__init__-en:

        * **English**

        ``FlexSNKernel`` can generate Triton multi-step spiking neuron kernels
        from a customized PyTorch single-step function ``core`` via FlexSN's
        triton / inductor backend.
        It is a lightweight ``Callable`` wrapper over the underlying FlexSN
        custom-op dispatch path.

        The input arguments of a ``FlexSNKernel`` object is ``[*input_seqs, *states]`` , where ``input_seqs`` is
        a list of input sequences, ``states`` is a list of initial states; the return value is
        ``[*output_seqs, *state_seqs]`` , where ``output_seqs`` is a list of output sequences, and
        ``state_seqs`` is a list of state sequences.

        :param core: function describing the single-step inference dynamics with
            signature ``[*inputs, *states] -> [*outputs, *updated_states]``.
            Inputs, outputs, and states should all be tensors.
        :type core: Callable

        :param num_inputs: number of input sequences.
        :type num_inputs: int

        :param num_states: number of initial state tensors, which should also
            match the number of updated states returned by ``core``.
        :type num_states: int

        :param num_outputs: number of output sequences.
        :type num_outputs: int

        :param example_inputs: example tensors passed to ``core`` in the form
            ``[*inputs, *states]``. They are used to help build inference and
            training kernels. If ``None``, the backend builders use their
            default example tensors.
        :type example_inputs: Optional[Tuple[torch.Tensor]]

        :param requires_grad: tuple indicating whether each argument of
            ``core`` requires gradients. It is only used when building the
            training kernels. If ``None``, the backend builders use their
            default behavior.
        :type requires_grad: Optional[Tuple[bool]]

        :raises RuntimeError: raised when CUDA is unavailable, because
            ``FlexSNKernel`` only supports the Triton kernels on CUDA devices.
        """
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlexSNKernel requires CUDA, but torch.cuda.is_available() "
                "is False. Use FlexSN with backend='torch' or backend='hop' "
                "for non-Triton paths."
            )
        from ..triton_kernel.flexsn.custom_ops import (
            attach_flexsn_handle_finalizer,
            register_flexsn_kernel_handle,
        )
        from ..triton_kernel.flexsn.kernel import (
            build_inference_kernel,
            build_training_kernels,
        )

        self.f_inf, _inference_info = build_inference_kernel(
            core,
            num_inputs,
            num_states,
            num_outputs,
            example_inputs=example_inputs,
        )
        self._core = core
        self._core_requires_grad = _core_requires_grad(core)
        self.f_fwd, self.f_bwd, self.info = build_training_kernels(
            core,
            num_inputs,
            num_states,
            num_outputs,
            example_inputs=example_inputs,
            requires_grad=requires_grad,
        )

        if (
            _inference_info.num_inputs != self.info.num_inputs
            or _inference_info.num_states != self.info.num_states
            or _inference_info.num_outputs != self.info.num_outputs
        ):
            raise RuntimeError(
                "FlexSNKernel inference/training metadata mismatch when building "
                "FlexSN Triton kernels."
            )

        self._num_visible_returns = self.info.num_outputs + self.info.num_states
        self._handle = register_flexsn_kernel_handle(
            inference_kernel=self.f_inf,
            inference_info=_inference_info,
            inference_final_state_kernel=None,
            inference_final_state_info=None,
            forward_kernel=self.f_fwd,
            backward_kernel=self.f_bwd,
            training_info=self.info,
        )
        self._handle_finalizer = attach_flexsn_handle_finalizer(self, self._handle)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        shared_runtime_keys = {
            "_core",
            "_core_requires_grad",
            "f_inf",
            "f_fwd",
            "f_bwd",
            "info",
            "_num_visible_returns",
            "_handle",
        }
        for key, value in self.__dict__.items():
            if key == "_handle_finalizer":
                continue
            if key in shared_runtime_keys:
                result.__dict__[key] = value
            else:
                result.__dict__[key] = copy.deepcopy(value, memo)

        handle = result.__dict__.get("_handle")
        if handle is not None:
            from ..triton_kernel.flexsn.custom_ops import (
                attach_flexsn_handle_finalizer,
                retain_owner_flexsn_kernel_handle,
            )

            retain_owner_flexsn_kernel_handle(handle)
            result._handle_finalizer = attach_flexsn_handle_finalizer(result, handle)
        else:
            result._handle_finalizer = None

        return result

    def __call__(self, *args):  # args: [*input_seqs, *states]
        flat_args = list(args)
        expected = self.info.num_inputs + self.info.num_states
        if len(flat_args) != expected:
            raise ValueError(
                "FlexSNKernel expected "
                f"{expected} tensors "
                f"({self.info.num_inputs} input sequences + "
                f"{self.info.num_states} initial states), "
                f"but got {len(flat_args)}."
            )
        bad_arg = next(
            (
                (i, type(arg).__name__)
                for i, arg in enumerate(flat_args)
                if not isinstance(arg, torch.Tensor)
            ),
            None,
        )
        if bad_arg is not None:
            index, type_name = bad_arg
            raise TypeError(
                f"FlexSNKernel expected tensor arguments, but arg #{index} is "
                f"{type_name}."
            )
        all_cuda, same_device = _flat_args_on_single_cuda_device(flat_args)
        if self._handle is None or not all_cuda or not same_device:
            raise RuntimeError(
                "FlexSNKernel requires all input sequences and initial states "
                "to be CUDA tensors on the same device."
            )

        use_training = torch.is_grad_enabled() and (
            _core_requires_grad(self._core)
            or any(tensor.requires_grad for tensor in flat_args)
        )
        if use_training:
            outputs = flexsn_inductor_training(self._handle, flat_args)[
                : self._num_visible_returns
            ]
        else:
            outputs = flexsn_inductor_inference(self._handle, flat_args)

        return tuple(outputs)


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
        example_outputs: Optional[Tuple[torch.Tensor]] = None,
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

        :param step_mode: 步进模式。``"triton"`` 和 ``"inductor"`` 内核仅在 ``"m"`` 模式下可用。默认 ``"m"``。
        :type step_mode: str

        :param backend: 使用的后端。``"triton"``、``"inductor"`` 和 ``"hop"``
            仅在 ``step_mode="m"`` 时可用; ``"torch"`` 始终可用。``"triton"``
            与 ``"inductor"`` 在 FlexSN 中是并列且等价的 Triton 类标签，
            当前共享同一条维护中的 Triton 执行路径。默认 ``"triton"``。
        :type backend: str

        :param store_state_seqs: 是否保存状态序列。如果为 ``True``，用户可以通过 ``state_seqs`` 属性访问。
            ``state_seqs`` 是个列表，每个元素是形状为 ``[T, ...]`` 的张量。默认 ``False``。
        :type store_state_seqs: bool

        :param example_outputs: ``core`` 的单步输出模板，形式为 ``tuple([*outputs])``。
            当 ``backend="torch"`` 且输入为空序列 ``T == 0`` 时, 需要用它来构造输出张量的
            形状和 dtype, 从而避免为了推断输出而执行 ``core``。对于 ``"triton"``
            与 ``"inductor"`` 这两个等价的后端, 若提供该参数, 每个模板张量都必须与第一个 ``example_inputs``
            张量的单步形状和 dtype 相匹配。``"hop"`` 后端会保留任意
            输出模板, 并在空序列/HOP 路径中按运行时输入设备物化它们, 不执行上述形状/dtype
            校验。若不需要空序列模板, 则可以为 ``None``。默认 ``None``。
        :type example_outputs: Optional[Tuple[torch.Tensor]]

        ----

        .. _FlexSN.__init__-en:

        * **English**

        ``FlexSN`` can generate Triton multi-step spiking neuron from a customized PyTorch
        single-step function ``core`` . ``FlexSN`` is built upon :class:`FlexSNKernel`
        and further implements other features of SpikingJelly neurons. The input / output
        semantics of a ``FlexSN`` object is similar to those of other SpikingJelly neurons,
        depending on ``step_mode`` .

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

        :param step_mode: step mode. ``"triton"`` and ``"inductor"`` backends are available only in
            "m" mode. Defaults to ``"m"``.
        :type step_mode: str

        :param backend: backend to use. ``"triton"``, ``"inductor"``, and
            ``"hop"`` are available only when ``step_mode="m"``. ``"torch"``
            is always available. In FlexSN, ``"triton"`` and ``"inductor"``
            are peer labels and currently dispatch the
            same maintained Triton execution path. Defaults to ``"triton"``.
        :type backend: str

        :param store_state_seqs: whether to store the state sequences. If ``True``,
            users can access the state sequences via ``state_seqs`` property.
            ``state_seqs`` is a list of tensors with shape ``[T, ...]``. Defaults
            to ``False``.
        :type store_state_seqs: bool

        :param example_outputs: per-step output templates for ``core`` with the form of
            ``tuple([*outputs])``. When ``backend="torch"`` and the input sequence is
            empty (``T == 0``), these templates are required to materialize output
            shapes and dtypes without executing ``core``. For the equivalent
            ``"triton"`` / ``"inductor"`` backends, each provided
            template must match the first
            ``example_inputs`` tensor's per-step shape and dtype. The ``"hop"``
            backend intentionally allows arbitrary output templates and materializes
            them on the runtime input device for empty-sequence/HOP paths, so it
            does not enforce that scan-backend shape/dtype check. Defaults to
            ``None`` when empty-sequence output templates are not needed.
        :type example_outputs: Optional[Tuple[torch.Tensor]]
        """
        super().__init__()
        self.core = core
        self.num_inputs = num_inputs
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.step_mode = step_mode
        self.backend = backend
        self.store_state_seqs = store_state_seqs
        self._inductor_scan_final_state_warmup_specs = (
            _make_inductor_final_state_warmup_specs(
                example_inputs,
                num_inputs + num_states,
            )
        )
        self._explicit_output_template_specs = _make_output_template_specs_from_outputs(
            num_outputs,
            example_outputs,
        )
        self._output_template_specs = _runtime_output_template_specs(
            self._explicit_output_template_specs
        )

        if step_mode == "m" and num_inputs == 0:
            raise ValueError(
                "FlexSN step_mode='m' requires at least one input sequence to "
                "derive T; got num_inputs=0."
            )
        if _is_flexsn_triton_backend(backend):
            validated_example_inputs = _validate_scan_backend_contract(
                core, num_inputs, num_states, num_outputs, example_inputs
            )
            _validate_scan_backend_output_template_specs(
                self._explicit_output_template_specs,
                validated_example_inputs,
            )
        elif backend == "hop":
            if _flexsn_eager_scan is None:
                raise ImportError(
                    "FlexSN backend='hop' is unavailable: missing _flexsn_eager_scan."
                )
            if num_inputs + num_states == 0:
                raise ValueError("FlexSN requires at least one input or state tensor.")
            if num_inputs == 0:
                raise ValueError(
                    "FlexSN HOP backend requires at least one input sequence to derive T."
                )

        register_flexsn_kernel_handle = None

        if _is_flexsn_triton_backend(backend) and torch.cuda.is_available():
            self._inductor_scan_final_state_device = _first_cuda_device(
                example_inputs
            ) or torch.device("cuda", torch.cuda.current_device())
            try:
                from ..triton_kernel.flexsn.custom_ops import (
                    attach_flexsn_handle_finalizer,
                    register_flexsn_kernel_handle,
                )
                from ..triton_kernel.flexsn.kernel import (
                    build_inference_final_state_kernel,
                    build_inference_kernel,
                    build_training_kernels,
                )
            except (ImportError, RuntimeError) as e:
                logging.warning(
                    "FlexSN: could not import inductor kernel builders (%s); "
                    "falling back to eager_scan/flex_sn_scan for all paths." % e
                )
                build_inference_kernel = None
                build_inference_final_state_kernel = None
                build_training_kernels = None
                attach_flexsn_handle_finalizer = None
                register_flexsn_kernel_handle = None
            if build_inference_kernel is not None:
                try:
                    self._inductor_scan_kernel, self._inductor_scan_info = (
                        build_inference_kernel(
                            core,
                            num_inputs,
                            num_states,
                            num_outputs,
                            example_inputs=example_inputs,
                        )
                    )
                except Exception as e:
                    logging.warning(
                        "FlexSN: could not build inductor inference kernel (%s); "
                        "inference falls back to eager_scan." % e
                    )
                    self._inductor_scan_kernel = None
                    self._inductor_scan_info = None
                try:
                    (
                        self._inductor_scan_final_state_kernel,
                        self._inductor_scan_final_state_info,
                    ) = build_inference_final_state_kernel(
                        core,
                        num_inputs,
                        num_states,
                        num_outputs,
                        example_inputs=example_inputs,
                    )
                except Exception as e:
                    # Triton/CUDA/driver compilation failures surface through
                    # several exception types; any failure here can safely fall
                    # back to the already-built regular inference path.
                    logging.warning(
                        "FlexSN: could not build inductor inference-final-state kernel (%s: %s); "
                        "store_state_seqs=False inference falls back to the regular inference kernel."
                        % (type(e).__name__, e)
                    )
                    self._inductor_scan_final_state_kernel = None
                    self._inductor_scan_final_state_info = None
                try:
                    (
                        self._inductor_fwd_kernel,
                        self._inductor_bwd_kernel,
                        self._inductor_train_info,
                    ) = build_training_kernels(
                        core,
                        num_inputs,
                        num_states,
                        num_outputs,
                        example_inputs=example_inputs,
                        requires_grad=requires_grad,
                    )
                except Exception as e:
                    logging.warning(
                        "FlexSN: could not build inductor training kernels (%s); "
                        "training falls back to eager_scan." % e
                    )
                    self._inductor_fwd_kernel = None
                    self._inductor_bwd_kernel = None
                    self._inductor_train_info = None
            else:
                self._inductor_scan_kernel = None
                self._inductor_scan_info = None
                self._inductor_scan_final_state_kernel = None
                self._inductor_scan_final_state_info = None
                self._inductor_fwd_kernel = None
                self._inductor_bwd_kernel = None
                self._inductor_train_info = None
        else:
            self._inductor_scan_kernel = None
            self._inductor_scan_info = None
            self._inductor_scan_final_state_kernel = None
            self._inductor_scan_final_state_info = None
            self._inductor_scan_final_state_device = None
            self._inductor_fwd_kernel = None
            self._inductor_bwd_kernel = None
            self._inductor_train_info = None
        self._inductor_handle = None
        self._inductor_inference_available = (
            self._inductor_scan_kernel is not None
            and self._inductor_scan_info is not None
        )
        self._inductor_inference_final_state_available = (
            self._inductor_scan_final_state_kernel is not None
            and self._inductor_scan_final_state_info is not None
        )
        self._inductor_training_available = (
            self._inductor_fwd_kernel is not None
            and self._inductor_bwd_kernel is not None
            and self._inductor_train_info is not None
        )
        if (
            _is_flexsn_cuda_scan_backend(backend)
            and register_flexsn_kernel_handle is not None
            and (
                self._inductor_inference_available
                or self._inductor_inference_final_state_available
                or self._inductor_training_available
            )
        ):
            self._inductor_handle = register_flexsn_kernel_handle(
                inference_kernel=self._inductor_scan_kernel,
                inference_info=self._inductor_scan_info,
                inference_final_state_kernel=self._inductor_scan_final_state_kernel,
                inference_final_state_info=self._inductor_scan_final_state_info,
                forward_kernel=self._inductor_fwd_kernel,
                backward_kernel=self._inductor_bwd_kernel,
                training_info=self._inductor_train_info,
            )
            self._inductor_handle_finalizer = attach_flexsn_handle_finalizer(
                self, self._inductor_handle
            )
            if self._inductor_inference_final_state_available:
                try:
                    _warmup_inductor_inference_final_state_kernel(self)
                except Exception as e:
                    logging.warning(
                        "FlexSN: could not warm up inductor inference-final-state "
                        "kernel (%s: %s); falling back to the regular inference "
                        "kernel for store_state_seqs=False." % (type(e).__name__, e)
                    )
                    self._inductor_scan_final_state_kernel = None
                    self._inductor_scan_final_state_info = None
                    self._inductor_inference_final_state_available = False
        else:
            self._inductor_handle_finalizer = None

        # register states as memory buffers
        self.register_memory("states", None)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Triton/Inductor handle/kernel state is intentionally not propagated to the
        # copy: the compiled kernels reference the original module's ``core``,
        # and some kernel objects are not safely deep-copyable. The copy falls
        # back to the HOP/eager path with its own deep-copied ``core``.
        _inductor_skip_keys = {
            "_inductor_handle",
            "_inductor_handle_finalizer",
            "_inductor_inference_available",
            "_inductor_inference_final_state_available",
            "_inductor_training_available",
            "_inductor_scan_kernel",
            "_inductor_scan_info",
            "_inductor_scan_final_state_kernel",
            "_inductor_scan_final_state_info",
            "_inductor_fwd_kernel",
            "_inductor_bwd_kernel",
            "_inductor_train_info",
        }
        for key, value in self.__dict__.items():
            if key in _inductor_skip_keys:
                continue
            result.__dict__[key] = copy.deepcopy(value, memo)

        # Explicitly reset Triton/Inductor state on the copy.
        result._inductor_handle = None
        result._inductor_handle_finalizer = None
        result._inductor_inference_available = False
        result._inductor_inference_final_state_available = False
        result._inductor_training_available = False
        result._inductor_scan_kernel = None
        result._inductor_scan_info = None
        result._inductor_scan_final_state_kernel = None
        result._inductor_scan_final_state_info = None
        result._inductor_fwd_kernel = None
        result._inductor_bwd_kernel = None
        result._inductor_train_info = None

        return result

    @property
    def kernel(self):
        return self._kernel_accessor

    def _kernel_accessor(self, *args):
        flat_args = list(args)
        expected = self.num_inputs + self.num_states
        if len(flat_args) != expected:
            raise ValueError(
                "FlexSN.kernel expected "
                f"{expected} tensors "
                f"({self.num_inputs} input sequences + "
                f"{self.num_states} initial states), "
                f"but got {len(flat_args)}."
            )
        bad_arg = next(
            (
                (i, type(arg).__name__)
                for i, arg in enumerate(flat_args)
                if not isinstance(arg, torch.Tensor)
            ),
            None,
        )
        if bad_arg is not None:
            index, type_name = bad_arg
            raise TypeError(
                f"FlexSN.kernel expected tensor arguments, but arg #{index} is "
                f"{type_name}."
            )

        all_cuda, same_device = _flat_args_on_single_cuda_device(flat_args)
        if self._inductor_handle is None or not all_cuda or not same_device:
            raise RuntimeError(
                "FlexSN.kernel is unavailable: FlexSN Triton kernels are not "
                "ready, or inputs are not CUDA tensors on a single device."
            )

        use_training = torch.is_grad_enabled() and (
            _core_requires_grad(self.core)
            or any(tensor.requires_grad for tensor in flat_args)
        )
        if use_training:
            if not self._inductor_training_available:
                raise RuntimeError(
                    "FlexSN.kernel training path is unavailable for the current "
                    "Triton/Inductor handle."
                )
            return tuple(
                flexsn_inductor_training(self._inductor_handle, flat_args)[
                    : self.num_outputs + self.num_states
                ]
            )

        if not self._inductor_inference_available:
            raise RuntimeError(
                "FlexSN.kernel inference path is unavailable for the current "
                "Triton/Inductor handle."
            )
        return tuple(flexsn_inductor_inference(self._inductor_handle, flat_args))

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        if _is_flexsn_triton_backend(value) and self.step_mode != "m":
            raise RuntimeError(
                f"Cannot set backend={value!r} when step_mode={self.step_mode!r}; "
                f"Triton/Inductor backends require step_mode='m'."
            )
        if not _is_flexsn_triton_backend(value):
            base.check_backend_library(value)
        elif "_inductor_handle" in self.__dict__ and self._inductor_handle is None:
            logging.warning(
                "Switching FlexSN.backend to %s without prebuilt Triton kernels; "
                "this module will fall back to the HOP/eager path.",
                value,
            )
        self._backend = value

    @property
    def supported_backends(self):
        return ("triton", "torch", "inductor", "hop")

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
            if args[0].shape[0] > 0:
                return [torch.zeros_like(args[0][0]) for _ in range(num_states)]
            return [args[0].new_zeros(args[0].shape[1:]) for _ in range(num_states)]
        else:
            raise ValueError(f"Unsupported step mode: {step_mode}")

    def single_step_forward(self, *args):
        # only torch backend is supported for single-step forward
        results = self.core(*args, *self.states)  # [*outputs, *states]
        self.states = results[self.num_outputs :]
        return results[: self.num_outputs]

    def multi_step_forward(self, *args):
        T = args[0].shape[0]
        if T == 0:
            if self.backend not in self.supported_backends:
                raise ValueError(f"Unsupported backend: {self.backend}")
            if self.states is None:
                self.states = self.init_states(self.num_states, self.step_mode, *args)
            if self.backend == "hop":
                state_args = [state.contiguous() for state in self.states]
                result_seqs = _run_hop_scan(
                    self.core,
                    self.num_inputs,
                    self.num_states,
                    self.num_outputs,
                    self.store_state_seqs,
                    *args,
                    *state_args,
                    output_template_specs=self._output_template_specs,
                )
                output_seqs = list(result_seqs[: self.num_outputs])
                state_results = list(result_seqs[self.num_outputs :])
                if self.store_state_seqs:
                    self.state_seqs = state_results
                    self.states = [
                        _last_state_or_current(v, self.states[i])
                        for i, v in enumerate(state_results)
                    ]
                else:
                    self.states = state_results
                return output_seqs
            if (
                self.backend == "torch"
                and self.num_outputs > 0
                and self._output_template_specs is None
            ):
                raise ValueError(
                    f"FlexSN backend='{self.backend}' requires example_outputs "
                    "for empty multi-step inputs so output shapes and dtypes "
                    "match core's per-step return contract without executing core."
                )
            output_seqs = _empty_multistep_outputs(
                args,
                self.states,
                self.num_outputs,
                self._output_template_specs,
                use_template_device=False,
            )
            if self.store_state_seqs:
                self.state_seqs = [s.new_empty((0, *s.shape)) for s in self.states]
            return output_seqs

        if self.backend == "torch":
            if self.states is None:
                self.states = self.init_states(self.num_states, self.step_mode, *args)
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

        elif self.backend == "hop":
            if self.states is None:
                self.states = self.init_states(self.num_states, self.step_mode, *args)
            state_args = [state.contiguous() for state in self.states]
            result_seqs = _run_hop_scan(
                self.core,
                self.num_inputs,
                self.num_states,
                self.num_outputs,
                self.store_state_seqs,
                *args,
                *state_args,
                output_template_specs=self._output_template_specs,
            )
            output_seqs = list(result_seqs[: self.num_outputs])
            state_results = list(result_seqs[self.num_outputs :])
            if self.store_state_seqs:
                state_seqs = state_results
                self.states = [
                    _last_state_or_current(v, self.states[i])
                    for i, v in enumerate(state_seqs)
                ]
                self.state_seqs = state_seqs
            else:
                self.states = state_results
            return output_seqs

        elif _is_flexsn_triton_backend(self.backend):
            result_has_state_seqs = self.store_state_seqs
            can_elide_zero_states = (
                self.states is None and _can_elide_zero_state_inputs(self)
            )
            # The first init_states branch handles the case where zero-state elision
            # is impossible. _no_grad then determines whether use_implicit_zero_states
            # can skip explicit state tensors entirely. The second init_states branch
            # only runs if self.states is still None, so re-initialization cannot occur.
            if self.states is None and not can_elide_zero_states:
                self.states = self.init_states(self.num_states, self.step_mode, *args)
            _no_grad = not torch.is_grad_enabled() or (
                not _value_requires_grad(args)
                and not _value_requires_grad(self.states)
                and not _core_requires_grad(self.core)
            )
            use_implicit_zero_states = can_elide_zero_states and _no_grad
            if self.states is None and not use_implicit_zero_states:
                self.states = self.init_states(self.num_states, self.step_mode, *args)
            state_args = [] if use_implicit_zero_states else list(self.states)
            flat_args = [*args, *state_args]
            all_cuda, same_device = _flat_args_on_single_cuda_device(flat_args)
            if self._inductor_handle is not None and all_cuda and same_device:
                if _no_grad:
                    if (
                        not self.store_state_seqs
                        and self._inductor_inference_final_state_available
                    ):
                        result_seqs = flexsn_inductor_inference_final_state(
                            self._inductor_handle, flat_args
                        )
                        output_seqs = list(result_seqs[: self.num_outputs])
                        self.states = list(result_seqs[self.num_outputs :])
                        return output_seqs
                    elif self._inductor_inference_available:
                        result_seqs = flexsn_inductor_inference(
                            self._inductor_handle, flat_args
                        )
                        result_has_state_seqs = True
                    else:
                        result_seqs = None
                elif (not _no_grad) and self._inductor_training_available:
                    if not self.store_state_seqs:
                        result_seqs = flexsn_inductor_training_final_state(
                            self._inductor_handle, flat_args
                        )
                        output_seqs = list(result_seqs[: self.num_outputs])
                        self.states = list(
                            result_seqs[
                                self.num_outputs : self.num_outputs + self.num_states
                            ]
                        )
                        return output_seqs
                    result_seqs = flexsn_inductor_training(
                        self._inductor_handle, flat_args
                    )
                    result_seqs = result_seqs[: self.num_outputs + self.num_states]
                else:
                    result_seqs = None
            else:
                result_seqs = None

            if result_seqs is None:
                if self.states is None:
                    self.states = self.init_states(
                        self.num_states, self.step_mode, *args
                    )
                state_args = [state.contiguous() for state in self.states]
                result_seqs = _run_hop_scan(
                    self.core,
                    self.num_inputs,
                    self.num_states,
                    self.num_outputs,
                    self.store_state_seqs,
                    *args,
                    *state_args,
                    output_template_specs=self._output_template_specs,
                )
                result_has_state_seqs = self.store_state_seqs
            output_seqs = list(result_seqs[: self.num_outputs])
            state_seqs = list(result_seqs[self.num_outputs :])
            if result_has_state_seqs:
                if self.states is None:
                    self.states = self.init_states(
                        self.num_states, self.step_mode, *args
                    )
                self.states = [
                    _last_state_or_current(v, self.states[i])
                    for i, v in enumerate(state_seqs)
                ]
                if self.store_state_seqs:
                    self.state_seqs = state_seqs
            else:
                self.states = state_seqs
            return output_seqs

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def forward(self, *args):
        can_elide = (
            self.step_mode == "m"
            and _can_elide_zero_state_inputs(self)
            and (
                not torch.is_grad_enabled()
                or (
                    not _value_requires_grad(args)
                    and not _core_requires_grad(self.core)
                )
            )
        )
        if self.states is None and not can_elide:
            self.states = self.init_states(self.num_states, self.step_mode, *args)
        output = super().forward(*args)
        return output[0] if len(output) == 1 else output

    def extra_repr(self):
        core_name = getattr(self.core, "__name__", type(self.core).__name__)
        if isinstance(self.core, functools.partial):
            core_name = f"partial({getattr(self.core.func, '__name__', type(self.core.func).__name__)})"
        return (
            f"core={core_name}, "
            f"num_inputs={self.num_inputs}, "
            f"num_states={self.num_states}, "
            f"num_outputs={self.num_outputs}, "
            f"step_mode={self.step_mode!r}, "
            f"backend={self.backend!r}"
        )
