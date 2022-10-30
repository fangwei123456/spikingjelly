import torch
import torch.nn.functional as F
import numpy as np
import logging

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.auto_cuda.neuronal_kernel: {e}')
    cupy = None
    pass

from .. import cuda_utils, surrogate
from ... import configure
from typing import Callable
from . import base, cfunction


def neuronal_hard_reset(v_next: str, h: str, spike: str, v_reset: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} * (1.0f - {spike}) + {v_reset} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), {spike}), __hmul2(v_reset, {spike}));'
    else:
        raise NotImplementedError(dtype)


def neuronal_soft_reset(v_next: str, h: str, spike: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} - {v_th} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));'
    else:
        raise NotImplementedError(dtype)


def neuronal_fire(spike: str, v: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return cfunction.heaviside(y=spike, x=f'({v} - {v_th})', dtype=dtype)
    elif dtype == 'half2':
        return cfunction.heaviside(y=spike, x=f'__hsub2({v}, {v_th})', dtype=dtype)
    else:
        raise NotImplementedError(dtype)


class NeuronFPTT(base.CKernel2D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}',
            reverse=False)
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='x_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'{dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='spike_seq')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    @property
    def neuronal_charge(self) -> str:
        # e.g., for IFNode, this function shoule return:
        #   cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=dtype)
        raise NotImplementedError

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge)

        core_codes.append(neuronal_fire(spike='spike_seq[t]', v='h_seq[t]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th',
                                                  dtype=self.dtype))

        self._core = core_codes.codes
        return self._core


class NeuronBPTT(base.CKernel2D):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}_{"detach_reset" if detach_reset else "nodetach_reset"}',
            reverse=True)
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='grad_spike_seq')
        self.add_param(ctype=f'const {dtype} *', cname='grad_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_x_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_v_init')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

        codes = base.CodeTyper(16)
        if dtype == 'float':
            codes.append('float grad_h = 0.0f;')
        elif dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
        else:
            raise NotImplementedError(dtype)

        self.pre_core = codes.codes

        codes = base.CodeTyper(16)
        codes.append(cfunction.mul(z='grad_v_init[index]', x='grad_h', y=self.grad_h_to_x, dtype=self.dtype))
        self.post_core = codes.codes

    @property
    def grad_h_next_to_v(self) -> str:
        raise NotImplementedError

    @property
    def grad_h_to_x(self) -> str:
        raise NotImplementedError

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike_seq_t', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                                            y='spike_seq_t', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))


        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_to_h', x='grad_v_to_h', y='temp_var', dtype=self.dtype))

        core_codes.append(cfunction.mul(z=f'grad_h', x='grad_h', y=self.grad_h_next_to_v, dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_v_to_h', dtype=self.dtype))
        with base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))

        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y=self.grad_h_to_x, dtype=self.dtype))

        self._core = core_codes.codes
        return self._core


class IFNodeFPTT(NeuronFPTT):
    @property
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)


class IFNodeBPTT(NeuronBPTT):
    @property
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(y=None, x=1., dtype=self.dtype)

    @property
    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=None, x=1., dtype=self.dtype)


def as_cp_array(x: float or int, dtype: str = None):
    if isinstance(x, int):
        return cupy.asarray(x)
    elif isinstance(x, float):
        if dtype == 'float':
            return cupy.asarray(x, dtype=np.float32)
        elif dtype == 'half2':
            return cupy.asarray([x, x], dtype=np.half)

def neuron_kernel_py_param_pre_processing(*args):
    requires_grad = False
    for item in args:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                requires_grad = True
                break

    device = None
    dtype = None

    for item in args:
        if isinstance(item, torch.Tensor):
            device = item.get_device()
            if item.dtype == torch.float32:
                dtype = 'float'
            elif item.dtype == torch.float16:
                dtype = 'half2'

    use_pad = False
    if dtype == 'half2':
        for item in args:
            if isinstance(item, torch.Tensor):
                if item.dim() == 2 and item.shape[1] % 2 != 0:
                    # shape = [T, N] and N % 2 != 0
                    use_pad = True
                    break

    ret = []
    for item in args:
        if isinstance(item, torch.Tensor):
            assert item.get_device() == device
            if dtype == 'float':
                assert item.dtype == torch.float32
            elif dtype == 'half2':
                assert item.dtype == torch.float16

            assert item.dim() <= 2
            if use_pad:
                item = F.pad(item, (0, 1))  # [T, N] -> [T, N + 1]


        elif isinstance(item, (float, int)):
            with cuda_utils.DeviceEnvironment(device):
                item = as_cp_array(item, dtype)

        else:
            pass

        ret.append(item)
    return requires_grad, device, dtype, use_pad, ret


class IFNodePTT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, forward_kernel: IFNodeFPTT, backward_kernel: IFNodeBPTT):

        requires_grad, device, dtype, use_pad, ret = neuron_kernel_py_param_pre_processing(
            x_seq, v_init, v_th, v_reset)

        x_seq, v_init, v_th, v_reset = ret

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            N = x_seq.shape[1]

            threads = configure.cuda_threads
            if dtype == 'half2':
                assert N % 2 == 0
                # we will take two neurons to calculate as one neuron in cuda half2
                N = N // 2
                numel = numel // 2

            blocks = cuda_utils.cal_blocks(N)
            print(forward_kernel.full_codes)
            kernel = cupy.RawKernel(forward_kernel.full_codes, forward_kernel.kernel_name, options=configure.cuda_compiler_options,
                           backend=configure.cuda_compiler_backend)

            numel = as_cp_array(numel)
            N = as_cp_array(N)

            if v_reset is not None:
                numel, N, x_seq, v_v_seq, h_seq, spike_seq, v_th, v_reset = cuda_utils.get_contiguous(numel, N, x_seq, v_v_seq, h_seq, spike_seq, v_th, v_reset)
                kernel_args = (numel, N, x_seq, v_v_seq, h_seq, spike_seq, v_th, v_reset)
            else:
                numel, N, x_seq, v_v_seq, h_seq, spike_seq, v_th = cuda_utils.get_contiguous(numel, N, x_seq,
                                                                                                      v_v_seq, h_seq,
                                                                                                      spike_seq, v_th)
                kernel_args = (numel, N, x_seq, v_v_seq, h_seq, spike_seq, v_th)


            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.save_for_backward(h_seq)

            ctx.use_pad = use_pad
            ctx.backward_kernel = backward_kernel

            ctx.blocks = blocks
            ctx.threads = threads

            ctx.numel = numel
            ctx.N = N
            ctx.v_th = v_th
            ctx.v_reset = v_reset

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        h_seq = ctx.saved_tensors[0]

        use_pad = ctx.use_pad
        backward_kernel = ctx.backward_kernel

        blocks = ctx.blocks
        threads = ctx.threads

        numel = ctx.numel
        N = ctx.N
        v_th = ctx.v_th
        v_reset = ctx.v_reset


        if use_pad:
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]



        with cuda_utils.DeviceEnvironment(device):
            print(backward_kernel.full_codes)

            kernel = cupy.RawKernel(backward_kernel.full_codes, backward_kernel.kernel_name,
                                    options=configure.cuda_compiler_options,
                                    backend=configure.cuda_compiler_backend)
            if v_reset is not None:
                numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th, v_reset = cuda_utils.get_contiguous(numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th, v_reset)
                kernel_args = [numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th, v_reset]
            else:
                numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th = cuda_utils.get_contiguous(
                    numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th)
                kernel_args = [numel, N, grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, grad_v_init, v_th]

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], None, None, None, None
        else:
            return grad_x_seq, grad_v_init, None, None, None, None
