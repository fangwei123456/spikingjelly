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
from typing import Callable, Iterable
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


class NeuronFPTTKernel(base.CKernel2D):
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
            core_codes.append(
                neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th',
                                    dtype=self.dtype))

        return core_codes.codes


class NeuronBPTTKernel(base.CKernel2D):
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







    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        if self.dtype == 'float':
            codes.append('float grad_h = 0.0f;')
        elif self.dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
        else:
            raise NotImplementedError(self.dtype)

        return codes.codes


    @property
    def post_core(self):

        codes = base.CodeTyper(16)
        codes.append(self.grad_h_to_x())
        codes.append(cfunction.mul(z='grad_v_init[index]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))
        return codes.codes

    def grad_h_next_to_v(self) -> str:
        raise NotImplementedError

    def grad_h_to_x(self) -> str:
        raise NotImplementedError

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike_seq_t', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
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

        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(cfunction.mul(z=f'grad_h', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_v_to_h', dtype=self.dtype))
        with base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))

        core_codes.append(self.grad_h_to_x())
        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        return core_codes.codes


class IFNodeFPTTKernel(NeuronFPTTKernel):
    @property
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)


class IFNodeBPTTKernel(NeuronBPTTKernel):
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)



def if_requires_grad(items: Iterable):
    requires_grad = False
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                requires_grad = True
                break

    return requires_grad


def pad_half2(py_dict: dict, ref: str = 'v_init'):
    if py_dict[ref].numel() % 2 == 0:
        return False

    for key, value in py_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float16:
                assert value.dim() <= 2
                value = F.pad(value, (0, 1))
                py_dict[key] = value

    return True


def scalar_to_cupy(py_dict: dict, ref: str = 'x_seq'):
    device = py_dict[ref].get_device()
    dtype = py_dict[ref].dtype

    with cuda_utils.DeviceEnvironment(device):
        for key, value in py_dict.items():
            if isinstance(value, float):
                if dtype == torch.float32:
                    value = cupy.asarray(value, dtype=np.float32)
                elif dtype == torch.float16:
                    value = cupy.asarray([value, value], dtype=np.float16)
                else:
                    raise NotImplementedError(dtype)
                py_dict[key] = value

            elif isinstance(value, int):
                py_dict[key] = cupy.asarray(value)

def new_tensors(news: tuple, py_dict: dict, ref: str = 'x_seq'):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype),ref.shape[0])):
        py_dict[news[i]] = item

class NeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        device = py_dict['x_seq'].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        scalar_to_cupy(py_dict)
        use_pad = pad_half2(py_dict)
        new_tensors(('h_seq', 'spike_seq', 'v_seq'), py_dict)
        py_dict['v_v_seq'] = torch.cat((py_dict.pop('v_init').unsqueeze(0), py_dict.pop('v_seq')))
        numel = py_dict['x_seq'].numel()
        N = py_dict['x_seq'].shape[1]
        threads = configure.cuda_threads
        if py_dict['x_seq'].dtype == torch.float16:
            assert N % 2 == 0
            # we will take two neurons to calculate as one neuron in cuda half2
            N = N // 2
            numel = numel // 2

        blocks = cuda_utils.cal_blocks(N)

        with cuda_utils.DeviceEnvironment(device):
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)

        py_dict['numel'] = numel
        py_dict['N'] = N

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        return requires_grad, use_pad, blocks, threads, py_dict

    @staticmethod
    def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
        if requires_grad:
            ctx.save_for_backward(*args)
            for key, value in kwargs.items():
                ctx.__setattr__(key, value)



    @staticmethod
    def pre_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads

        h_seq = ctx.saved_tensors[0]
        use_pad = ctx.use_pad
        numel = ctx.numel
        N = ctx.N
        v_th = ctx.v_th
        v_reset = ctx.v_reset

        if use_pad:
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        py_dict = {
            'numel': numel,
            'N': N,
            'grad_spike_seq': grad_spike_seq,
            'grad_v_seq': grad_v_seq,
            'h_seq': h_seq,
            'grad_x_seq': grad_x_seq,
            'grad_v_init': grad_v_init,
            'v_th': v_th,
        }
        if v_reset is not None:
            py_dict['v_reset'] = v_reset

        return backward_kernel, blocks, threads, py_dict


class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, use_pad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        forward_kernel((blocks,), (threads,), py_dict)

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], use_pad=use_pad, blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)

        if use_pad:
            return py_dict['spike_seq'][..., :-1], py_dict['v_v_seq'][1:, ..., :-1]
        else:
            return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        backward_kernel((blocks,), (threads,), py_dict)

        if ctx.use_pad:
            return py_dict['grad_x_seq'][..., :-1], py_dict['grad_v_init'][..., :-1], None, None, None, None
        else:
            return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None


class LIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    @property
    def neuronal_charge(self) -> str:
        codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes



class LIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay;'



class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, decay: float,
                forward_kernel: LIFNodeFPTTKernel, backward_kernel: LIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, use_pad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        forward_kernel((blocks,), (threads,), py_dict)

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], use_pad=use_pad, blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])

        if use_pad:
            return py_dict['spike_seq'][..., :-1], py_dict['v_v_seq'][1:, ..., :-1]
        else:
            return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay
        backward_kernel((blocks,), (threads,), py_dict)


        if ctx.use_pad:
            return py_dict['grad_x_seq'][..., :-1], py_dict['grad_v_init'][..., :-1], None, None, None, None, None
        else:
            return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None, None




