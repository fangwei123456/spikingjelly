from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
import logging

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.auto_cuda.ss_neuronal_kernel: {e}')
    cupy = None
    

from .. import cuda_utils, surrogate
from ... import configure
from typing import Callable, Iterable
from . import base, cfunction
import math

def if_requires_grad(items: Iterable):
    requires_grad = False
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                requires_grad = True
                break

    return requires_grad

def scalar_to_cupy(py_dict: dict, ref: str = 'x'):
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

def new_tensors(news: tuple, py_dict: dict, ref: str = 'x'):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype),ref.shape[0])):
        py_dict[news[i]] = item


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

class NeuronFPKernel(base.CKernel1D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}')
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='x')
        self.add_param(ctype=f'const {dtype} *', cname='v')
        self.add_param(ctype=f'{dtype} *', cname='h')
        self.add_param(ctype=f'{dtype} *', cname='v_next')
        self.add_param(ctype=f'{dtype} *', cname='spike')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    def neuronal_charge(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`H = f(X, V, ...)`.

        This function should define how ``h`` is calculated by ``x[index], v[index]`` and other params if
        the neuron needs.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def neuronal_charge(self) -> str:
                return cfunction.add(z='h[index]', x='x[index]', y='v[index]', dtype=self.dtype)
        """
        return '// neuronal_charge should be defined here!'
    
    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(neuronal_fire(spike='spike[index]', v='h[index]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_next[index]', h='h[index]', spike='spike[index]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(
                neuronal_soft_reset(v_next='v_next[index]', h='h[index]', spike='spike[index]', v_th='v_th',
                                    dtype=self.dtype))

        self._core = core_codes.codes
        return self._core

class NeuronBPKernel(base.CKernel1D):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}_{"detach_reset" if detach_reset else "nodetach_reset"}')
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='grad_spike')
        self.add_param(ctype=f'const {dtype} *', cname='grad_v_next')
        self.add_param(ctype=f'const {dtype} *', cname='h')
        self.add_param(ctype=f'{dtype} *', cname='grad_x')
        self.add_param(ctype=f'{dtype} *', cname='grad_v')
        self.add_param(ctype=f'{dtype} &', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    @property
    def post_core(self):

        codes = base.CodeTyper(16)
        codes.append(self.grad_h_next_to_v())
        codes.append(cfunction.mul(z='grad_v[index]', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        self._post_core = codes.codes
        return self._post_core

    def grad_h_to_v(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H}{\\mathrm{d} V}`.

        This function should define how ``grad_h_to_v`` is calculated. Note that ``grad_h_to_v`` has not been
        declared. Thus, this function should also declare ``grad_h_to_v``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_to_v(self) -> str:
                return cfunction.constant(y=f'const {self.dtype} grad_h_to_v', x=1., dtype=self.dtype)
        """
        return '// grad_h_to_v should be defined here!'


    def grad_h_to_x(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t]}{\\mathrm{d} X[t]}`.

        This function should define how ``grad_h_to_x`` is calculated. Note that ``grad_h_to_x`` has not been
        declared. Thus, this function should also declare ``grad_h_to_x``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_to_x(self) -> str:
                return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        """
        return '// grad_h_to_x should be defined here!'

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h[index]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_next_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                              y='spike', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h[index]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_next_to_h', x='temp_var', y='grad_v_next_to_h', dtype=self.dtype))


        else:
            core_codes.append(f'{self.dtype} grad_v_next_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_next_to_h', x='grad_v_next_to_h', y='temp_var', dtype=self.dtype))

        core_codes.append(cfunction.mul(z=f'{self.dtype} grad_h', x='grad_s_to_h', y='grad_spike[index]', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x=cfunction.mul(z=None, x='grad_v_next[index]', y='grad_v_next_to_h', dtype=self.dtype), y='grad_h', dtype=self.dtype))

        core_codes.append(self.grad_h_to_v())
        core_codes.append(cfunction.mul(z='grad_v[index]', x='grad_h', y='grad_h_to_v', dtype=self.dtype))

        core_codes.append(self.grad_h_to_x())
        core_codes.append(cfunction.mul(z='grad_x[index]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        self._core = core_codes.codes
        return self._core  
    
class NeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        """
        :param py_dict: a dict built from the neuron's forward autograd function. It should at least contain ``x, v, v_reset``
        :type py_dict: dict
        :return: requires_grad, blocks, threads, py_dict

            requires_grad: bool
                if any tensor in ``py_dict`` requires grad, then ``requires_grad = True``;else ``requires_grad = False``

            blocks: int
                CUDA param used in calling CUDA kernel

            threads: int
                CUDA param used in calling CUDA kernel. The default value is ``spikingjelly.configure.cuda_threads``

            py_dict: dict
                Compared with the input ``py_dict``, the returned ``py_dict`` will:

                    * convert all ``float/int`` scalars in ``py_dict`` to ``cupy.ndarray``

                    * add ``h, spike, v_next`` to ``py_dict``. They are zero tensors
                      with the same shape with ``x`` or ``v``.

                    * add ``numel`` to ``py_dict``. Note that ``x.shape = [numel]``.
                      A specific case is that ``x.dtype == torch.half``, then ``numel = math.ceil(numel / 2)``.
                      Note that ``numel`` in the returned ``py_dict`` is ``cupy.ndarray``


        :rtype: tuple
        """
        device = py_dict['x'].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        scalar_to_cupy(py_dict)

        new_tensors(('h', 'spike', 'v_next'), py_dict)
        numel = py_dict['x'].numel()
        threads = configure.cuda_threads
        if py_dict['x'].dtype == torch.float16:
            # we will take two neurons to calculate as one neuron in cuda half2
            # pad will be implemented by the kernel.__call__
            numel = math.ceil(numel / 2)

        blocks = cuda_utils.cal_blocks(numel)

        with cuda_utils.DeviceEnvironment(device):
            numel = cupy.asarray(numel)

        py_dict['numel'] = numel

        return requires_grad, blocks, threads, py_dict

    @staticmethod
    def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param requires_grad: if any tensor in forward params requires grad
        :type requires_grad: bool
        :param args: tensors that need to be saved by ``ctx.save_for_backward``
        :param kwargs: items that need to be saved by ``ctx.xx = xx``

        Saves ``*args, **kwargs`` in ``ctx`` by ``ctx.save_for_backward(*args)`` and ``ctx.xx = xx`` for all ``xx`` in ``kwargs.items()``.
        """
        if requires_grad:
            ctx.save_for_backward(*args)
            for key, value in kwargs.items():
                ctx.__setattr__(key, value)



    @staticmethod
    def pre_backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param grad_spike: gradients of ``spike``
        :type grad_spike: torch.Tensor
        :param grad_v_next: gradients of ``v_next``
        :type grad_v_next: torch.Tensor
        :return: backward_kernel, blocks, threads, py_dict

            backward_kernel: NeuronBPTTKernel
                The CUDA kernel used for backward. It should be provided in ``ctx.backward_kernel``

            blocks: int
                CUDA param used in calling CUDA kernel. It should be provided in ``ctx.blocks``

            threads: int
                CUDA param used in calling CUDA kernel. It should be provided in ``ctx.threads``
        :rtype: tuple
        """
        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads

        h = ctx.saved_tensors[0]
        numel = ctx.numel
        v_th = ctx.v_th
        v_reset = ctx.v_reset

        zero_shape = list(grad_spike.shape)
        zero_shape[0] *= 2
        zero_data = torch.zeros(zero_shape, device=grad_spike.device, dtype=grad_spike.dtype)

        # For fp16, ctx.numel will be divided by 2 later. Here is a reliable way to divide tensor equally
        real_numel = grad_spike.size(0)
        grad_x = zero_data[:real_numel]
        grad_v = zero_data[real_numel:]

        py_dict = {
            'numel': numel,
            'grad_spike': grad_spike,
            'grad_v_next': grad_v_next,
            'h': h,
            'grad_x': grad_x,
            'grad_v': grad_v,
            'v_th': v_th,
            'v_reset': v_reset
        }

        return backward_kernel, blocks, threads, py_dict
    
class IFNodeFPKernel(NeuronFPKernel):
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h[index]', x='x[index]', y='v[index]', dtype=self.dtype)

class IFNodeBPKernel(NeuronBPKernel):
    def grad_h_to_v(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_v', x=1., dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
    
class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, v_th: float, v_reset: Optional[float],
                forward_kernel: IFNodeFPKernel, backward_kernel: IFNodeBPKernel):
        py_dict = {
            'x': x,
            'v': v,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)


        return py_dict['spike'], py_dict['v_next']

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike, grad_v_next)
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['grad_x'], py_dict['grad_v'], None, None, None, None
    
class LIFNodeFPKernel(NeuronFPKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')


    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPKernel_temp_var', x='v[index]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPKernel_temp_var = v[index];'

        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPKernel_temp_var', x='x[index]', y='LIFNodeFPKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPKernel_temp_var', x='decay', y='LIFNodeFPKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPKernel_temp_var', x='decay', y='LIFNodeFPKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPKernel_temp_var', x='x[index]', y='LIFNodeFPKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h[index]', x='LIFNodeFPKernel_temp_var', y='v[index]', dtype=self.dtype)

        return codes



class LIFNodeBPKernel(NeuronBPKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def grad_h_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay;'
        
class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, v_th: float, v_reset: Optional[float], decay: float,
                forward_kernel: LIFNodeFPKernel, backward_kernel: LIFNodeBPKernel):
        py_dict = {
            'x': x,
            'v': v,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])


        return py_dict['spike'], py_dict['v_next']

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike, grad_v_next)
        py_dict['decay'] = ctx.decay

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x'], py_dict['grad_v'], None, None, None, None, None