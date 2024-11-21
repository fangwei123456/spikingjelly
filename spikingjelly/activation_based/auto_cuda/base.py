import numpy as np
import logging
try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.auto_cuda.base: {e}')
    cupy = None

import torch
import torch.nn.functional as F
import sys
import logging
from .. import cuda_utils
from ... import configure


def wrap_with_comment(code: str, comment: str):
    if logging.DEBUG >= logging.root.level:
        return '\n//------' + comment + ' start------\n' + code + '\n//------' + comment + ' end--------\n\n'
    else:
        return code


def startswiths(x: str, prefixes: tuple):
    ret = False
    for prefix in prefixes:
        if x.startswith(prefix):
            ret = True

    return ret

class CKernel:
    def __init__(self, kernel_name: str):
        """
        :param kernel_name: the name of kernel
        :type kernel_name: str

        The base python class for simplifying the using of custom CUDA kernel.

        Some critical attributes:

            cparams:
                a dict for saving parameters name and type.

            reserved_cnames:
                a list for saving reserved variables names, which can not be used to name variable again.


        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base

            example_ck = base.CKernel(kernel_name='example_ck')
            print(example_ck.full_codes)

        The outputs are:

        .. code-block:: c++

            #include <cuda_fp16.h>
            extern "C" __global__
            void example_ck(
            )
            {}



        A ``CKernel`` is composed of three parts: declaration, head, core, and tail.
        When setting ``logging level <= DEBUG``, some debug information will be added to cuda codes or printed.
        And we can check where is each part.
        Here is an example:

        .. code-block:: python

            import logging
            logging.basicConfig(level=logging.DEBUG)
            from spikingjelly.activation_based.auto_cuda import base

            example_ck = base.CKernel(kernel_name='example_ck')
            print(example_ck.full_codes)

        The outputs are:

        .. code-block:: c++

            //------declaration start------

            #include <cuda_fp16.h>
            extern "C" __global__
            void example_ck(
            )

            //------declaration end--------


            //------head start------
            {
            //------head end--------


            //------core start------

            //------core end--------


            //------tail start------
            }
            //------tail end--------

        In most cases, ``CKernel`` is used as a base class. Refer to :class:`CKernel1D <spikingjelly.activation_based.auto_cuda.base.CKernel1D>` and :class:`CKernel2D <spikingjelly.activation_based.auto_cuda.base.CKernel2D>` for more details.
        """
        self.cparams = {}
        self.reserved_cnames = []
        self.kernel_name = kernel_name
        self._core = ''

    def check_attributes(self, **kwargs):
        """
        :param kwargs: a dict of attributes
        :type kwargs: dict
        :return: if all ``value`` in ``kwargs[key]`` is identical to ``self.__getattribute__(key)``
        :rtype: bool

        This function can be used to check if a ``CKernel`` is changed by if any of its attributes changes.
        """
        for key, value in kwargs.items():
            if value != self.__getattribute__(key):
                return False

        else:
            return True


    @property
    def core(self):
        return self._core

    @core.setter
    def core(self, value):
        self._core = value

    def set_contiguous(self, py_dict: dict):
        """
        :param py_dict: a dict whose value is ``torch.Tensor`` or ``cupy.ndarray``
        :type py_dict: dict

        Check if all values in py_dict are ``torch.Tensor`` or ``cupy.ndarray`` and contiguous.
        If not, this function will raise an error.
        """
        # get contiguous
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.contiguous()


            elif isinstance(value, cupy.ndarray):
                value = cupy.ascontiguousarray(value)
            else:
                raise TypeError(type(value))

            py_dict[key] = value

    def get_device(self, py_dict: dict) -> int:
        """
        :param py_dict: a dict
        :type py_dict: dict

        Traverse the dict and return the device id of the first met ``torch.Tensor``.
        If no ``torch.Tensor`` in ``py_dict``, this function will raise an error.
        """

        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                return item.get_device()

            elif isinstance(item, cupy.ndarray):
                return item.device.id

        raise ValueError

    def check_device(self, device: int, py_dict: dict):
        """
        :param device: the cuda device id
        :type device: int
        :param py_dict: a dict
        :type py_dict: dict

        Check if the device id of each ``torch.Tensor`` or ``cupy.ndarray`` in py_dict is identical to ``device``.
        If not, this function will raise an error.
        """
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                assert item.get_device() == device

            elif isinstance(item, cupy.ndarray):
                assert item.device.id == device

    def check_keys(self, py_dict: dict):
        """
        :param py_dict: a dict
        :type py_dict: dict

        Check if keys of ``py_dict`` are identical to keys of ``self.cparams``.
        If not, this function will raise an error.
        """
        if py_dict.keys() != self.cparams.keys():
            missed_keys = (py_dict.keys() | self.cparams.keys()) - (py_dict.keys() & self.cparams.keys())

            if missed_keys.__len__() > 0:
                if (missed_keys & py_dict.keys()).__len__() > 0:
                    msg = f'{missed_keys} is in py_dict but not in cparams!'
                else:
                    msg = f'{missed_keys} is in cparams but not in py_dict!'
                raise ValueError(msg)

    def check_ctypes(self, py_dict: dict):
        """
        :param py_dict: a dict
        :type py_dict: dict

        Check if the value in ``py_dict`` has the corresponding ``ctype`` in ``self.cparams``, which includes:

        ``torch.float`` or ``np.float32``------ ``'const float'`` or ``'float'``

        ``torch.half`` or ``np.float16`` ------ ``'const half2'`` or ``'half2'``

        ``np.int_``  ------  ``'const int'`` or ``'int'``

        If not, this function will raise an error.
        """
        for key, value in py_dict.items():
            ctype: str = self.cparams[key]
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float:
                    assert startswiths(ctype, ('const float', 'float'))

                elif value.dtype == torch.half:
                    assert startswiths(ctype, ('const half2', 'half2'))

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float32:
                    assert startswiths(ctype, ('const float', 'float'))

                elif value.dtype == np.float16:
                    assert startswiths(ctype, ('const half2', 'half2'))

                elif value.dtype == int or (hasattr(np, 'int') and value.dtype == np.int):
                    assert startswiths(ctype, ('const int', 'int'))

    def check_half2(self, py_dict: dict):
        """
        This function is implemented for sub-class when needed.
        """
        raise NotImplementedError

    def get_ptrs(self, py_dict: dict):
        """
        :param py_dict: a dict
        :type py_dict: dict
        :return: a tuple of data ptr
        :rtype: tuple

        Get the address of the first element of each ``torch.Tensor`` or ``cupy.ndarray`` in ``py_dict``.
        """
        ret_list = []
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                ret_list.append(item.data_ptr())

            elif isinstance(item, cupy.ndarray):
                ret_list.append(item)

            else:
                raise TypeError
        return tuple(ret_list)

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        """
        :param grid: the grid number of CUDA kernel
        :type grid: tuple
        :param block: the block number of CUDA kernel
        :type block: tuple
        :param py_dict: the dict that contains parameters for CUDA kernel
        :type py_dict: dict

        Execute the CUDA kernel. ``*args_1, **kwargs`` are used as ``*args_1, **kwargs`` in :class:`cupy.RawKernel`.

        ``py_dict`` should contain ``key: value`` where ``key`` is the cuda kernel function param name, and ``value`` is
        the variable. This dict should be one-to-one correspondence to ``self.cparams``.

        For example, if ``self.cparams`` is

        .. code-block:: python

            {
                'numel': 'const int &',
                'x': 'const float *',
                'y': 'const float *'
            }


        Then ``py_dict`` sould be

        .. code-block:: python

            {
                'numel': numel,
                'x': x,
                'y': y
            }

        where ``numel, x, y`` should be ``torch.Tensor`` or ``cupy.ndarray`` with the corresponding data type, e.g.,
        ``x`` in ``py_dict`` should have data type ``torch.float`` because ``x`` in ``self.cparams`` have value ``'const float *'`` .

        The keys order is arbitrary because this function will sort keys to align formal and actual parameters.

        """

        device = self.get_device(py_dict)
        self.check_device(device, py_dict)

        self.set_contiguous(py_dict)

        self.check_ctypes(py_dict)

        self.check_half2(py_dict)

        py_dict = dict(sorted(py_dict.items()))
        self.check_keys(py_dict)
        assert sys.version_info.major >= 3 and sys.version_info.minor >= 6
        # 需要使用有序词典
        # python >= 3.6时，字典默认是有序的


        cp_kernel = cupy.RawKernel(self.full_codes, self.kernel_name, options=configure.cuda_compiler_options,
                                   backend=configure.cuda_compiler_backend)

        with cuda_utils.DeviceEnvironment(device):
            cp_kernel(grid, block, self.get_ptrs(py_dict), *args_1, **kwargs)

    def add_param(self, ctype: str, cname: str):
        """
        :param ctype: the type of the CUDA param
        :type ctype: str
        :param cname: the name of the CUDA param
        :type cname: str

        Add a param to ``self.cparams``.

        .. admonition:: Note
            :class: note

            When calling ``self.__call__``, the params order in the CUDA kernel are sorted by the dictionary order. Thus,
            the user do not need to call ``add_param`` by some specific order.

        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base

            example_ck = base.CKernel(kernel_name='example_ck')
            print('origin:')
            print(example_ck.full_codes)


            example_ck.add_param(ctype='const float*', cname='x')
            example_ck.add_param(ctype='const float*', cname='y')
            example_ck.add_param(ctype='float', cname='z')

            print('after:')
            print(example_ck.full_codes)

        .. code-block:: c++

            origin:

                    #include <cuda_fp16.h>
                    extern "C" __global__
                    void example_ck(
                    const int & numel
                    )

            after:

                    #include <cuda_fp16.h>
                    extern "C" __global__
                    void example_ck(
                    const int & numel, const float* x, const float* y, float z
                    )


        """
        # example: ctype = 'const float *', cname = 'x'
        if cname in self.cparams:
            raise ValueError(f'{cname} has been added to cparams!')

        if cname in self.reserved_cnames:
            raise ValueError(
                f'{cname} is the reserved cname. You should change the name of your variable to avoid conflict.')

        self.cparams[cname] = ctype

    @property
    def declaration(self):
        codes = f'''
        #include <cuda_fp16.h>
        extern "C" __global__
        void {self.kernel_name}(
        '''
        self.cparams = dict(sorted(self.cparams.items()))
        params_list = []
        for cname, ctype in self.cparams.items():
            params_list.append(f'{ctype} {cname}')

        codes += ', '.join(params_list)

        codes += '''
        )
        '''
        return codes

    @property
    def head(self):
        return '{'

    @property
    def tail(self):
        return '}'

    @property
    def full_codes(self):
        """
        :return: the full cuda codes
        :rtype: str

        """
        return wrap_with_comment(self.declaration, 'declaration') + wrap_with_comment(self.head,
                                                                                      'head') + wrap_with_comment(
            self.core, 'core') + wrap_with_comment(self.tail, 'tail')


class CKernel1D(CKernel):
    def __init__(self, *args, **kwargs):
        """
        :param kernel_name: the name of kernel
        :type kernel_name: str

        The 1D (element-wise) CUDA kernel, which is extended from :class:`CKernel <spikingjelly.activation_based.auto_cuda.base.CKernel>`.
        All input/output tensors will be regarded as 1D tensors.

        Some critical attributes:

            cparams:
                A dict for saving parameters name and type.
                The default value is ``{'numel': 'const int &'}``.
                ``numel`` represents the numel of elements for element-wise operations, which is also the numer of cuda
                threads.

            reserved_cnames:
                A list for saving reserved variables names, which can not be used to name variable again.
                The defaule value is ``['index']``.
                ``index`` represents the index of element, which is also the cuda thread index.

        Now let us check what the empty 1d kernel looks like:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base
            temp_kernel = base.CKernel1D(kernel_name='temp_kernel')
            print(temp_kernel.full_codes)

        The outputs are:

            .. code-block:: c++

                #include <cuda_fp16.h>
                extern "C" __global__
                void temp_kernel(
                const int & numel
                )

                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {

                    }
                }

        With setting logging level, we can check each part of the kernel:

        .. code-block:: python

            import logging
            logging.basicConfig(level=logging.DEBUG)
            from spikingjelly.activation_based.auto_cuda import base
            temp_kernel = base.CKernel1D(kernel_name='temp_kernel')
            print(temp_kernel.full_codes)

        The outputs are:

        .. code-block:: c++

            //------declaration start------

                    #include <cuda_fp16.h>
                    extern "C" __global__
                    void temp_kernel(
                    const int & numel
                    )

            //------declaration end--------


            //------head start------

                    {
                        const int index = blockIdx.x * blockDim.x + threadIdx.x;
                        if (index < numel)
                        {

            //------head end--------


            //------core start------

            //------core end--------


            //------tail start------

                        }
                    }

            //------tail end--------

        ``self.code`` can be specified by user.
         For example, if we want to write a heaviside kernel, we can implement it easily with the cuda code
         ``y[index] = x[index] >= 0.0f ? 1.0f: 0.0f;``, and add two params ``x, y``, which are inputs and outputs.

        Here is the example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base

            c_heaviside = base.CKernel1D(kernel_name='heaviside')
            c_heaviside.add_param(ctype='const float *', cname='x')
            c_heaviside.add_param(ctype='float *', cname='y')
            c_heaviside.core = '''
                        y[index] = x[index] >= 0.0f ? 1.0f: 0.0f;
            '''
            print(c_heaviside.full_codes)

        The outputs are:

        .. code-block:: c++

                #include <cuda_fp16.h>
                extern "C" __global__
                void heaviside(
                const int & numel, const float * x, float * y
                )

                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {

                    y[index] = x[index] >= 0.0f ? 1.0f: 0.0f;

                    }
                }

        Here is an example of how to execute the kernel:

        .. code-block:: bash

            import torch
            from spikingjelly.activation_based import cuda_utils

            device = 'cuda:0'
            x = torch.rand([4, 4], device=device) - 0.5
            y = torch.zeros_like(x)

            numel = x.numel()
            threads = 1024
            blocks = cuda_utils.cal_blocks(numel, threads)
            print('x=')
            print(x)

            with cuda_utils.DeviceEnvironment(device=x.get_device()):
                numel = cupy.asarray(numel)
                py_dict = {
                    'numel': numel,
                    'x': x,
                    'y': y
                }
                c_heaviside((blocks, ), (threads, ), py_dict)


            print('y=')
            print(y)

        The outputs are:

        .. code-block:: bash

            x=
            tensor([[-0.0423, -0.1383, -0.0238,  0.1018],
                    [ 0.3422,  0.1449, -0.2938, -0.1858],
                    [-0.3503,  0.0004, -0.4274, -0.2012],
                    [-0.0227,  0.2229, -0.0776,  0.2687]], device='cuda:0')
            y=
            tensor([[0., 0., 0., 1.],
                    [1., 1., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 1.]], device='cuda:0')



        """
        super().__init__(*args, **kwargs)
        self.cparams['numel'] = 'const int &'
        self.reserved_cnames.append('index')

    @property
    def head(self):
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < numel)
            {
        '''
        return codes

    @property
    def tail(self):
        codes = '''
            }
        }
        '''
        return codes

    def check_half2(self, py_dict: dict):
        """
        :param py_dict: a dict
        :type py_dict: dict

        Check value in ``py_dict``.  If the value is ``torch.Tensor`` with ``value.dtype == torch.half`` or
        ``cupy.ndarray`` with ``value.dtype == np.float16``, this function will check whether the number of elements of
        value is even.

        We assert when using half dtype, the numel should be even because we will use ``half2`` in CUDA kernel.

        .. admonition:: Note
            :class: note

            :class:`CKernel1D.__call__ <spikingjelly.activation_based.auto_cuda.CKernel1D.__call__>` will pad half
            tensor to even numel before executing the kernel. Thus, the user does not need to worry about padding.


        """
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    assert value.numel() % 2 == 0, f'please pad the numel of {key} to assert mod 2 == 0! (for half2)'

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    assert value.size % 2 == 0, f'please pad the numel of {key} to assert mod 2 == 0! (for half2)'

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        """
        :param grid: the grid number of CUDA kernel
        :type grid: tuple
        :param block: the block number of CUDA kernel
        :type block: tuple
        :param py_dict: the dict that contains parameters for CUDA kernel
        :type py_dict: dict

        Execute the CUDA kernel. ``*args_1, **kwargs`` are used as ``*args_1, **kwargs`` in :class:`cupy.RawKernel`.

        ``py_dict`` should contain ``key: value`` where ``key`` is the cuda kernel function param name, and ``value`` is
        the variable. This dict should be one-to-one correspondence to ``self.cparams``.

        For example, if ``self.cparams`` is

        .. code-block:: python

            {
                'numel': 'const int &',
                'x': 'const float *',
                'y': 'const float *'
            }


        Then ``py_dict`` sould be

        .. code-block:: python

            {
                'numel': numel,
                'x': x,
                'y': y
            }

        where ``numel, x, y`` should be ``torch.Tensor`` or ``cupy.ndarray`` with the corresponding data type, e.g.,
        ``x`` in ``py_dict`` should have data type ``torch.float`` because ``x`` in ``self.cparams`` have value ``'const float *'`` .

        The keys order is arbitrary because this function will sort keys to align formal and actual parameters.

        .. admonition:: Note
            :class: note

            All tensors in ``py_dict`` will be regarded as 1D.


        .. admonition:: Note
            :class: note

            If any tensor ``x`` in ``py_dict`` with data type ``torch.half`` or ``np.float16`` but odd numel will be
            flattened and padded by ``x = [x, x[-1]]`` before executing the CUDA kernel. After execution, padded values
            in ``x`` will be removed, and ``x`` will be reshaped to the origin shape.


        """
        # pad half2
        pad_keys = []
        pad_shapes = []
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.half:
                if value.numel() % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    value = value.flatten()

                    value = torch.cat((value, value[-1].view(1)))

                    py_dict[key] = value


            elif isinstance(value, cupy.ndarray) and value.dtype == np.float16:
                if value.size % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    value = cupy.reshape(value, -1)

                    value = cupy.concatenate((value, cupy.reshape(value[-1], 1)))

                    py_dict[key] = value

        super().__call__(grid, block, py_dict, *args_1, **kwargs)

        # move pad values
        for key, shape in zip(pad_keys, pad_shapes):
            value = py_dict[key]
            value = value[: -1]

            if isinstance(value, torch.Tensor):
                value = value.view(shape)

            elif isinstance(value, cupy.ndarray):
                value = cupy.reshape(value, shape)

            py_dict[key] = value

    def simple_call(self, **kwargs):
        """
        :param kwargs: the dict that contains parameters for CUDA kernel
        :type kwargs: dict


        The simplified calling function, which is simplified from the standard calling function is :class:`CKernel1D.simple_call <spikingjelly.activation_based.auto_cuda.CKernel1D.__call__>`.

        Compared with :class:`CKernel1D.simple_call <spikingjelly.activation_based.auto_cuda.CKernel1D.__call__>`,
        the device, numel, numbers of CUDA threads and blocks are calculated automatically from tensors in ``kwargs``.

        Here is the example:

        .. code-block:: python

            import torch
            from spikingjelly.activation_based import cuda_utils
            from spikingjelly.activation_based.auto_cuda import base

            c_heaviside = base.CKernel1D(kernel_name='heaviside')
            c_heaviside.add_param(ctype='const float *', cname='x')
            c_heaviside.add_param(ctype='float *', cname='y')
            c_heaviside.core = '''
                        y[index] = x[index] >= 0.0f ? 1.0f: 0.0f;
            '''
            device = 'cuda:0'

            x = torch.rand([4, 4], device=device) - 0.5
            y = torch.zeros_like(x)

            print('x=')
            print(x)
            c_heaviside.simple_call(x=x, y=y)
            print('y=')
            print(y)

        The outputs are:

        .. code-block:: bash

            x=
            tensor([[-0.1706,  0.2063, -0.2077,  0.3335],
                    [-0.0180, -0.2429,  0.3488,  0.1146],
                    [ 0.0362,  0.1584,  0.4828, -0.1389],
                    [-0.2684,  0.1898,  0.0560,  0.2058]], device='cuda:0')
            y=
            tensor([[0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [1., 1., 1., 0.],
                    [0., 1., 1., 1.]], device='cuda:0')

        """
        py_dict = kwargs
        device = self.get_device(py_dict)
        numel = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                numel = value.numel()
            elif isinstance(value, cupy.ndarray):
                numel = value.size

        if numel is None:
            raise ValueError('No torch.Tensor or cupy.ndarray in kwargs!')

        with cuda_utils.DeviceEnvironment(device):
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            py_dict['numel'] = numel
            self.__call__((blocks, ), (threads, ), py_dict)


class CKernel2D(CKernel):
    def __init__(self, kernel_name: str, reverse: bool = False):

        """
        :param kernel_name: the name of kernel
        :type kernel_name: str
        :param reverse: If ``True``, then the for-loop in kernel is ``for(int t = index; t < numel; t += dt)``.
            If ``False``, then the for-loop in kernel is ``for(int t = numel - N + index; t >= 0; t -= dt)``.
        :type reverse: bool


        The 2D CUDA kernel, which is extended from :class:`CKernel <spikingjelly.activation_based.auto_cuda.base.CKernel>`.

        All input/output tensors should have dimensions no more than 2. All 2D tensors will be regarded as ``shape = [T, N]``,
        where ``T`` is the sequence length and ``N`` is the elements number of  data at one time-step

        Some critical attributes:

            cparams:
                A dict for saving parameters name and type.
                The default value is ``{'numel': 'const int &', 'N': 'const int &'}``.

                ``N``: the number of elements number of sequence data at one time-step (the numel of 1-th dimension)

                ``numel``: the numel of elements in input/output tensors, which is ``T * N``


            reserved_cnames:
                A list for saving reserved variables names, which can not be used to name variable again.
                The defaule value is ``['index', 'dt', 't']``.

                ``index``: the index in 1-th dimension, which is also the CUDA thread index

                ``t``: the index in 0-th dimension

                ``dt``: used in CUDA kernel as the time-step stride. When ``x[t_py][j]`` in python code is identical to
                ``x[t]`` in CUDA code, then ``x[t_py + 1][j]`` in python code is identical to ``x[t + dt]`` in CUDA code.

        Now let us check what the empty 2d kernel looks like:

        .. code-block:: python

           from spikingjelly.activation_based.auto_cuda import base

            temp_kernel = base.CKernel2D(kernel_name='temp_kernel')
            print(temp_kernel.full_codes)

        The outputs are:

            .. code-block:: c++

                #include <cuda_fp16.h>
                extern "C" __global__
                void temp_kernel(
                const int & numel, const int & N
                )

                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < N)
                    {
                        const int dt = N;

                        for(int t = index; t < numel; t += dt)
                        {

                        }

                    }
                }

        With setting logging level, we can check each part of the kernel:

        .. code-block:: python

            import logging
            logging.basicConfig(level=logging.DEBUG)
            from spikingjelly.activation_based.auto_cuda import base

            temp_kernel = base.CKernel2D(kernel_name='temp_kernel')
            print(temp_kernel.full_codes)

        The outputs are:

            .. code-block:: c++

                //------declaration start------

                #include <cuda_fp16.h>
                extern "C" __global__
                void temp_kernel(
                const int & numel, const int & N
                )

                //------declaration end--------


                //------head start------

                        {
                            const int index = blockIdx.x * blockDim.x + threadIdx.x;
                            if (index < N)
                            {
                                const int dt = N;

                //------pre_core start------

                //------pre_core end--------


                                for(int t = index; t < numel; t += dt)
                                {

                //------head end--------


                //------core start------

                //------core end--------


                //------tail start------

                                }

                //------post_core start------

                //------post_core end--------


                            }
                        }

                //------tail end--------

        ``self.pre_core, self.post_core, self.core`` can be specified by user.

        Here is the example of how to implement the :class:`cumsum <torch.cumsum>` operation:

        .. code-block:: python

            import torch
            import cupy
            from spikingjelly.activation_based.auto_cuda import base
            from spikingjelly.activation_based import cuda_utils

            cumsum = base.CKernel2D(kernel_name='cumsum')
            cumsum.add_param(ctype='const float *', cname='x')
            cumsum.add_param(ctype='float *', cname='y')

            cumsum.core = '''
                                if (t - dt < 0)
                                {
                                    y[t] = x[t];
                                }
                                else
                                {
                                    y[t] = x[t] + y[t - dt];
                                }
            '''

            print(cumsum.full_codes)

            T = 4
            N = 3
            device = 'cuda:0'

            x = torch.randint(low=0, high=4, size=[T, N], device=device).float()
            y = torch.zeros_like(x)

            threads = 1024
            blocks = cuda_utils.cal_blocks(N, threads)

            with cuda_utils.DeviceEnvironment(device=x.get_device()):
                numel = cupy.asarray(T * N)
                N = cupy.asarray(N)
                py_dict = {
                    'N': N,
                    'numel': numel,
                    'x': x,
                    'y': y
                }
                cumsum((blocks, ), (threads, ), py_dict)

            print('x=')
            print(x)
            print('y=')
            print(y)


        The outputs are:

        .. code-block:: c++

            #include <cuda_fp16.h>
            extern "C" __global__
            void cumsum(
            const int & numel, const int & N, const float * x, float * y
            )

            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int dt = N;

                    for(int t = index; t < numel; t += dt)
                    {

                        if (t - dt < 0)
                        {
                            y[t] = x[t];
                        }
                        else
                        {
                            y[t] = x[t] + y[t - dt];
                        }

                    }

                }
            }

        .. code-block:: bash

            x=
            tensor([[3., 0., 2.],
                    [2., 0., 0.],
                    [2., 3., 2.],
                    [2., 1., 0.]], device='cuda:0')
            y=
            tensor([[3., 0., 2.],
                    [5., 0., 2.],
                    [7., 3., 4.],
                    [9., 4., 4.]], device='cuda:0')

        """
        super().__init__(kernel_name)
        self.cparams['numel'] = 'const int &'
        self.reverse = reverse
        self.cparams['N'] = 'const int &'
        self.reserved_cnames.append('index')
        self.reserved_cnames.append('dt')
        self.reserved_cnames.append('t')

        self._pre_core = ''
        self._post_core = ''

    @property
    def pre_core(self):
        return self._pre_core

    @pre_core.setter
    def pre_core(self, value: str):
        self._pre_core = value

    @property
    def post_core(self):
        return self._post_core

    @post_core.setter
    def post_core(self, value: str):
        self._post_core = value

    def check_shape(self, py_dict: dict):
        # all tensors should be ndim <= 2
        for value in py_dict.values():
            if isinstance(value, torch.Tensor):
                assert value.ndim <= 2

            elif isinstance(value, cupy.ndarray):
                assert value.ndim <= 2

    def check_half2(self, py_dict: dict):
        """
        :param py_dict: a dict
        :type py_dict: dict

        Check value in ``py_dict``.  If the value is ``torch.Tensor`` with ``value.dtype == torch.half`` or
        ``cupy.ndarray`` with ``value.dtype == np.float16``, this function will check whether the number of elements of
        value is even.

        If the tensor ``x`` is 1D, it will be padded when ``x.numel() % 2 != 0``.
        If the tensor ``x`` is 2D, it will be padded when ``x.shape[1] % 2 != 0``.

        We assert when using half dtype, the numel should be even because we will use ``half2`` in CUDA kernel.

        .. admonition:: Note
            :class: note

            :class:`CKernel2D.__call__ <spikingjelly.activation_based.auto_cuda.CKernel2D.__call__>` will pad half
            tensor to even numel before executing the kernel. Thus, the user does not need to worry about padding.


        """
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):

                if value.dtype == torch.half:
                    if value.ndim <= 1:
                        assert value.numel() % 2 == 0
                    elif value.ndim == 2:
                        assert value.shape[1] % 2 == 0

            elif isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    if value.ndim <= 1:
                        assert value.size % 2 == 0
                    elif value.ndim == 2:
                        assert value.shape[1] % 2 == 0

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        """
        :param grid: the grid number of CUDA kernel
        :type grid: tuple
        :param block: the block number of CUDA kernel
        :type block: tuple
        :param py_dict: the dict that contains parameters for CUDA kernel
        :type py_dict: dict

        Execute the CUDA kernel. ``*args_1, **kwargs`` are used as ``*args_1, **kwargs`` in :class:`cupy.RawKernel`.

        ``py_dict`` should contain ``key: value`` where ``key`` is the cuda kernel function param name, and ``value`` is
        the variable. This dict should be one-to-one correspondence to ``self.cparams``.

        For example, if ``self.cparams`` is

        .. code-block:: python

            {
                'numel': 'const int &',
                'x': 'const float *',
                'y': 'const float *'
            }


        Then ``py_dict`` sould be

        .. code-block:: python

            {
                'numel': numel,
                'x': x,
                'y': y
            }

        where ``numel, x, y`` should be ``torch.Tensor`` or ``cupy.ndarray`` with the corresponding data type, e.g.,
        ``x`` in ``py_dict`` should have data type ``torch.float`` because ``x`` in ``self.cparams`` have value ``'const float *'`` .

        The keys order is arbitrary because this function will sort keys to align formal and actual parameters.

        .. admonition:: Note
            :class: note

            All tensors in ``py_dict`` should be 1D or 2D.


        .. admonition:: Note
            :class: note

            If any 1D tensor ``x`` in ``py_dict`` with data type ``torch.half`` or ``np.float16`` but odd numel will be
            flattened and padded by ``x = [x, x[-1]]`` before executing the CUDA kernel.

            If any 2D tensor ``x`` with shape ``[T, N]`` in ``py_dict`` with data type ``torch.half`` or ``np.float16``
            but ``N`` is odd, then ``x`` will be padded as ``x = [x, x[:, -1]]``, whose shape is ``[T, N + 1]``.

            After execution, padded values in ``x`` will be removed, and ``x`` will be reshaped to the origin shape.
        """
        self.check_shape(py_dict)

        # pad half2
        pad_keys = []
        pad_shapes = []
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.half:

                if value.ndim <= 1:
                    # 1D tensor
                    if value.numel() % 2 != 0:
                        pad_shapes.append(value.shape)
                        pad_keys.append(key)
                        value = value.flatten()

                        value = torch.cat((value, value[-1].view(1)))

                        py_dict[key] = value



                elif value.shape[1] % 2 != 0:
                    # 2D tensor with shape = [T, N] and N % 2 != 0
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)

                    value = torch.cat((value, value[:, -1].view(-1, 1)), dim=1)
                    # [T, N] -> [T, N + 1]
                    py_dict[key] = value



            elif isinstance(value, cupy.ndarray) and value.dtype == np.float16:

                if value.ndim <= 1:
                    # 1D tensor
                    if value.size % 2 != 0:
                        pad_shapes.append(value.shape)
                        pad_keys.append(key)
                        value = cupy.reshape(value, -1)

                        value = cupy.concatenate((value, cupy.reshape(value[-1], 1)))

                        py_dict[key] = value


                elif value.shape[1] % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    # [T, N] -> [T, N + 1]

                    value = cupy.concatenate((value, cupy.reshape(value[:, -1], (-1, 1))), axis=1)
                    py_dict[key] = value

        super().__call__(grid, block, py_dict, *args_1, **kwargs)

        # move pad values
        for i, key in enumerate(pad_keys):
            value = py_dict[key]
            shape = pad_shapes[i]
            if isinstance(value, torch.Tensor):
                if value.ndim <= 1:
                    value = value[: -1]
                    value = value.view(shape)
                else:
                    value = value[:, : -1]

            elif isinstance(value, cupy.ndarray):
                if value.ndim <= 1:
                    value = value[:, -1]
                    value = cupy.reshape(value, shape)

                else:
                    value = value[:, : -1]

            py_dict[key] = value

    @property
    def head(self):
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        '''

        codes += wrap_with_comment(self.pre_core, 'pre_core')

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes

    @property
    def tail(self):
        codes = '''
                }
        '''

        codes += wrap_with_comment(self.post_core, 'post_core')

        codes += '''
            }
        }
        '''
        return codes

    def simple_call(self, **kwargs):
        """
        :param kwargs: the dict that contains parameters for CUDA kernel
        :type kwargs: dict


        The simplified calling function, which is simplified from the standard calling function is :class:`CKernel2D.simple_call <spikingjelly.activation_based.auto_cuda.CKernel2D.__call__>`.

        Compared with :class:`CKernel2D.simple_call <spikingjelly.activation_based.auto_cuda.CKernel2D.__call__>`,
        the device, N, numel, numbers of CUDA threads and blocks are calculated automatically from tensors in ``kwargs``.

        Here is the example:

        .. code-block:: python

            import torch
            import cupy
            from spikingjelly.activation_based.auto_cuda import base
            from spikingjelly.activation_based import cuda_utils

            cumsum = base.CKernel2D(kernel_name='cumsum')
            cumsum.add_param(ctype='const float *', cname='x')
            cumsum.add_param(ctype='float *', cname='y')

            cumsum.core = '''
                                if (t - dt < 0)
                                {
                                    y[t] = x[t];
                                }
                                else
                                {
                                    y[t] = x[t] + y[t - dt];
                                }
            '''

            T = 4
            N = 3
            device = 'cuda:0'

            x = torch.randint(low=0, high=4, size=[T, N], device=device).float()
            y = torch.zeros_like(x)

            cumsum.simple_call(x=x, y=y)
            print('x=')
            print(x)
            print('y=')
            print(y)

        The outputs are:

        .. code-block:: bash

            x=
            tensor([[0., 2., 1.],
                    [1., 3., 1.],
                    [2., 2., 0.],
                    [2., 0., 1.]], device='cuda:0')
            y=
            tensor([[0., 2., 1.],
                    [1., 5., 2.],
                    [3., 7., 2.],
                    [5., 7., 3.]], device='cuda:0')

        """
        py_dict = kwargs
        device = self.get_device(py_dict)

        numel = None
        N = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                numel = value.numel()
                N = value.shape[1]
            elif isinstance(value, cupy.ndarray) and value.ndim == 2:
                numel = value.size
                N = value.shape[1]

        if numel is None or N is None:
            raise ValueError('No 2D torch.Tensor or cupy.ndarray in kwargs!')

        with cuda_utils.DeviceEnvironment(device):
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)
            py_dict['numel'] = numel
            py_dict['N'] = N
            self.__call__((blocks,), (threads,), py_dict)



class CodeTyper:
    def __init__(self, indent_num: int):
        """
        :param indent_num: the number of indents
        :type indent_num: int

        A CUDA code formatter with adding indents. The full code can be accessed by ``self.codes``.

        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base, cfunction

            code0 = cfunction.if_else(z='z', x='x', y='y', mask='mask', dtype='float')
            code1 = cfunction.sigmoid_backward(y='y', x='x', alpha=2., dtype='float')

            codes = ''
            codes += code0
            codes += code1

            print('// Without CodeTyper:')
            print('// ------------------')
            print(codes)
            print('// ------------------')

            ctyper = base.CodeTyper(4)
            ctyper.append(code0)
            ctyper.append(code1)
            print('// With CodeTyper:')
            print('// ------------------')
            print(ctyper.codes)
            print('// ------------------')

        .. code-block:: c++

            // Without CodeTyper:
            // ------------------
            z = x * mask + y * (1.0f - mask);const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (2.0f) * x));
            y = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (2.0f);
            // ------------------
            // With CodeTyper:
            // ------------------

                z = x * mask + y * (1.0f - mask);
                const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (2.0f) * x));
                y = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (2.0f);

            // ------------------


        """
        self.indent = ' ' * indent_num
        self.codes = '\n'

    def append(self, codes: str):
        """
        :param codes: cuda codes to be added
        :type codes: str

        Append codes in ``self.codes``.
        """
        codes = codes.replace('\n', '')
        codes = codes.split(';')
        for i in range(codes.__len__()):
            if codes[i].__len__() > 0:
                if codes[i] in ('{', '}'):
                    self.codes += (self.indent + codes[i] + '\n')
                else:
                    self.codes += (self.indent + codes[i] + ';\n')


class CodeBlock:
    def __init__(self, env: CodeTyper):
        """
        :param env: a CodeTyper
        :type env: CodeTyper

        A tool for adding a CUDA code block in ``CodeTyper.code``. It is helpful when we want to calculate by intermediate variables.

        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base

            ctyper = base.CodeTyper(4)
            with base.CodeBlock(ctyper):
                ctyper.append('// swap x and y')
                ctyper.append('float temp_var = x;')
                ctyper.append('x = y;')
                ctyper.append('y = temp_var;')

            print(ctyper.codes)

        The outputs are:

        .. code-block:: c++

            {
             // swap x and y;
             float temp_var = x;
             x = y;
             y = temp_var;
            }


        """
        self.env = env

    def __enter__(self):
        self.env.append('{')
        self.env.indent += ' '

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.indent = self.env.indent[: -1]
        self.env.append('}')

