import numpy as np

import cupy
import torch
import sys
import logging
from .. import cuda_utils
from ... import configure

def startswiths(x: str, prefixes: tuple):
    ret = False
    for prefix in prefixes:
        if x.startswith(prefix):
            ret = True

    return ret


class CKernel:
    def __init__(self, kernel_name: str):
        self.cparams = {'numel': 'const int &'}
        self.reserved_cnames = ['index']
        self.kernel_name = kernel_name

    def check_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if value != self.__getattribute__(key):
                return False

        else:
            return True

    def set_contiguous(self, py_dict: dict):
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
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                return item.get_device()

            elif isinstance(item, cupy.ndarray):
                return item.device.id

        raise ValueError


    def check_device(self, device: int, py_dict: dict):
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                assert item.get_device() == device

            elif isinstance(item, cupy.ndarray):
                assert item.device.id == device

    def check_keys(self, py_dict: dict):
        if py_dict.keys() != self.cparams.keys():
            missed_keys = (py_dict.keys() | self.cparams.keys()) - (py_dict.keys() & self.cparams.keys())

            if missed_keys.__len__() > 0:
                if (missed_keys & py_dict.keys()).__len__() > 0:
                    msg = f'{missed_keys} is in py_dict but not in cparams!'
                else:
                    msg = f'{missed_keys} is in cparams but not in py_dict!'
                raise ValueError(msg)

    def check_ctypes(self, py_dict: dict):
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

                elif value.dtype == np.int:
                    assert startswiths(ctype, ('const int', 'int'))


    def check_shape(self, py_dict: dict):
        raise NotImplementedError

    def get_ptr(self, py_dict: dict):
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


        self.check_keys(py_dict)


        assert sys.version_info.major >= 3 and sys.version_info.minor >= 6
        # 需要使用有序词典
        # python >= 3.6时，字典默认是有序的
        py_dict = dict(sorted(py_dict.items()))
        self.cparams = dict(sorted(self.cparams.items()))

        device = self.get_device(py_dict)

        self.check_device(device, py_dict)

        self.set_contiguous(py_dict)

        self.check_ctypes(py_dict)

        self.check_shape(py_dict)


        cp_kernel = cupy.RawKernel(self.full_codes, self.kernel_name, options=configure.cuda_compiler_options,
                           backend=configure.cuda_compiler_backend)

        with cuda_utils.DeviceEnvironment(device):
            cp_kernel(grid, block, self.get_ptr(py_dict))











    def add_param(self, ctype: str, cname: str):
        # example: ctype = 'const float *', cname = 'x'
        if cname in self.cparams:
            raise ValueError(f'{cname} has been added to cparams!')

        if cname in self.reserved_cnames:
            raise ValueError(
                f'{cname} is the reserved cname. You should change the name of your variable to avoid conflict.')

        self.cparams[cname] = ctype

    @property
    def core(self):
        return ''



    @property
    def declaration(self):
        codes = f'''
        #include <cuda_fp16.h>
        extern "C" __global__
        void {self.kernel_name}(
        '''
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
        return self.declaration + self.head + self.core + self.tail


class CKernel1D(CKernel):
    """
    
    example:
    
        c_heaviside = CKernel1D(kernel_name='heaviside')
        c_heaviside.add_param(ctype='const float *', cname='x')
        c_heaviside.add_param(ctype='float *', cname='y')
        c_heaviside.core = '''
                    y[index] = x[index] >= 0.0f ? 1.0f: 0.0f;
        '''
        print(c_heaviside.full_codes)

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
    """

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


    def check_shape(self, py_dict: dict):
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    assert value.numel() % 2 == 0, f'please pad the numel of {key} to assert mod 2 == 0! (for half2)'

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    assert value.size % 2 == 0, f'please pad the numel of {key} to assert mod 2 == 0! (for half2)'


class CKernel2D(CKernel):
    """
    c_sum_T = CKernel2D(kernel_name='sum_T')
    c_sum_T.add_param(ctype='const float *', cname='x_seq')
    c_sum_T.add_param(ctype='float *', cname='y')
    c_sum_T.pre_core = '''
                y[index] = 0.0f;
    '''
    c_sum_T.core = '''
                    y[index] += x[t];
    '''
    print(c_sum_T.full_codes)


    #include <cuda_fp16.h>
    extern "C" __global__
    void sum_T(
    const int & numel, const int & N, const float * x_seq, float * y
    )

    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < N)
        {
            const int dt = N;

            y[index] = 0.0f;

            for(int t = index; t < numel; t += dt)
            {

                y[index] += x[t];

            }

        }
    }


    """
    def __init__(self, kernel_name: str, reverse: bool = False):
        super().__init__(kernel_name)
        self.reverse = reverse
        self.cparams['N'] = 'const int &'
        self.reserved_cnames.append('dt')
        self.reserved_cnames.append('t')

    def check_shape(self, py_dict: dict):
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.dim() <= 2
                if value.dtype == torch.half:
                    if value.dim() == 1:
                        assert value.numel() % 2 == 0
                    elif value.dim() == 2:
                        assert value.shape[1] % 2 == 0

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    assert value.ndim <= 2
                    if value.ndim == 1:
                        assert value.size % 2 == 0
                    elif value.dim == 2:
                        assert value.shape[1] % 2 == 0



    @property
    def pre_core(self):
        return ''



    @property
    def post_core(self):
        return ''



    @property
    def head(self):
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

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

        codes += self.post_core

        codes += '''
            }
        }
        '''
        return codes


class CodeTyper:
    def __init__(self, indent_num: int):
        self.indent = ' ' * indent_num
        self.codes = '\n'

    def append(self, codes: str):
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
        self.env = env

    def __enter__(self):
        self.env.append('{')
        self.env.indent += ' '

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.indent = self.env.indent[: -1]
        self.env.append('}')


if __name__ == '__main__':
    c_sum_T = CKernel2D(kernel_name='sum_T')
    c_sum_T.add_param(ctype='const float *', cname='x_seq')
    c_sum_T.add_param(ctype='float *', cname='y')
    c_sum_T.pre_core = '''
                y[index] = 0.0f;
    '''
    c_sum_T.core = '''
                    y[index] += x[t];
    '''
    print(c_sum_T.full_codes)
