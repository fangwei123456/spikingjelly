import cupy
import torch
import sys
import logging
from .. import cuda_utils
from ... import configure

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

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):

        assert sys.version_info.major >= 3 and sys.version_info.minor >= 6
        # python >= 3.6时，字典默认是有序的

        assert py_dict.keys() == self.cparams.keys(), f'{(py_dict.keys() | self.cparams.keys()) - (py_dict.keys() & self.cparams.keys())} is not in both dicts!'

        py_dict = dict(sorted(py_dict.items()))
        self.cparams = dict(sorted(self.cparams.items()))

        device = None

        # get contiguous
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.contiguous()
                device = value.get_device()

            elif isinstance(value, cupy.ndarray):
                value = cupy.ascontiguousarray(value)
            else:
                raise TypeError(type(value))

            py_dict[key] = value


        cp_kernel = cupy.RawKernel(self.full_codes, self.kernel_name, options=configure.cuda_compiler_options,
                           backend=configure.cuda_compiler_backend)

        with cuda_utils.DeviceEnvironment(device):
            cp_kernel(grid, block, cuda_utils.wrap_args_to_raw_kernel(device, *list(py_dict.values())))











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
