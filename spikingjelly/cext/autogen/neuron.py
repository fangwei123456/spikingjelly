from spikingjelly.cext.autogen import check_function, utils
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils import cpp_extension

default_threads = '1024'
c_headers = ['iostream', 'torch/extension.h', 'math.h']
cuda_headers = ['cuda.h', 'cuda_runtime.h', 'cuda_fp16.h', 'torch/extension.h', 'math.h', 'stdio.h']



class NeuronFPTT:
    @staticmethod
    def pybind_name(neuron_name, hard_reset):
        if hard_reset:
            fname = f'{neuron_name}_hard_reset_fptt'
        else:
            fname = f'{neuron_name}_soft_reset_fptt'

        return f'm.def("{fname}", {fname});'

    @staticmethod
    def def_fptt(neuron_name, hard_reset, extra_arg=''):
        if hard_reset:
            return f'std::vector<at::Tensor> {neuron_name}_hard_reset_fptt' \
                   f'(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset{utils.add_comma(extra_arg)})'
        else:
            return f'std::vector<at::Tensor> {neuron_name}_soft_reset_fptt' \
                   f'(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th{utils.add_comma(extra_arg)})'

    @staticmethod
    def impl_fptt(neuron_name, hard_reset, extra_arg='', extra_cuda_kernel_args='', extra_cuda_kernel_half_args='', extra_codes=''):
        # define the function name

        # all inputs will be regarded as [T, N] = [seq_len, neuron_num]
        ret = NeuronFPTT.def_fptt(neuron_name, hard_reset, extra_arg)
        ret += '\n{\n'

        ret += 'auto s_seq = torch::zeros_like(x_seq.data());\n'
        ret += 'auto v_seq = at::cat({v.unsqueeze(0), torch::zeros_like(x_seq.data())}, 0);\n'

        # check variables
        for tensor_name in ['x_seq', 'v', 's_seq', 'v_seq']:
            ret += check_function.check_cuda_and_contiguous(tensor_name)
            ret += '\n'

        # add extra codes
        ret += extra_codes

        # define variables for CUDA kernel
        ret += f'const int seq_len = x_seq.size(0);\n' \
               f'const int size = x_seq.numel();\n' \
               f'const int threads = {default_threads};\n' \
               f'const int neuron_num = size / seq_len;\n' \
               f'const int blocks = (neuron_num + threads - 1) / threads;\n'

        # set device
        ret += check_function.check_cuda_operation('cudaSetDevice(x_seq.get_device())')
        ret += '\n'

        # start cuda kernel
        ret += 'if (x_seq.scalar_type() == c10::ScalarType::Float)\n'
        ret += '{\n'

        if hard_reset:
            ret += f'{neuron_name}_hard_reset_fptt_kernel<<<blocks, threads>>>(x_seq.data_ptr<float>(), s_seq.data_ptr<float>(), v_seq.data_ptr<float>(), v_th, v_reset, neuron_num, size{utils.add_comma(extra_cuda_kernel_args)});\n'
        else:
            ret += f'{neuron_name}_soft_reset_fptt_kernel<<<blocks, threads>>>(x_seq.data_ptr<float>(), s_seq.data_ptr<float>(), v_seq.data_ptr<float>(), v_th, neuron_num, size{utils.add_comma(extra_cuda_kernel_args)});\n'

        ret += '}\n'
        ret += 'else if (x_seq.scalar_type() == c10::ScalarType::Half)\n'
        ret += '{\n'

        if hard_reset:
            ret += f'{neuron_name}_hard_reset_fptt_kernel_half<<<blocks, threads>>>(x_seq.data_ptr<at::Half>(), s_seq.data_ptr<at::Half>(), v_seq.data_ptr<at::Half>(), __float2half(v_th), __float2half(v_reset), neuron_num, size{utils.add_comma(extra_cuda_kernel_half_args)});\n'
        else:
            ret += f'{neuron_name}_soft_reset_fptt_kernel_half<<<blocks, threads>>>(x_seq.data_ptr<at::Half>(), s_seq.data_ptr<at::Half>(), v_seq.data_ptr<at::Half>(), __float2half(v_th), neuron_num, size{utils.add_comma(extra_cuda_kernel_half_args)});\n'

        ret += '}\n'




        ret += 'return {s_seq, v_seq.index({torch::indexing::Slice(1, torch::indexing::None)})};\n'

        ret += '}\n\n'

        return ret


    @staticmethod
    def impl_cuda_kernel(neuron_name, hard_reset, charge_codes, charge_codes_half, extra_cuda_kernel_args='', extra_cuda_kernel_half_args=''):
        if hard_reset:
            ret = f'__global__ void {neuron_name}_hard_reset_fptt_kernel(const float* __restrict__ x_seq, float* __restrict__ s_seq, float* __restrict__ v_seq, const float v_th, const float v_reset, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_args)})\n'
        else:
            ret = f'__global__ void {neuron_name}_soft_reset_fptt_kernel(const float* __restrict__ x_seq, float* __restrict__ s_seq, float* __restrict__ v_seq, const float v_th, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_args)})\n'

        ret += '{\n'


        ret += '''
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {
            const int dt = neuron_num;
            for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
            {
                    const int t = index + mem_offset;
        '''

        ret += f'const float h = {charge_codes};\n'

        if hard_reset:
            ret += '''
                    if (h >= v_th)
                    {
                        s_seq[t] = 1.0f;
                        v_seq[t + dt] = v_reset;
                    }
                    '''
        else:
            ret += '''
                    if (h >= v_th)
                    {
                        s_seq[t] = 1.0f;
                        v_seq[t + dt] -= v_th;
                    }
                    '''


        ret += '''
                    else
                    {
                        s_seq[t] = 0.0f;
                        v_seq[t + dt] = h;
                    }
                    
                }
                
            }
        '''

        ret += '\n}\n\n'


        # half precision

        if hard_reset:
            ret += f'__global__ void {neuron_name}_hard_reset_fptt_kernel_half(const at::Half* __restrict__ x_seq, at::Half* __restrict__ s_seq, at::Half* __restrict__ v_seq, const half v_th, const half v_reset, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_half_args)})\n'
        else:
            ret += f'__global__ void {neuron_name}_soft_reset_fptt_kernel_half(const at::Half* __restrict__ x_seq, at::Half* __restrict__ s_seq, at::Half* __restrict__ v_seq, const half v_th, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_half_args)})\n'

        ret += '{\n'

        ret += '''
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {
            const int dt = neuron_num;
            for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
            {
                    const int t = index + mem_offset;
        '''

        ret += f'const half h = {charge_codes_half};\n'

        if hard_reset:
            ret += '''
                    if (__hgeu(h, v_th))
                    {
                        s_seq[t] = __float2half(1.0f);
                        v_seq[t + dt] = v_reset;
                    }
                    '''
        else:
            ret += '''
                    if (__hgeu(h, v_th))
                    {
                        s_seq[t] = __float2half(1.0f);
                        v_seq[t + dt] -= v_th;
                    }
                    '''

        ret += '''
                    else
                    {
                        s_seq[t] = __float2half(0.0f);
                        v_seq[t + dt] = h;
                    }

                }

            }
        '''

        ret += '\n}\n\n'
        return ret

    @staticmethod
    def create_IFNode():
        neuron_name = 'IFNode'
        cpp_file = f'./csrc/neuron/{neuron_name}.cpp'
        cu_file = f'./csrc/neuron/{neuron_name}.cu'
        cext_fun_names = []
        codes = utils.add_include(c_headers)

        for hard_reset in [True, False]:
            codes += NeuronFPTT.def_fptt(neuron_name=neuron_name, hard_reset=hard_reset) + ';\n\n'
            cext_fun_names.append(NeuronFPTT.pybind_name(neuron_name=neuron_name, hard_reset=hard_reset))
        codes += utils.bind_fun(cext_fun_names)
        utils.write_str_to_file(cpp_file, codes)

        charge_codes = 'v_seq[t] + x_seq[t]'
        charge_codes_half = '__hadd((half)v_seq[t], (half)x_seq[t])'
        codes = utils.add_include(cuda_headers)
        for hard_reset in [True, False]:
            codes += NeuronFPTT.impl_cuda_kernel(neuron_name=neuron_name, hard_reset=hard_reset, charge_codes=charge_codes, charge_codes_half=charge_codes_half)
            codes += NeuronFPTT.impl_fptt(neuron_name=neuron_name, hard_reset=hard_reset)
        utils.write_str_to_file(cu_file, codes)
        return cpp_file, cu_file

    @staticmethod
    def test_IFNode():

        def py_if_fptt(x_seq: torch.Tensor, v: torch.Tensor, v_th, v_reset):
            v_seq = torch.cat((v.unsqueeze(0), torch.zeros_like(x_seq.data)), 0)
            s_seq = torch.zeros_like(x_seq.data)
            for t in range(x_seq.shape[0]):
                h = v_seq[t] + x_seq[t]
                s_seq[t] = (h >= v_th).to(h)
                if v_reset is None:
                    v_seq[t + 1] = h - s_seq[t]
                else:
                    v_seq[t + 1] = v_reset * s_seq[t] + h * (1. - s_seq[t])

            return s_seq, v_seq[1:]


        test_cext = cpp_extension.load(name='test_cext_if',
                                       sources=list(NeuronFPTT.create_IFNode()),
                                       verbose=True)

        with torch.no_grad():
            device = 'cuda:0'
            T = 8
            N = 4
            x_seq = torch.rand([T, N], device=device)
            v = torch.rand([N], device=device)
            v_th = 1.
            v_reset = 0.

            s_seq_py, v_seq_py = py_if_fptt(x_seq, v, v_th, v_reset)
            s_seq, v_seq = test_cext.IFNode_hard_reset_fptt(x_seq, v, v_th, v_reset)

            print('fp32 cuda-py error', utils.max_error(s_seq_py, s_seq), utils.max_error(v_seq_py, v_seq))


            v_th = 1.
            v_reset = 0.
            s_seq, v_seq = test_cext.IFNode_hard_reset_fptt(x_seq.half(), v.half(), v_th, v_reset)
            print('fp16 cuda-py error',utils.max_error(s_seq_py, s_seq), utils.max_error(v_seq_py, v_seq))

    @staticmethod
    def create_LIFNode():
        neuron_name = 'LIFNode'
        cpp_file = f'./csrc/neuron/{neuron_name}.cpp'
        cu_file = f'./csrc/neuron/{neuron_name}.cu'
        cext_fun_names = []
        codes = utils.add_include(c_headers)

        extra_arg = 'const float & reciprocal_tau'

        for hard_reset in [True, False]:
            codes += NeuronFPTT.def_fptt(neuron_name=neuron_name, hard_reset=hard_reset, extra_arg=extra_arg) + ';\n\n'
            cext_fun_names.append(NeuronFPTT.pybind_name(neuron_name=neuron_name, hard_reset=hard_reset))
        codes += utils.bind_fun(cext_fun_names)
        utils.write_str_to_file(cpp_file, codes)

        extra_cuda_kernel_args = 'const float reciprocal_tau'
        extra_cuda_kernel_half_args = 'const half reciprocal_tau'
        codes = utils.add_include(cuda_headers)
        for hard_reset in [True, False]:
            if hard_reset:
                charge_codes = 'v_seq[t] + reciprocal_tau * (x_seq[t] - v_seq[t] + v_reset)'
                charge_codes_half = '__hfma(reciprocal_tau, __hadd(__hsub(x_seq[t], v_seq[t]), v_reset), v_seq[t])'
            else:
                charge_codes = 'v_seq[t] + reciprocal_tau * (x_seq[t] - v_seq[t])'
                charge_codes_half = '__hfma(reciprocal_tau, __hsub(x_seq[t], v_seq[t]), v_seq[t])'

            codes += NeuronFPTT.impl_cuda_kernel(neuron_name=neuron_name, hard_reset=hard_reset, charge_codes=charge_codes, charge_codes_half=charge_codes_half, extra_cuda_kernel_args=extra_cuda_kernel_args, extra_cuda_kernel_half_args=extra_cuda_kernel_half_args)
            codes += NeuronFPTT.impl_fptt(neuron_name=neuron_name, hard_reset=hard_reset, extra_arg=extra_arg, extra_cuda_kernel_args='reciprocal_tau', extra_cuda_kernel_half_args='__float2half(reciprocal_tau)')
        utils.write_str_to_file(cu_file, codes)
        return cpp_file, cu_file

    @staticmethod
    def test_LIFNode():

        def py_lif_fptt(x_seq: torch.Tensor, v: torch.Tensor, v_th, v_reset, reciprocal_tau):
            v_seq = torch.cat((v.unsqueeze(0), torch.zeros_like(x_seq.data)), 0)
            s_seq = torch.zeros_like(x_seq.data)
            for t in range(x_seq.shape[0]):
                if v_reset is None:
                    h = v_seq[t] + reciprocal_tau * (x_seq[t] - v_seq[t])
                else:
                    h = v_seq[t] + reciprocal_tau * (x_seq[t] - v_seq[t] + v_reset)

                s_seq[t] = (h >= v_th).to(h)
                if v_reset is None:
                    v_seq[t + 1] = h - s_seq[t]
                else:
                    v_seq[t + 1] = v_reset * s_seq[t] + h * (1. - s_seq[t])

            return s_seq, v_seq[1:]


        test_cext = cpp_extension.load(name='test_cext_lif',
                                       sources=list(NeuronFPTT.create_LIFNode()),
                                       verbose=True)

        with torch.no_grad():
            device = 'cuda:0'
            T = 8
            N = 4
            tau = 100.
            reciprocal_tau = 1 / tau
            x_seq = torch.rand([T, N], device=device)
            v = torch.rand([N], device=device)
            v_th = 1.
            v_reset = 0.

            s_seq_py, v_seq_py = py_lif_fptt(x_seq, v, v_th, v_reset, reciprocal_tau)
            s_seq, v_seq = test_cext.LIFNode_hard_reset_fptt(x_seq, v, v_th, v_reset, reciprocal_tau)

            print('fp32 cuda-py error', utils.max_error(s_seq_py, s_seq), utils.max_error(v_seq_py, v_seq))


            v_th = 1.
            v_reset = 0.
            s_seq, v_seq = test_cext.LIFNode_hard_reset_fptt(x_seq.half(), v.half(), v_th, v_reset, reciprocal_tau)
            print('fp16 cuda-py error',utils.max_error(s_seq_py, s_seq), utils.max_error(v_seq_py, v_seq))

NeuronFPTT.test_LIFNode()