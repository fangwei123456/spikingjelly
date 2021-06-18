from spikingjelly.cext.autogen import check_function, utils
import os

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
    def impl_fptt(neuron_name, hard_reset, extra_cuda_kernel_args='', extra_codes=''):
        # define the function name

        # all inputs will be regarded as [T, N] = [seq_len, neuron_num]
        ret = NeuronFPTT.def_fptt(neuron_name, hard_reset)
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
        if hard_reset:
            ret += f'{neuron_name}_hard_reset_fptt_kernel(x_seq.data_ptr<float>(), s_seq.data_ptr<float>(), v_seq.data_ptr<float>(), v_th, v_reset, neuron_num, size{utils.add_comma(extra_cuda_kernel_args)});\n'
        else:
            ret += f'{neuron_name}_soft_reset_fptt_kernel(x_seq.data_ptr<float>(), s_seq.data_ptr<float>(), v_seq.data_ptr<float>(), v_th, neuron_num, size{utils.add_comma(extra_cuda_kernel_args)});\n'


        ret += 'return {s_seq, v_seq.index({torch::indexing::Slice(1, torch::indexing::None)})};\n'

        ret += '}\n\n'

        return ret


    @staticmethod
    def impl_cuda_kernel(neuron_name, hard_reset, charge_codes, extra_cuda_kernel_args=''):
        if hard_reset:
            ret = f'__global__ void {neuron_name}_hard_reset_fptt_kernel(const float* __restrict__ x_seq, float* __restrict__ s_seq, float* __restrict__ v_seq, const float v_th, const float v_reset, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_args)})\n'
        else:
            ret = f'{neuron_name}_soft_reset_fptt_kernel(const float* __restrict__ x_seq, float* __restrict__ s_seq, float* __restrict__ v_seq, const float v_th, const int neuron_num, const int size{utils.add_comma(extra_cuda_kernel_args)})\n'

        ret += '{\n'


        ret += '''
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const init dt = neuron_num;
        if (index < neuron_num)
        {
            for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
            {
                    const int t = index + mem_offset;
        '''

        ret += f'const float h = {charge_codes};\n'

        if hard_reset:
            ret += '''
                    if (h >= v_th)
                    {
                        spike_seq[t] = 1.0f;
                        v_seq[t + dt] = v_reset;
                    }
                    '''
        else:
            ret += '''
                    if (h >= v_th)
                    {
                        spike_seq[t] = 1.0f;
                        v_seq[t + dt] -= v_th;
                    }
                    '''


        ret += '''
                    else
                    {
                        spike_seq[t] = 0.0f;
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
        cext_fun_names = []
        codes = utils.add_include(c_headers)

        for hard_reset in [True, False]:
            codes += NeuronFPTT.def_fptt(neuron_name=neuron_name, hard_reset=hard_reset) + ';\n\n'
            cext_fun_names.append(NeuronFPTT.pybind_name(neuron_name=neuron_name, hard_reset=hard_reset))
        codes += utils.bind_fun(cext_fun_names)
        utils.write_str_to_file(f'./csrc/neuron/{neuron_name}.cpp', codes)
        charge_codes = 'v_seq[t] + x_seq[t]'
        codes = utils.add_include(cuda_headers)


        for hard_reset in [True, False]:
            codes += NeuronFPTT.impl_cuda_kernel(neuron_name=neuron_name, hard_reset=True, charge_codes=charge_codes)
            codes += NeuronFPTT.impl_fptt(neuron_name=neuron_name, hard_reset=hard_reset)
        utils.write_str_to_file(f'./csrc/neuron/{neuron_name}.cu', codes)