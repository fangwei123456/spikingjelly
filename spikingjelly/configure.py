'''
This py file defines some variables used in SpikingJelly.
Here is an example of how you can change them to make effect in your codes:

    import spikingjelly
    spikingjelly.configure.cuda_threads = 512

Do not change them in this way, which will not make effect:

    from spikingjelly.configure import cuda_threads
    cuda_threads = 512

'''
max_threads_number_for_datasets_preprocess = 16
'''
`max_threads_number_for_datasets_preprocess` defines the maximum threads for datasets preprocessing, which is 
1. reading binary events and saving them to numpy format
2. integrating events to frames.

Note that a too larger `max_threads_number_for_datasets_preprocess` will overload the disc and slow down the speed.
'''

cuda_threads = 1024
'''
`cuda_threads` defines the default threads number for CUDA kernel.

It is recommended that `cuda_threads` is the power of 2.
'''

cuda_compiler_options = ('-use_fast_math',)
'''
`cuda_compiler_options` defines the compiler options passed to the backend (NVRTC or NVCC). 

For more details, refer to 
1. https://docs.nvidia.com/cuda/nvrtc/index.html#group__options 
2. https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description
3. https://github.com/fangwei123456/spikingjelly/discussions/116
'''

cuda_compiler_backend = 'nvrtc'
'''
`cuda_compiler_backend` defines the compiler for CUDA(cupy).

It can be set to 'nvcc' or 'nvrtc'.
'''

save_datasets_compressed = True
'''
If `save_datasets_compressed == True`, events and frames in spikingjelly.datasets will be saved in compressed npz format.

The compressed npz file consumes less memory in disk but more time in reading.
'''

save_spike_as_bool_in_neuron_kernel = False
'''
If `save_spike_as_bool_in_neuron_kernel == True`, the neuron kernel used in the neuron's cupy backend will save the spike as a bool, rather than float/half tensor for backward, which can reduce the memory consumption.
'''

save_bool_spike_level = 0
'''
`save_bool_spike_level` take effects on SpikeConv/SpikeLinear, and on neuron's cupy kernel when `save_spike_as_bool_in_neuron_kernel == True`.

If `save_bool_spike_level == 0`, spikes will be saved in bool. Note that bool uses 8-bit, rather than 1-bit.

If `save_bool_spike_level == 1`, spikes will be saved in uint8 with each 8-bit storing 8 spikes.

A larger `save_bool_spike_level` means less memory consumption but slower speed.
'''

