from typing import Union
import torch
import torch.nn.functional as F
import threading
from ... import configure
from . import cuda_utils
import logging

try:
    import cupy
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.tensor_cache: {e}")
    cupy = None


class DataTypeConvertCUDACode:
    float2bool = r"""
    extern "C" __global__
            void float2bool(const float* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) fs[mem_offset + i]) << i);
                    }
                }
            }
    """

    half2bool = r"""
    #include <cuda_fp16.h>
    extern "C" __global__
            void half2bool(const half* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) __half2float(fs[mem_offset + i])) << i);
                    }
                }
            }
    """

    bool2float = r"""
    extern "C" __global__
            void bool2float(const unsigned char* bs, float* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = (float) (compressed_v % 2);
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    """

    bool2half = r"""
    #include <cuda_fp16.h>
    extern "C" __global__
            void bool2half(const unsigned char* bs, half* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = __float2half((float) (compressed_v % 2));
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    """


def float_spike_to_bool(spike: torch.Tensor):
    """
    **API Language:**
    :ref:`中文 <float_spike_to_bool-cn>` | :ref:`English <float_spike_to_bool-en>`

    ----

    .. _float_spike_to_bool-cn:

    * **中文**

    将浮点脉冲张量 ``spike`` 压缩为 ``torch.uint8`` 张量。压缩后的每个元素保存原始
    脉冲中的8个元素（二值位打包）。

    :param spike: 浮点脉冲张量， ``dtype`` 必须为 ``torch.float`` 或 ``torch.half`` ，
        且元素取值为0或1
    :type spike: torch.Tensor

    :return: ``(spike_b, s_dtype, s_shape, s_padding)``

        - ``spike_b``: 压缩后的张量， ``dtype=torch.uint8`` ，每个元素保存8个脉冲
        - ``s_dtype``: 原始脉冲张量的数据类型
        - ``s_shape``: 原始脉冲张量的形状
        - ``s_padding``: 为对齐到8的倍数所填充的元素数量
    :rtype: tuple

    ----

    .. _float_spike_to_bool-en:

    * **English**

    Compress a floating-point spike tensor ``spike`` into a ``torch.uint8`` tensor.
    Each element in the compressed tensor stores 8 binary spikes (bit packing).

    :param spike: Spike tensor whose ``dtype`` is ``torch.float`` or ``torch.half``,
        and all elements are 0 or 1
    :type spike: torch.Tensor

    :return: ``(spike_b, s_dtype, s_shape, s_padding)``

        - ``spike_b``: Compressed spike tensor with ``dtype=torch.uint8`` and each
          element storing 8 spikes
        - ``s_dtype``: Dtype of the original spike tensor
        - ``s_shape``: Shape of the original spike tensor
        - ``s_padding``: Number of padded elements used for 8-element alignment
    :rtype: tuple
    """
    s_dtype = spike.dtype
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.float2bool
        kernel_name = "float2bool"
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.half2bool
        kernel_name = "half2bool"
    else:
        raise NotImplementedError

    s_shape = spike.shape

    spike = spike.flatten()
    s_padding = 8 - spike.numel() % 8
    if s_padding != 0 and s_padding != 8:
        spike = F.pad(spike, (0, s_padding))
    device_id = spike.get_device()
    spike_b = torch.zeros([spike.numel() // 8], device=spike.device, dtype=torch.uint8)

    if device_id >= 0 and cupy is not None:
        with cuda_utils.DeviceEnvironment(device_id):
            numel = spike_b.numel()
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            spike, spike_b, numel = cuda_utils.get_contiguous(spike, spike_b, numel)
            kernel_args = [spike, spike_b, numel]
            kernel = cupy.RawKernel(
                kernel_codes,
                kernel_name,
                options=configure.cuda_compiler_options,
                backend=configure.cuda_compiler_backend,
            )
            kernel(
                (blocks,),
                (configure.cuda_threads,),
                cuda_utils.wrap_args_to_raw_kernel(device_id, *kernel_args),
            )
    else:
        spike = spike.view(-1, 8).to(torch.uint8)
        for i in range(8):
            spike_b += spike[:, i] << i

    return spike_b, s_dtype, s_shape, s_padding


def bool_spike_to_float(
    spike_b: torch.Tensor, s_dtype: torch.dtype, s_shape: torch.Size, s_padding: int = 0
):
    """
    **API Language:**
    :ref:`中文 <bool_spike_to_float-cn>` | :ref:`English <bool_spike_to_float-en>`

    ----

    .. _bool_spike_to_float-cn:

    * **中文**

    将压缩后的 ``torch.uint8`` 脉冲张量解压为原始浮点脉冲张量，
    并按 ``s_shape`` 恢复形状。

    :param spike_b: 压缩后的脉冲张量， ``dtype=torch.uint8`` ，
        每个元素保存8个脉冲
    :type spike_b: torch.Tensor

    :param s_dtype: 原始脉冲张量的数据类型
    :type s_dtype: torch.dtype

    :param s_shape: 原始脉冲张量的形状
    :type s_shape: torch.Size

    :param s_padding: 压缩时为对齐到8的倍数而填充的元素数量
    :type s_padding: int

    :return: 解压并恢复形状后的原始脉冲张量
    :rtype: torch.Tensor

    ----

    .. _bool_spike_to_float-en:

    * **English**

    Decompress a packed ``torch.uint8`` spike tensor back to the original
    floating-point spike tensor and reshape it with ``s_shape``.

    :param spike_b: Compressed spike tensor with ``dtype=torch.uint8`` and each
        element storing 8 spikes
    :type spike_b: torch.Tensor

    :param s_dtype: Dtype of the original spike tensor
    :type s_dtype: torch.dtype

    :param s_shape: Shape of the original spike tensor
    :type s_shape: torch.Size

    :param s_padding: Number of padded elements used for 8-element alignment
        during compression
    :type s_padding: int

    :return: The decompressed spike tensor restored to ``s_shape``
    :rtype: torch.Tensor
    """
    device_id = spike_b.get_device()
    spike = torch.zeros(spike_b.numel() * 8, device=spike_b.device, dtype=s_dtype)
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.bool2float
        kernel_name = "bool2float"
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.bool2half
        kernel_name = "bool2half"
    else:
        raise NotImplementedError

    if device_id >= 0 and cupy is not None:
        with cuda_utils.DeviceEnvironment(device_id):
            numel = spike_b.numel()
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            spike_b, spike, numel = cuda_utils.get_contiguous(spike_b, spike, numel)
            kernel_args = [spike_b, spike, numel]
            kernel = cupy.RawKernel(
                kernel_codes,
                kernel_name,
                options=configure.cuda_compiler_options,
                backend=configure.cuda_compiler_backend,
            )
            kernel(
                (blocks,),
                (configure.cuda_threads,),
                cuda_utils.wrap_args_to_raw_kernel(device_id, *kernel_args),
            )
    else:
        spike = spike.view(-1, 8)
        for i in range(8):
            spike[:, i] = spike_b % 2
            spike_b = spike_b >> 1

    if s_padding != 0 and s_padding != 8:
        spike = spike[0 : spike.numel() - s_padding]
    return spike.reshape(s_shape)


def tensor_key(x: torch.Tensor):
    x = x.flatten()
    return x.data_ptr(), x[-1].data_ptr(), x.numel()


class BoolTensorCache:
    def __init__(self):
        super().__init__()
        self.cache_dict = {}
        self.cache_refcount_dict = {}
        self.lock = threading.Lock()

    def store_bool(self, spike: Union[torch.FloatTensor, torch.HalfTensor]):
        tk = tensor_key(spike)

        self.lock.acquire()
        if tk not in self.cache_dict:
            if configure.save_bool_spike_level == 0:
                self.cache_dict[tk] = (spike.bool(), spike.dtype)
            elif configure.save_bool_spike_level == 1:
                self.cache_dict[tk] = float_spike_to_bool(spike)
            else:
                raise NotImplementedError
            self.cache_refcount_dict[tk] = 1
        else:
            self.cache_refcount_dict[tk] += 1
        self.lock.release()

        return tk

    def get_float(self, tk, spike_shape: torch.Size):
        if configure.save_bool_spike_level == 0:
            spike, s_dtype = self.cache_dict[tk]
            spike = spike.to(s_dtype)
        elif configure.save_bool_spike_level == 1:
            spike = bool_spike_to_float(*self.cache_dict[tk])
        else:
            raise NotImplementedError

        self.lock.acquire()
        self.cache_refcount_dict[tk] -= 1
        if self.cache_refcount_dict[tk] == 0:
            del self.cache_refcount_dict[tk]
            del self.cache_dict[tk]
        self.lock.release()

        return spike.view(spike_shape)


BOOL_TENSOR_CACHE = BoolTensorCache()
