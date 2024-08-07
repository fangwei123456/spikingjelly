import logging
import torch
import time
import numpy as np
from .. import configure
from typing import Callable, Union
try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.cuda_utils: {e}')
    cupy = None

def cpu_timer(f: Callable, *args, **kwargs):
    """
    * :ref:`API in English <cpu_timer-en>`

    .. _cpu_timer-cn:

    计算在CPU上执行 ``f(*args, **kwargs)`` 所需的时间

    :param f: 函数
    :type f: Callable
    :return: 用时，单位是毫秒
    :rtype: float

    * :ref:`中文 API <cpu_timer-cn>`

    .. _cpu_timer-en:

    Returns the used time for calling ``f(*args, **kwargs)`` in CPU

    :param f: a function
    :type f: Callable
    :return: used time in milliseconds
    :rtype: float
    """
    start = time.perf_counter()
    f(*args, **kwargs)
    return time.perf_counter() - start

def cuda_timer(device: Union[torch.device, int], f: Callable, *args, **kwargs):
    """
    * :ref:`API in English <cuda_timer-en>`

    .. _cuda_timer-cn:

    计算在CUDA上执行 ``f(*args, **kwargs)`` 所需的时间

    :param device: ``f`` 运行的CUDA设备
    :type device: Union[torch.device, int]
    :param f: 函数
    :type f: Callable
    :return: 用时，单位是毫秒
    :rtype: float

    * :ref:`中文 API <cuda_timer-cn>`

    .. _cuda_timer-en:

    Returns the used time for calling ``f(*args, **kwargs)`` in CUDA

    :param device: on which cuda device that ``f`` is running
    :type device: Union[torch.device, int]
    :param f: a function
    :type f: Callable
    :return: used time in milliseconds
    :rtype: float
    """
    torch.cuda.set_device(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f(*args, **kwargs)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)

def cal_fun_t(n: int, device: Union[str, torch.device, int], f: Callable, *args, **kwargs):
    """
    * :ref:`API in English <cal_fun_t-en>`

    .. _cal_fun_t-cn:

    测量在 ``device`` 上执行 ``n`` 次 ``f(*args, **kwargs)`` 的平均用时

    .. note::

        当 ``n > 1`` 时，实际上会执行 ``2n`` 次，然后返回后 ``n`` 次的平均用时，以减小误差。

    :param n: 重复的次数
    :type n: int
    :param device: ``f`` 执行的设备，可以为 'cpu' 或CUDA设备
    :type device: Union[str, torch.device, int]
    :param f: 函数
    :type f: Callable
    :return: 用时，单位是毫秒
    :rtype: float

    * :ref:`中文 API <cal_fun_t-cn>`

    .. _cal_fun_t-en:

    Returns the used time averaged by calling ``f(*args, **kwargs)`` over ``n`` times

    .. admonition:: Note
        :class: note

        If ``n > 1``, this function will call ``f`` for ``2n`` times and return the average used time by the last ``n``
        times to reduce the measure error.

    :param n: repeat times
    :type n: int
    :param device: on which cuda device that ``f`` is running. It can be 'cpu' or a cuda deivce
    :type device: Union[str, torch.device, int]
    :param f: function
    :type f: Callable
    :return: used time in milliseconds
    :rtype: float

    """
    if n == 1:
        if device == 'cpu':
            return cpu_timer(f, *args, **kwargs)
        else:
            return cuda_timer(device, f, *args, **kwargs)

    # warm up
    if device == 'cpu':
        cpu_timer(f, *args, **kwargs)
    else:
        cuda_timer(device, f, *args, **kwargs)

    t_list = []
    for _ in range(n * 2):
        if device == 'cpu':
            ti = cpu_timer(f, *args, **kwargs)
        else:
            ti = cuda_timer(device, f, *args, **kwargs)
        t_list.append(ti)


    t_list = np.asarray(t_list)
    return t_list[n:].mean()

def cal_blocks(numel: int, threads: int = -1):
    """
    * :ref:`API in English <cal_blocks-en>`

    .. _cal_blocks-cn:

    :param numel: 并行执行的CUDA内核的数量
    :type numel: int
    :param threads: 每个cuda block中threads的数量，默认为-1，表示使用 ``configure.cuda_threads``
    :type threads: int
    :return: blocks的数量
    :rtype: int

    此函数返回 blocks的数量，用来按照 ``kernel((blocks,), (configure.cuda_threads,), ...)`` 调用 :class:`cupy.RawKernel`

    * :ref:`中文 API <cal_blocks-cn>`

    .. _cal_blocks-en:

    :param numel: the number of parallel CUDA kernels
    :type numel: int
    :param threads: the number of threads in each cuda block.
        The defaule value is -1, indicating to use ``configure.cuda_threads``
    :type threads: int
    :return: the number of blocks
    :rtype: int

    Returns the number of blocks to call :class:`cupy.RawKernel` by ``kernel((blocks,), (threads,), ...)``

    """
    if threads == -1:
        threads = configure.cuda_threads
    return (numel + threads - 1) // threads

def get_contiguous(*args):
    """
    * :ref:`API in English <get_contiguous-en>`

    .. _get_contiguous-cn:

    将 ``*args`` 中所有的 ``torch.Tensor`` 或 ``cupy.ndarray`` 进行连续化。

    .. note::

        连续化的操作无法in-place，因此本函数返回一个新的list。

    :return: 一个元素全部为连续的 ``torch.Tensor`` 或 ``cupy.ndarray`` 的 ``list``
    :rtype: list

    * :ref:`中文 API <get_contiguous-cn>`

    .. _get_contiguous-en:

    :return: a list that contains the contiguous ``torch.Tensor`` or ``cupy.ndarray``
    :rtype: list

    Makes ``torch.Tensor`` or ``cupy.ndarray`` in ``*args`` to be contiguous

    .. admonition:: Note
        :class: note

        The making contiguous operation can not be done in-place. Hence, this function will return a new list.

    """
    ret_list = []

    for item in args:
        if isinstance(item, torch.Tensor):
            ret_list.append(item.contiguous())

        elif isinstance(item, cupy.ndarray):
            ret_list.append(cupy.ascontiguousarray(item))
        else:
            raise TypeError(type(item))
    return ret_list

def wrap_args_to_raw_kernel(device: int, *args):
    """
    * :ref:`API in English <wrap_args_to_raw_kernel-en>`

    .. _wrap_args_to_raw_kernel-cn:

    :param device: raw kernel运行的CUDA设备
    :type device: int
    :return: 一个包含用来调用 :class:`cupy.RawKernel` 的 ``tuple``
    :rtype: tuple

    此函数可以包装 ``torch.Tensor`` 和 ``cupy.ndarray`` 并将其作为 :class:`cupy.RawKernel.__call__` 的 ``args``

    * :ref:`中文 API <wrap_args_to_raw_kernel-cn>`

    .. _wrap_args_to_raw_kernel-en:

    :param device: on which CUDA device the raw kernel will run
    :type device: int
    :return: a ``tuple`` that contains args to call :class:`cupy.RawKernel`
    :rtype: tuple

    This function can wrap ``torch.Tensor`` or ``cupy.ndarray`` to ``args`` in :class:`cupy.RawKernel.__call__`

    """
    # note that the input must be contiguous
    # check device and get data_ptr from tensor
    ret_list = []
    for item in args:
        if isinstance(item, torch.Tensor):
            assert item.get_device() == device
            assert item.is_contiguous()
            ret_list.append(item.data_ptr())

        elif isinstance(item, cupy.ndarray):
            assert item.device.id == device
            assert item.flags['C_CONTIGUOUS']
            ret_list.append(item)

        else:
            raise TypeError
    return tuple(ret_list)

class DeviceEnvironment:
    def __init__(self, device: int):
        """
        * :ref:`API in English <DeviceEnvironment.__init__-en>`

        .. _DeviceEnvironment.__init__-cn:

        这个模块可以被用作在指定的 ``device`` 上执行CuPy函数的上下文，用来避免 `torch.cuda.current_device()` 被CuPy意外改变( https://github.com/cupy/cupy/issues/6569 )。

        代码示例：

        .. code-block:: python

            with DeviceEnvironment(device):
                kernel((blocks,), (configure.cuda_threads,), ...)


        * :ref:`中文 API <DeviceEnvironment.__init__-cn>`

        .. _DeviceEnvironment.__init__-en:

        :param device: the CUDA device
        :type device: int

        This module is used as a context to make CuPy use the specific device, and avoids `torch.cuda.current_device()` is changed by CuPy ( https://github.com/cupy/cupy/issues/6569 ).

        Codes example:

        .. code-block:: python

            with DeviceEnvironment(device):
                kernel((blocks,), (configure.cuda_threads,), ...)

        """
        self.device = device
        self.previous_device = None

    def __enter__(self):
        current_device = torch.cuda.current_device()
        if current_device != self.device:
            torch.cuda.set_device(self.device)
            self.previous_device = current_device

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_device is not None:
            torch.cuda.set_device(self.previous_device)

