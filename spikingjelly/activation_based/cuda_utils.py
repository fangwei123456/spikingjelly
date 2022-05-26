import logging
import torch
import time
import numpy as np
from .. import configure
import os
import threading
import datetime
from torch.utils.tensorboard import SummaryWriter
import re
try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.cuda_utils: {e}')
    pass

def cpu_timer(f, *args, **kwargs):
    start = time.perf_counter()
    f(*args, **kwargs)
    return time.perf_counter() - start

def cuda_timer(device, f, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f(*args, **kwargs)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)

def cal_fun_t(n, device, f, *args, **kwargs):
    if device == 'cpu':
        c_timer = cpu_timer
    else:
        c_timer = cuda_timer

    if n == 1:
        return c_timer(device, f, *args, **kwargs)

    # warm up
    c_timer(device, f, *args, **kwargs)

    t_list = []
    for _ in range(n * 2):
        t_list.append(c_timer(device, f, *args, **kwargs))
    t_list = np.asarray(t_list)
    return t_list[n:].mean()

def cal_blocks(numel: int):
    return (numel + configure.cuda_threads - 1) // configure.cuda_threads

def get_contiguous(*args):
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
        This module is used as a context to make CuPy use the specific device, and avoids `torch.cuda.current_device()` is changed by CuPy.
        Refer to https://github.com/cupy/cupy/issues/6569 for more details.
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

