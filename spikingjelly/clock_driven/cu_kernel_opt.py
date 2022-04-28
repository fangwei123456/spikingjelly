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
    logging.info(f'spikingjelly.clock_driven.cu_kernel_opt: {e}')
    pass

def cuda_timer(device, f, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f(*args, **kwargs)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)

def cal_fun_t(n, device, f, *args, **kwargs):
    assert n > 2
    # warm up
    cuda_timer(device, f, *args, **kwargs)

    t_list = []
    for _ in range(n * 2):
        t_list.append(cuda_timer(device, f, *args, **kwargs))
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
            raise TypeError
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

class GPUMonitor(threading.Thread):
    def __init__(self, log_dir: str = None, gpu_ids: tuple = (0, ), interval: float = 60., start_now=True):
        """
        :param log_dir: the directory for saving logs with tensorboard. If it is None, this module will print logs
        :type log_dir: str
        :param gpu_ids: the id of GPUs to be monitored, e.g., `(0, 1, 2, 3)`. The default value is `(0, )`
        :type gpu_ids: tuple
        :param interval: the recording interval (in seconds)
        :type interval: float
        :param start_now: if true, the monitor will start to record now. Otherwise, it will start after the user call `start()` manually
        :type start_now:

        The GPU monitor, which starts a new thread to record the utilization and memory used of `gpu_ids` every `interval` seconds.

        .. admonition:: Warning
        :class: warning

            Do not forget to call `stop()` after the main thread finishes its job, otherwise the main thread will never stop!

        Codes example:

        .. code-block:: python

            import time

            gm = GPUMonitor(interval=1)
            time.sleep(2)  # make the main thread sleep
            gm.stop()

            # The outputs are:

            # 2022-04-28 10:52:25
            # utilization.gpu [%], memory.used [MiB]
            # 0 %, 376 MiB
        """
        super().__init__()
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.stopped = False
        self.cmds = 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
        self.cmds += ' -i '
        id_str = []
        for gpu_id in self.gpu_ids:
            id_str.append(str(gpu_id))
        self.cmds += ','.join(id_str)
        self.step = 0

        if log_dir is None:
            self.writer = None
        else:
            self.writer = []
            for i in range(self.gpu_ids.__len__()):
                self.writer.append(SummaryWriter(os.path.join(log_dir, f'gpu_{id_str[i]}')))

        if start_now:
            self.start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            with os.popen(self.cmds) as fp:
                outputs = fp.read()
                if self.writer is not None:
                    outputs = outputs.split('\n')[1:-1]
                    # skip the first row 'utilization.gpu [%], memory.used [MiB]' and the last row ('\n')
                    for i in range(outputs.__len__()):
                        utilization_memory = re.findall(r'\d+', outputs[i])
                        utilization = int(utilization_memory[0])
                        memory_used = int(utilization_memory[1])
                        self.writer[i].add_scalar('utilization', utilization, self.step)
                        self.writer[i].add_scalar('memory_used', memory_used, self.step)
                else:
                    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print(outputs)
                    '''
                    2022-04-20 18:14:26
                    utilization.gpu [%], memory.used [MiB]
                    4 %, 1816 MiB
                    0 %, 1840 MiB
                    0 %, 1840 MiB
                    0 %, 1720 MiB
                    '''
            time.sleep(self.interval)
            self.step += 1


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

