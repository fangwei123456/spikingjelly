import torch
import numpy as np
from torch import nn
from typing import Callable, Any
from spikingjelly.activation_based import neuron
import threading
from torch.utils.tensorboard import SummaryWriter
import os
import time
import re
import datetime

def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.forward_hooks = []
        self.backward_hooks = []
        self.forward_pre_hooks = []
        self.records = []
        self._enable = True

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        for hook in self.forward_pre_hooks:
            hook.remove()

        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.forward_pre_hooks.clear()

    def register_forward_hook(self, module: nn.Module, instance: Any or tuple = (nn.Module,), hook: Callable = None):
        for m in module.modules():
            if isinstance(m, instance):
                self.forward_hooks.append(m.register_forward_hook(hook))

    def register_backward_hook(self, module: nn.Module, instance: Any or tuple = (nn.Module,), hook: Callable = None):
        for m in module.modules():
            if isinstance(m, instance):
                if torch.__version__ >= torch.torch_version.TorchVersion('1.8.0'):
                    self.backward_hooks.append(m.register_full_backward_hook(hook))
                else:
                    self.backward_hooks.append(m.register_backward_hook(hook))

    def register_forward_pre_hook(self, module: nn.Module, instance: Any or tuple = (nn.Module,), hook: Callable = None):
        for m in module.modules():
            if isinstance(m, instance):
                self.forward_pre_hooks.append(m.register_forward_pre_hook(hook))

    def __del__(self):
        self.remove_hooks()


class OutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = (nn.Module,), function_on_output: Callable = lambda x: x):
        super().__init__()
        self.function_on_output = function_on_output
        self.register_forward_hook(net, instance, self.record_output_hook)

    def record_output_hook(self, module: nn.Module, x, y):
        if self.is_enable():
            with torch.no_grad():
                self.records.append(self.function_on_output(unpack_len1_tuple(y)))


class InputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = (nn.Module,), function_on_input: Callable = lambda x: x):
        super().__init__()
        self.function_on_input = function_on_input
        self.register_forward_hook(net, instance, self.record_input_hook)

    def record_input_hook(self, module: nn.Module, x, y):
        if self.is_enable():
            with torch.no_grad():
                self.records.append(self.function_on_input(unpack_len1_tuple(x)))


class AttributeMonitor(BaseMonitor):
    def __init__(self, attribute_name: str, pre_forward: bool, net: nn.Module, instance: Any or tuple = (nn.Module,),
                 function_on_attribute: Callable = lambda x: x):
        super().__init__()
        self.attribute_name = attribute_name
        self.function_on_attribute = function_on_attribute
        if pre_forward:
            self.register_forward_pre_hook(net, instance, self.record_attribute_hook)
        else:
            self.register_forward_hook(net, instance, self.record_attribute_hook)

    def record_attribute_hook(self, module: nn.Module, x, y):
        if self.is_enable():
            with torch.no_grad():
                self.records.append(self.function_on_attribute(module.__getattr__(self.attribute_name)))

class GradInputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = (nn.Module,), function_on_grad_input: Callable = lambda x: x):
        super().__init__()
        self.function_on_grad_input = function_on_grad_input
        self.register_backward_hook(net, instance, self.record_grad_input_hook)

    def record_grad_input_hook(self, module: nn.Module, grad_input, grad_output):
        if self.is_enable():
            with torch.no_grad():
                self.records.append(self.function_on_grad_input(unpack_len1_tuple(grad_input)))


class GradOutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = (nn.Module,), function_on_grad_output: Callable = lambda x: x):
        super().__init__()
        self.function_on_grad_output = function_on_grad_output
        self.register_backward_hook(net, instance, self.record_grad_output_hook)

    def record_grad_output_hook(self, module: nn.Module, grad_input, grad_output):
        if self.is_enable():
            with torch.no_grad():
                self.records.append(self.function_on_grad_output(unpack_len1_tuple(grad_output)))

class GPUMonitor(threading.Thread):
    def __init__(self, log_dir: str = None, gpu_ids: tuple = (0,), interval: float = 600., start_now=True):
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
            self.writer = SummaryWriter(os.path.join(log_dir, 'gpu_monitor'))

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
                        self.writer.add_scalar(f'utilization_{self.gpu_ids[i]}', utilization, self.step)
                        self.writer.add_scalar(f'memory_used_{self.gpu_ids[i]}', memory_used, self.step)
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
