import abc
import gc
import inspect
from typing import Tuple, Optional
from collections import defaultdict
import time
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim

KB = 1024.0
MB = 1024.0 * 1024.0

__all__ = [
    "BaseProfiler",
    "HookProfiler",
    "CategoryMemoryProfiler",
    "LayerWiseMemoryProfiler",
    "LayerWiseFPCUDATimeProfiler",
    "LayerWiseBPCUDATimeProfiler",
]


def _get_caller_info(depth=1):
    caller_frame = inspect.currentframe().f_back
    for _ in range(depth - 1):
        caller_frame = caller_frame.f_back
    caller_file = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    caller_func = caller_frame.f_code.co_name
    caller_str = f"{caller_file}:line{caller_lineno}, {caller_func}"
    return caller_str


def _cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception:
            pass


class BaseProfiler(abc.ABC):
    def __init__(self, models: Tuple[nn.Module]):
        r"""
        **API Language:**
        :ref:`中文 <BaseProfiler.__init__-cn>` | :ref:`English <BaseProfiler.__init__-en>`

        ----

        .. _BaseProfiler.__init__-cn:

        * **中文**

        分析器的基类。欲实现新的分析器，需实现 :meth:`export` 和 :meth:`close` 方法。

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        ----

        .. _BaseProfiler.__init__-en:

        * **English**

        The base class of profilers.
        To implement a new profiler, you need to implement the :meth:`export` and :meth:`close` methods.

        :param models: a tuple of target neural network modules
        :type models: Tuple[nn.Module]
        """
        if isinstance(models, nn.Module):
            models = (models,)
        elif not isinstance(models, tuple):
            models = tuple(models)
        self.models = models
        self._entered = False

    @abc.abstractmethod
    def export(self):
        r"""
        **API Language:**
        :ref:`中文 <BaseProfiler.export-cn>` | :ref:`English <BaseProfiler.export-en>`

        ----

        .. _BaseProfiler.export-cn:

        * **中文**

        导出分析结果。

        ----

        .. _BaseProfiler.export-en:

        * **English**

        Export profiling results.
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Profiler already entered")
        self._entered = True
        return self

    def __exit__(self, *args):
        self.close()
        self._entered = False
        return False

    def __del__(self):
        self.close()


class CategoryMemoryProfiler(BaseProfiler):
    def __init__(
        self,
        models: Tuple[nn.Module],
        optimizers: Tuple[optim.Optimizer],
        log_path="snn_memory.prof.txt",
    ):
        r"""
        **API Language:**
        :ref:`中文 <CategoryMemoryProfiler.__init__-cn>` | :ref:`English <CategoryMemoryProfiler.__init__-en>`

        ----

        .. _CategoryMemoryProfiler.__init__-cn:

        * **中文**

        类别内存分析器。

        调用 :meth:`export` 方法时，该分析器将输出当前时刻权重、缓冲区（buffer）、梯度、优化器状态和激活值等数据占据的显存大小。

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        :param optimizers: 优化器元组
        :type optimizers: Tuple[optim.Optimizer]

        :param log_path: 日志文本文件路径
        :type log_path: str

        ----

        .. _CategoryMemoryProfiler.__init__-en:

        * **English**

        Category memory profiler.

        When :meth:`export` is called, this profiler will output the memory usage of weights, buffers, gradients,
        optimizer states and activations.

        :param models: a tuple of target neural network modules
        :type models: Tuple[nn.Module]

        :param optimizers: a tuple of optimizers
        :type optimizers: Tuple[optim.Optimizer]

        :param log_path: path to the log text file
        :type log_path: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            with CategoryMemoryProfiler((net,), (optimizer,)) as prof:
                x = torch.randn(32, 10)
                y = net(x)
                loss = y.sum()
                results = prof.export()
                loss.backward()
                results = prof.export()
                optimizer.step()
        """
        super().__init__(models)

        if isinstance(optimizers, optim.Optimizer):
            optimizers = (optimizers,)
        elif not isinstance(optimizers, tuple):
            optimizers = tuple(optimizers)
        self.optimizers = optimizers

        self.log_path = Path(log_path)
        if self.log_path.exists():
            os.remove(self.log_path)

        self.device_count = torch.cuda.device_count()

    def _get_memory_stats(self):
        memory_usage = defaultdict(float)  # KB

        # model weights
        weight_tensors = set()
        for model in self.models:
            for param in model.parameters():
                if param.is_cuda:
                    nbytes = param.element_size() * param.numel()
                    memory_usage["weight"] += nbytes
                    weight_tensors.add(param.data_ptr())
            for buffer in model.buffers():
                if buffer.is_cuda:
                    nbytes = buffer.element_size() * buffer.numel()
                    memory_usage["buffer"] += nbytes
                    weight_tensors.add(buffer.data_ptr())

        # gradients
        gradient_tensors = set()
        for model in self.models:
            for param in model.parameters():
                if param.grad is not None and param.grad.is_cuda:
                    nbytes = param.grad.element_size() * param.grad.numel()
                    memory_usage["gradient"] += nbytes
                    gradient_tensors.add(param.grad.data_ptr())

        # optimizer state
        optimizer_state_tensors = set()
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        for key, value in state.items():
                            if torch.is_tensor(value) and value.is_cuda:
                                nbytes = value.element_size() * value.numel()
                                memory_usage["optimizer_state"] += nbytes
                                optimizer_state_tensors.add(value.data_ptr())

        classified_tensors = weight_tensors | gradient_tensors | optimizer_state_tensors
        for x in _cuda_tensors():
            if x.data_ptr() not in classified_tensors:
                nbytes = x.element_size() * x.numel()
                memory_usage["input_or_state"] += nbytes
                classified_tensors.add(x.data_ptr())

        return memory_usage

    def export(self, depth=2, output: bool = True, *args, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <CategoryMemoryProfiler.export-cn>` | :ref:`English <CategoryMemoryProfiler.export-en>`

        ----

        .. _CategoryMemoryProfiler.export-cn:

        * **中文**

        导出类别内存分析结果。

        :param depth: 调用栈深度，用于在输出文本中显示当前处于那个函数调用的内部
        :type depth: int

        :param output: 是否输出到控制台和文件
        :type output: bool

        :return: 总内存信息和分类内存统计信息
        :rtype: Tuple[dict, defaultdict]

        ----

        .. _CategoryMemoryProfiler.export-en:

        * **English**

        Export category memory profiling results.

        :param depth: call stack depth. Used to show which function is currently in
        :type depth: int

        :param output: whether to output to console and file
        :type output: bool

        :return: total memory info and category-wise memory statistics
        :rtype: Tuple[dict, defaultdict]
        """
        memory_usage = self._get_memory_stats()

        total_mem = {}
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                total_mem[device_id] = {
                    "allocated": torch.cuda.memory_allocated() / MB,
                    "reserved": torch.cuda.memory_reserved() / MB,
                }

        if output:
            caller_str = _get_caller_info(depth)
            header_str = f"=== Category-wise Memory ({time.ctime()}; {caller_str}) ==="

            out_str = "=" * len(header_str) + "\n"
            out_str += header_str + "\n"
            out_str += "=" * len(header_str) + "\n"
            for device_id in range(self.device_count):
                out_str += (
                    f"cuda:{device_id} - "
                    f"Allocated: {total_mem[device_id]['allocated']:.2f} MB, "
                    f"Reserved: {total_mem[device_id]['reserved']:.2f} MB\n"
                )
            out_str += "Memory Usage by Category:\n"
            for category, usage in memory_usage.items():
                out_str += f"  {category}: {usage / MB:.2f} MB\n"
            out_str += f"  Total: {sum(memory_usage.values()) / MB:.2f} MB\n"
            out_str += "=" * len(header_str) + "\n"
            out_str += "=" * len(header_str) + "\n" * 3

            print(out_str)
            with open(self.log_path, "a") as f:
                f.write(out_str)

        return total_mem, memory_usage

    def close(self):
        pass


class HookProfiler(BaseProfiler):
    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        log_path: str = "prof.txt",
    ):
        """
        **API Language:**
        :ref:`中文 <HookProfiler.__init__-cn>` | :ref:`English <HookProfiler.__init__-en>`

        ----

        .. _HookProfiler.__init__-cn:

        * **中文**

        需注册钩子函数的分析器基类。

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        :param model_names: 模型名称元组。应与 ``models`` 一一对应，用于显示结果
        :type model_names: Tuple[str]

        :param search_mode: 搜索模式元组。应与 ``models`` 一一对应，用于指定对那些模块添加钩子。
            若 ``search_mode[i] == "self"``，则对 ``models[i]`` 添加钩子。
            若 ``search_mode[i] == "submodules"``，则对 ``models[i]`` 的所有子模块添加钩子。
            若 ``search_mode[i] == "direct_children"``，则对 ``models[i]`` 的直接子模块添加钩子。
        :type search_mode: Tuple[str]

        :param instances: 目标模块类型元组。只有类型匹配的模块才会被添加钩子。默认为 ``nn.Module`` 。
        :type instances: Tuple[nn.Module]

        :param log_path: 日志文本文件路径
        :type log_path: str

        ----

        .. _HookProfiler.__init__-en:

        * **English**

        Base class for profilers that register hook functions.

        :param models: target neural network modules
        :type models: Tuple[nn.Module]

        :param model_names: model names. Should have the same length as ``models``.
        :type model_names: Tuple[str]

        :param search_mode: search mode. Should have the same length as ``models``. Used to
            specify which modules to add hooks to.
            If ``search_mode[i] == "self"``, add hooks to ``models[i]``.
            If ``search_mode[i] == "submodules"``, add hooks to all submodules of ``models[i]``.
            If ``search_mode[i] == "direct_children"``, then add hooks to all direct children of ``models[i]``.
        :type search_mode: Tuple[str]

        :param instances: target module types. Only modules of the specified type will be added hooks.
            Default is ``nn.Module``.
        :type instances: Tuple[nn.Module]

        :param log_path: path to the log text file
        :type log_path: str
        """
        super().__init__(models)

        if model_names is None:
            model_names = tuple([f"net{i}" for i in range(len(models))])
        if not isinstance(model_names, (tuple, list)):
            raise ValueError("model_names should be a tuple of strings")
        if len(model_names) != len(models):
            raise ValueError("model_names should have the same length as models")
        self.model_names = model_names

        if isinstance(search_mode, str):
            search_mode = (search_mode,)
        elif not isinstance(search_mode, tuple):
            search_mode = tuple(search_mode)
        if len(search_mode) != len(self.models):
            raise ValueError("search_mode should have the same length as models")
        self.search_mode = search_mode

        if isinstance(instances, nn.Module):
            instances = (instances,)
        elif not isinstance(instances, tuple):
            instances = tuple(instances)
        self.instances = instances

        self.log_path = Path(log_path)
        if self.log_path.exists():
            os.remove(self.log_path)

        self.hooks = []
        self.module_obj = {}  # mapping from module name to the module itself

    def __enter__(self):
        super().__enter__()
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self.close()
        return super().__exit__(*args)

    def _remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def close(self):
        self._remove_hooks()

    def _get_module_iterator(self, mode: str, model: nn.Module):
        """Get iterator for modules based on search mode."""
        if mode == "self":
            return (("self", model),)
        elif mode == "submodules":
            return model.named_modules()
        elif mode == "direct_children":
            return model.named_children()
        else:
            raise ValueError(f"Unknown search mode: {mode}")


class LayerWiseMemoryProfiler(HookProfiler):
    field_idx = {
        "name": 0,
        "forward_start_memory": 1,
        "forward_end_memory": 2,
        "forward_peak_memory": 3,
        "forward_delta_memory": 4,
        "backward_start_memory": 5,
        "backward_end_memory": 6,
        "backward_peak_memory": 7,
        "backward_delta_memory": 8,
    }

    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        log_path="layer_memory.prof.txt",
        data_path="layer_memory.prof.pt",
        device: str = "cuda",
    ):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseMemoryProfiler.__init__-cn>` | :ref:`English <LayerWiseMemoryProfiler.__init__-en>`

        ----

        .. _LayerWiseMemoryProfiler.__init__-cn:

        * **中文**

        逐层显存分析器。

        对于每个目标模块，该分析器将记录以下显存使用情况：

        - 前向传播开始前
        - 前向传播结束后
        - 前向传播期间的峰值
        - 前向传播显存变化量：峰值显存减去开始前显存
        - 反向传播开始前
        - 反向传播结束后
        - 反向传播期间的峰值
        - 反向传播显存变化量：峰值显存减去开始前显存

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        :param model_names: 模型名称元组。应与 ``models`` 一一对应，用于显示结果
        :type model_names: Tuple[str]

        :param search_mode: 搜索模式元组。应与 ``models`` 一一对应，用于指定对那些模块添加钩子。
            若 ``search_mode[i] == "self"``，则对 ``models[i]`` 添加钩子。
            若 ``search_mode[i] == "submodules"``，则对 ``models[i]`` 的所有子模块添加钩子。
            若 ``search_mode[i] == "direct_children"``，则对 ``models[i]`` 的直接子模块添加钩子。
        :type search_mode: Tuple[str]

        :param instances: 目标模块类型元组。只有类型匹配的模块才会被添加钩子。默认为 ``nn.Module`` 。
        :type instances: Tuple[nn.Module]

        :param log_path: 日志文本文件路径
        :type log_path: str

        :param data_path: 输出数据文件路径
        :type data_path: str

        :param device: 设备名称
        :type device: str

        ----

        .. _LayerWiseMemoryProfiler.__init__-en:

        * **English**

        Layer-wise memory profiler.

        For each target module, this profiler records the following memory usage:

        - before forward propagation starts
        - after forward propagation ends
        - the peak during forward propagation
        - the change in memory usage during forward propagation: peak memory usage minus start memory
        - before backward propagation starts
        - after backward propagation ends
        - the peak during backward propagation
        - the change in memory usage during backward propagation: peak memory usage minus start memory

        :param models: target neural network modules
        :type models: Tuple[nn.Module]

        :param model_names: model names. Should have the same length as ``models``.
        :type model_names: Tuple[str]

        :param search_mode: search mode. Should have the same length as ``models``. Used to
            specify which modules to add hooks to.
            If ``search_mode[i] == "self"``, add hooks to ``models[i]``.
            If ``search_mode[i] == "submodules"``, add hooks to all submodules of ``models[i]``.
            If ``search_mode[i] == "direct_children"``, then add hooks to all direct children of ``models[i]``.
        :type search_mode: Tuple[str]

        :param instances: target module types. Only modules of the specified type will be added hooks.
            Default is ``nn.Module``.
        :type instances: Tuple[nn.Module]

        :param log_path: path to the log text file
        :type log_path: str

        :param data_path: path to the output data file
        :type data_path: str

        :param device: device name
        :type device: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            with LayerWiseMemoryProfiler((net,)) as prof:
                x = torch.randn(32, 10)
                y = net(x)
                loss = y.sum()
                loss.backward()

            results = prof.export()
        """
        super().__init__(models, model_names, search_mode, instances, log_path)
        self.device = device

        self.data_path = Path(data_path)
        if self.data_path.exists():
            os.remove(self.data_path)

        self.forward_start_memory = defaultdict(float)
        self.forward_end_memory = defaultdict(float)
        self.forward_peak_memory = defaultdict(float)
        self.backward_start_memory = defaultdict(float)
        self.backward_end_memory = defaultdict(float)
        self.backward_peak_memory = defaultdict(float)

    def _register_hooks(self):
        def pre_hook_generator(name):
            def pre_hook(module, input):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                self.forward_start_memory[name] = torch.cuda.memory_allocated(
                    self.device
                )
                self.forward_peak_memory[name] = 0

            return pre_hook

        def post_hook_generator(name):
            def post_hook(module, input, output):
                torch.cuda.synchronize(self.device)
                self.forward_peak_memory[name] = max(
                    torch.cuda.max_memory_allocated(self.device),
                    self.forward_peak_memory[name],
                )
                self.forward_end_memory[name] = torch.cuda.memory_allocated(self.device)

            return post_hook

        def backward_pre_hook_generator(name):
            def backward_pre_hook(module, grad_output):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                self.backward_start_memory[name] = torch.cuda.memory_allocated(
                    self.device
                )
                self.backward_peak_memory[name] = 0

            return backward_pre_hook

        def backward_post_hook_generator(name):
            def backward_post_hook(module, grad_input, grad_output):
                torch.cuda.synchronize(self.device)
                self.backward_peak_memory[name] = max(
                    torch.cuda.max_memory_allocated(self.device),
                    self.backward_peak_memory[name],
                )
                self.backward_end_memory[name] = torch.cuda.memory_allocated(
                    self.device
                )

            return backward_post_hook

        for i, model in enumerate(self.models):
            for name, m in self._get_module_iterator(self.search_mode[i], model):
                if isinstance(m, self.instances):
                    mname = f"{self.model_names[i]}'s {name}"
                    self.module_obj[mname] = m
                    h1 = m.register_forward_pre_hook(pre_hook_generator(mname))
                    h2 = m.register_forward_hook(post_hook_generator(mname))
                    h3 = m.register_full_backward_pre_hook(
                        backward_pre_hook_generator(mname)
                    )
                    h4 = m.register_full_backward_hook(
                        backward_post_hook_generator(mname)
                    )
                    self.hooks += [h1, h2, h3, h4]

    def export(
        self,
        depth=2,
        sort_by: Optional[str] = None,
        output: bool = True,
        *args,
        **kwargs,
    ):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseMemoryProfiler.export-cn>` | :ref:`English <LayerWiseMemoryProfiler.export-en>`

        ----

        .. _LayerWiseMemoryProfiler.export-cn:

        * **中文**

        导出分层内存分析结果。

        :param depth: 调用栈深度，用于在输出中显示当前处于哪个函数内部
        :type depth: int

        :param sort_by: 排序字段。将按照所选字段从大到小顺序排列所有目标模块。可选字段为 ``"name"``,
            ``"forward_start_memory"``, ``"forward_end_memory"``, ``"forward_peak_memory"``,
            ``"forward_delta_memory"``, ``"backward_start_memory"``, ``"backward_end_memory"``,
            ``"backward_peak_memory"``, ``"backward_delta_memory"`` 。默认为 ``None``，即
            不额外排序，而是按照前向传播执行的拓扑顺序。
        :type sort_by: Optional[str]

        :param output: 是否输出到控制台和文件
        :type output: bool

        :return: 分层内存统计结果
        :rtype: list

        ----

        .. _LayerWiseMemoryProfiler.export-en:

        * **English**

        Export layer-wise memory profiling results.

        :param depth: call stack depth. Used to show which function is currently in
        :type depth: int

        :param sort_by: sorting field. The results will be sorted in descending order
            according to the selected field. Available fields are ``"name"``,
            ``"forward_start_memory"``, ``"forward_end_memory"``, ``"forward_peak_memory"``,
            ``"forward_delta_memory"``, ``"backward_start_memory"``, ``"backward_end_memory"``,
            ``"backward_peak_memory"``, ``"backward_delta_memory"`` . Default is ``None``,
            which means sorted according to the topological order of forward propagation.
        :type sort_by: str

        :param output: whether to output to console and file
        :type output: bool

        :return: layer-wise memory statistics
        :rtype: list
        """
        results = []
        for name in self.forward_peak_memory.keys():
            f_start = self.forward_start_memory[name]
            f_end = self.forward_end_memory[name]
            f_peak = self.forward_peak_memory[name]
            f_delta = f_peak - f_start
            b_start = self.backward_start_memory[name]
            b_end = self.backward_end_memory[name]
            b_peak = self.backward_peak_memory[name]
            b_delta = b_peak - b_start

            results.append(
                (name, f_start, f_end, f_peak, f_delta, b_start, b_end, b_peak, b_delta)
            )

        if sort_by is not None:
            results = sorted(
                results, key=lambda x: x[self.field_idx[sort_by]], reverse=True
            )
        else:
            results = sorted(results, key=lambda x: max(x[1:]), reverse=True)

        if output:
            caller_str = _get_caller_info(depth)
            header_str = (
                f"=== Layer-wise Memory Stats ({time.ctime()}; {caller_str}) ==="
            )

            out_str = "=" * len(header_str) + "\n"
            out_str += header_str + "\n"
            out_str += "=" * len(header_str) + "\n"
            for (
                name,
                f_start,
                f_end,
                f_peak,
                f_delta,
                b_start,
                b_end,
                b_peak,
                b_delta,
            ) in results:
                out_str += f"{name}:{str(self.module_obj[name])}\n"
                out_str += f" forward_start_memory: {f_start / MB:.2f} MB, "
                out_str += f"forward_end_memory: {f_end / MB:.2f} MB, "
                out_str += f"forward_peak_memory: {f_peak / MB:.2f} MB, "
                out_str += f"forward_delta_memory: {f_delta / MB:.2f} MB, "
                out_str += f"\n backward_start_memory: {b_start / MB:.2f} MB, "
                out_str += f"backward_end_memory: {b_end / MB:.2f} MB, "
                out_str += f"backward_peak_memory: {b_peak / MB:.2f} MB, "
                out_str += f"backward_delta_memory: {b_delta / MB:.2f} MB\n"
            out_str += "=" * len(header_str) + "\n"
            out_str += "=" * len(header_str) + "\n" * 3

            print(out_str)
            with open(self.log_path, "a") as f:
                f.write(out_str)

            torch.save(
                {
                    "forward_start_memory": self.forward_start_memory,
                    "forward_end_memory": self.forward_end_memory,
                    "forward_peak_memory": self.forward_peak_memory,
                    "backward_start_memory": self.backward_start_memory,
                    "backward_end_memory": self.backward_end_memory,
                    "backward_peak_memory": self.backward_peak_memory,
                },
                self.data_path,
            )

        return results


class LayerWiseFPCUDATimeProfiler(HookProfiler):
    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        warmup: int = 10,
        log_path="layer_time_fp.prof.txt",
    ):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseFPCUDATimeProfiler.__init__-cn>` | :ref:`English <LayerWiseFPCUDATimeProfiler.__init__-en>`

        ----

        .. _LayerWiseFPCUDATimeProfiler.__init__-cn:

        * **中文**

        逐层前向传播CUDA时间分析器。

        对于每个目标模块，该分析器将测量前向传播的时间消耗。测量过程如下：

        1. 在模块前向传播开始前记录CUDA事件
        2. 在模块前向传播结束后记录CUDA事件
        3. 计算两个事件之间的时间差作为该模块的前向传播时间
        4. 重复多次测量以获得平均时间

        前 ``warmup`` 次测量将被忽略，以消除冷启动效应。

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        :param model_names: 模型名称元组。应与 ``models`` 一一对应，用于显示结果
        :type model_names: Tuple[str]

        :param search_mode: 搜索模式元组。应与 ``models`` 一一对应，用于指定对那些模块添加钩子。
            若 ``search_mode[i] == "self"``，则对 ``models[i]`` 添加钩子。
            若 ``search_mode[i] == "submodules"``，则对 ``models[i]`` 的所有子模块添加钩子。
            若 ``search_mode[i] == "direct_children"``，则对 ``models[i]`` 的直接子模块添加钩子。
        :type search_mode: Tuple[str]

        :param instances: 目标模块类型元组。只有类型匹配的模块才会被添加钩子。默认为 ``nn.Module`` 。
        :type instances: Tuple[nn.Module]

        :param warmup: 预热迭代次数。前 ``warmup`` 次测量结果将被忽略，以消除冷启动效应。
        :type warmup: int

        :param log_path: 日志文本文件路径
        :type log_path: str

        ----

        .. _LayerWiseFPCUDATimeProfiler.__init__-en:

        * **English**

        Layer-wise forward propagation CUDA time profiler.

        For each target module, this profiler measures the time consumption of forward propagation.
        The measurement process is as follows:

        1. Record a CUDA event before the module's forward propagation starts
        2. Record a CUDA event after the module's forward propagation ends
        3. Calculate the time difference between the two events as the forward propagation time of this module
        4. Repeat measurements multiple times to obtain average time

        The first ``warmup`` measurements will be ignored to eliminate cold start effects.

        :param models: target neural network modules
        :type models: Tuple[nn.Module]

        :param model_names: model names. Should have the same length as ``models``.
        :type model_names: Tuple[str]

        :param search_mode: search mode. Should have the same length as ``models``. Used to
            specify which modules to add hooks to.
            If ``search_mode[i] == "self"``, add hooks to ``models[i]``.
            If ``search_mode[i] == "submodules"``, add hooks to all submodules of ``models[i]``.
            If ``search_mode[i] == "direct_children"``, then add hooks to all direct children of ``models[i]``.
        :type search_mode: Tuple[str]

        :param instances: target module types. Only modules of the specified type will
            be added hooks. Default is ``nn.Module``.
        :type instances: Tuple[nn.Module]

        :param warmup: number of warmup iterations. The first ``warmup`` measurement
            results will be ignored to eliminate cold start effects.
        :type warmup: int

        :param log_path: path to the log text file
        :type log_path: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            with LayerWiseFPCUDATimeProfiler((net,)) as prof:
                net.eval()
                with torch.no_grad():
                    for _ in range(10):
                        x = torch.randn(32, 10)
                        _ = net(x)
            results = prof.export()
        """
        super().__init__(models, model_names, search_mode, instances, log_path)
        self.warmup = warmup
        self.result = defaultdict(list)
        self.start_events = {}

    def _register_hooks(self):
        def pre_hook_generator(name):
            def pre_hook(module, input):
                start_event = torch.cuda.Event(enable_timing=True)
                # make sure that previous modules have been FPed
                torch.cuda.synchronize()
                start_event.record()
                self.start_events[name] = start_event

            return pre_hook

        def post_hook_generator(name):
            def post_hook(module, input, output):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # make sure that the current module have been FPed
                torch.cuda.synchronize()
                start_event = self.start_events.get(name, None)
                if start_event is not None:
                    elapsed_time = start_event.elapsed_time(end_event)
                    self.result[name].append(elapsed_time)
                    self.start_events[name] = None

            return post_hook

        for i, model in enumerate(self.models):
            for name, m in self._get_module_iterator(self.search_mode[i], model):
                if isinstance(m, self.instances):
                    mname = f"{self.model_names[i]}'s {name}"
                    self.module_obj[mname] = m
                    pre_handle = m.register_forward_pre_hook(pre_hook_generator(mname))
                    post_handle = m.register_forward_hook(post_hook_generator(mname))
                    self.start_events[mname] = None
                    self.hooks += [pre_handle, post_handle]

    def export(self, output: bool = True, *args, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseFPCUDATimeProfiler.export-cn>` | :ref:`English <LayerWiseFPCUDATimeProfiler.export-en>`

        ----

        .. _LayerWiseFPCUDATimeProfiler.export-cn:

        * **中文**

        导出分层前向传播时间分析结果。

        :param output: 是否输出到控制台和文件
        :type output: bool

        :return: 时间统计结果
        :rtype: list

        ----

        .. _LayerWiseFPCUDATimeProfiler.export-en:

        * **English**

        Export layer-wise forward propagation time profiling results.

        :param output: whether to output to console and file
        :type output: bool

        :return: time statistics
        :rtype: list
        """
        table = []
        for name in self.result.keys():
            forward_time = self.result[name][self.warmup :]
            avg_t = sum(forward_time) / len(forward_time)
            table.append((name, avg_t))

        table = sorted(table, key=lambda x: x[1], reverse=True)

        if output:
            out_str = ""
            for name, avg_t in table:
                out_str += f"{name}:{str(self.module_obj[name])} forward => "
                out_str += f"{avg_t:.2f} ms\n\n"

            print(out_str)
            with open(self.log_path, "a") as f:
                f.write(out_str)

        return table


class LayerWiseBPCUDATimeProfiler(HookProfiler):
    def __init__(
        self,
        models: Tuple[nn.Module],
        model_names: Tuple[str] = None,
        search_mode: Tuple[str] = ("direct_children",),
        instances: Tuple[nn.Module] = (nn.Module,),
        warmup: int = 10,
        log_path="layer_time_bp.prof.txt",
    ):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseBPCUDATimeProfiler.__init__-cn>` | :ref:`English <LayerWiseBPCUDATimeProfiler.__init__-en>`

        ----

        .. _LayerWiseBPCUDATimeProfiler.__init__-cn:

        * **中文**

        逐层反向传播CUDA时间分析器。

        对于每个目标模块，该分析器将测量反向传播的时间消耗。测量过程如下：

        1. 在模块反向传播开始前记录CUDA事件
        2. 在模块反向传播结束后记录CUDA事件
        3. 计算两个事件之间的时间差作为该模块的反向传播时间
        4. 重复多次测量以获得平均时间

        前 ``warmup`` 次测量将被忽略，以消除冷启动效应。

        :param models: 目标神经网络模块元组
        :type models: Tuple[nn.Module]

        :param model_names: 模型名称元组。应与 ``models`` 一一对应，用于显示结果
        :type model_names: Tuple[str]

        :param search_mode: 搜索模式元组。应与 ``models`` 一一对应，用于指定对那些模块添加钩子。
            若 ``search_mode[i] == "self"``，则对 ``models[i]`` 添加钩子。
            若 ``search_mode[i] == "submodules"``，则对 ``models[i]`` 的所有子模块添加钩子。
            若 ``search_mode[i] == "direct_children"``，则对 ``models[i]`` 的直接子模块添加钩子。
        :type search_mode: Tuple[str]

        :param instances: 目标模块类型元组。只有类型匹配的模块才会被添加钩子。默认为 ``nn.Module`` 。
        :type instances: Tuple[nn.Module]

        :param warmup: 预热迭代次数。前 ``warmup`` 次测量结果将被忽略，以消除冷启动效应。
        :type warmup: int

        :param log_path: 日志文本文件路径
        :type log_path: str

        ----

        .. _LayerWiseBPCUDATimeProfiler.__init__-en:

        * **English**

        Layer-wise backward propagation CUDA time profiler.

        For each target module, this profiler measures the time consumption of backward propagation.
        The measurement process is as follows:

        1. Record a CUDA event before the module's backward propagation starts
        2. Record a CUDA event after the module's backward propagation ends
        3. Calculate the time difference between the two events as the backward propagation time of this module
        4. Repeat measurements multiple times to obtain average time

        The first ``warmup`` measurements will be ignored to eliminate cold start effects.

        :param models: target neural network modules
        :type models: Tuple[nn.Module]

        :param model_names: model names. Should have the same length as ``models``.
        :type model_names: Tuple[str]

        :param search_mode: search mode. Should have the same length as ``models``. Used to
            specify which modules to add hooks to.
            If ``search_mode[i] == "self"``, add hooks to ``models[i]``.
            If ``search_mode[i] == "submodules"``, add hooks to all submodules of ``models[i]``.
            If ``search_mode[i] == "direct_children"``, then add hooks to all direct children of ``models[i]``.
        :type search_mode: Tuple[str]

        :param instances: target module types. Only modules of the specified type will
            be added hooks. Default is ``nn.Module``.
        :type instances: Tuple[nn.Module]

        :param warmup: number of warmup iterations. The first ``warmup`` measurement
            results will be ignored to eliminate cold start effects.
        :type warmup: int

        :param log_path: path to the log text file
        :type log_path: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            with LayerWiseBPCUDATimeProfiler((net,)) as prof:
                for _ in range(10):
                    x = torch.randn(32, 10, requires_grad=True)
                    y = net(x)
                    loss = y.sum()
                    loss.backward()
                    functional.reset_net(net)
            results = prof.export()
        """
        super().__init__(models, model_names, search_mode, instances, log_path)
        self.warmup = warmup
        self.result = defaultdict(list)
        self.start_events = {}

    def _register_hooks(self):
        def bp_pre_hook_generator(name):
            def pre_hook(module, grad_input):
                start_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                self.start_events[name] = start_event

            return pre_hook

        def bp_post_hook_generator(name):
            def post_hook(module, grad_input, grad_output):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                start_event = self.start_events.get(name, None)
                if start_event is not None:
                    elapsed_time = start_event.elapsed_time(end_event)
                    self.result[name].append(elapsed_time)
                    self.start_events[name] = None

            return post_hook

        for i, model in enumerate(self.models):
            for name, m in self._get_module_iterator(self.search_mode[i], model):
                if isinstance(m, self.instances):
                    mname = f"{self.model_names[i]}'s {name}"
                    self.module_obj[mname] = m
                    h_pre = m.register_full_backward_pre_hook(
                        bp_pre_hook_generator(mname)
                    )
                    h_post = m.register_full_backward_hook(
                        bp_post_hook_generator(mname)
                    )
                    self.start_events[mname] = None
                    self.hooks += [h_pre, h_post]

    def export(self, output: bool = True, *args, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <LayerWiseBPCUDATimeProfiler.export-cn>` | :ref:`English <LayerWiseBPCUDATimeProfiler.export-en>`

        ----

        .. _LayerWiseBPCUDATimeProfiler.export-cn:

        * **中文**

        导出分层反向传播时间分析结果。

        :param output: 是否输出到控制台和文件
        :type output: bool

        :return: 时间统计结果
        :rtype: list

        ----

        .. _LayerWiseBPCUDATimeProfiler.export-en:

        * **English**

        Export layer-wise backward propagation time profiling results.

        :param output: whether to output to console and file
        :type output: bool

        :return: time statistics
        :rtype: list
        """
        table = []
        for name in self.result.keys():
            bp_times = self.result[name][self.warmup :]
            avg_t = sum(bp_times) / len(bp_times)
            table.append((name, avg_t))

        table = sorted(table, key=lambda x: x[1], reverse=True)

        if output:
            out_str = ""
            for name, avg_t in table:
                out_str += f"{name}:{self.module_obj[name]} backward => "
                out_str += f"{avg_t:.2f} ms\n\n"

            print(out_str)
            with open(self.log_path, "a") as f:
                f.write(out_str)

        return table
