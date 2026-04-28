import numpy as np
import logging

try:
    import cupy
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.cuda_kernel.auto_cuda.base: {e}")
    cupy = None

import torch
import sys
import logging

from .. import cuda_utils
from .... import configure


def wrap_with_comment(code: str, comment: str):
    if logging.DEBUG >= logging.root.level:
        return (
            "\n//------"
            + comment
            + " start------\n"
            + code
            + "\n//------"
            + comment
            + " end--------\n\n"
        )
    else:
        return code


def startswiths(x: str, prefixes: tuple):
    ret = False
    for prefix in prefixes:
        if x.startswith(prefix):
            ret = True

    return ret


class CKernel:
    def __init__(self, kernel_name: str):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-init-cn>` | :ref:`English <ckernel-init-en>`

        ----

        .. _ckernel-init-cn:

        * **中文**

        自定义 CUDA kernel 的基础封装类。它维护 kernel 形参表 ``cparams``、保留变量名
        ``reserved_cnames``，以及可拼接的代码片段（declaration/head/core/tail）。

        :param kernel_name: CUDA kernel 名称
        :type kernel_name: str

        ----

        .. _ckernel-init-en:

        * **English**

        Base wrapper for custom CUDA kernels. It stores kernel parameter metadata
        (``cparams``), reserved C variable names (``reserved_cnames``), and code
        segments (declaration/head/core/tail).

        :param kernel_name: CUDA kernel name
        :type kernel_name: str
        """
        self.cparams = {}
        self.reserved_cnames = []
        self.kernel_name = kernel_name
        self._core = ""

    def check_attributes(self, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-check-attributes-cn>` |
        :ref:`English <ckernel-check-attributes-en>`

        ----

        .. _ckernel-check-attributes-cn:

        * **中文**

        检查 ``kwargs`` 中给定属性值是否与当前对象属性一致。

        :param kwargs: 待检查的属性键值对
        :type kwargs: dict
        :return: 全部属性一致时返回 ``True``，否则返回 ``False``
        :rtype: bool

        ----

        .. _ckernel-check-attributes-en:

        * **English**

        Check whether provided attribute values in ``kwargs`` match current
        attributes on this object.

        :param kwargs: Attribute key-value pairs to check
        :type kwargs: dict
        :return: ``True`` if all attributes match; otherwise ``False``
        :rtype: bool
        """
        for key, value in kwargs.items():
            if value != self.__getattribute__(key):
                return False

        else:
            return True

    @property
    def core(self):
        return self._core

    @core.setter
    def core(self, value):
        self._core = value

    def set_contiguous(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-set-contiguous-cn>` |
        :ref:`English <ckernel-set-contiguous-en>`

        ----

        .. _ckernel-set-contiguous-cn:

        * **中文**

        将 ``py_dict`` 中的 ``torch.Tensor``/``cupy.ndarray`` 转为连续内存；若出现
        其他类型则抛出异常。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel-set-contiguous-en:

        * **English**

        Make ``torch.Tensor``/``cupy.ndarray`` values in ``py_dict`` contiguous.
        Raise an error for unsupported value types.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        # get contiguous
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.contiguous()

            elif isinstance(value, cupy.ndarray):
                value = cupy.ascontiguousarray(value)
            else:
                raise TypeError(type(value))

            py_dict[key] = value

    def get_device(self, py_dict: dict) -> int:
        r"""
        **API Language:**
        :ref:`中文 <ckernel-get-device-cn>` | :ref:`English <ckernel-get-device-en>`

        ----

        .. _ckernel-get-device-cn:

        * **中文**

        遍历 ``py_dict``，返回首个张量对象所在 CUDA 设备编号。

        :param py_dict: kernel 参数字典
        :type py_dict: dict
        :return: CUDA 设备编号
        :rtype: int
        :raise ValueError: 当 ``py_dict`` 中没有张量参数时

        ----

        .. _ckernel-get-device-en:

        * **English**

        Traverse ``py_dict`` and return the CUDA device id of the first tensor-like
        value.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        :return: CUDA device id
        :rtype: int
        :raise ValueError: If no tensor-like value is found
        """

        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                return item.get_device()

            elif isinstance(item, cupy.ndarray):
                return item.device.id

        raise ValueError

    def check_device(self, device: int, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-check-device-cn>` |
        :ref:`English <ckernel-check-device-en>`

        ----

        .. _ckernel-check-device-cn:

        * **中文**

        检查 ``py_dict`` 中所有张量是否都位于 ``device`` 指定的 CUDA 设备上。

        :param device: 目标 CUDA 设备编号
        :type device: int
        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel-check-device-en:

        * **English**

        Validate that all tensor-like values in ``py_dict`` are on the target CUDA
        device ``device``.

        :param device: Target CUDA device id
        :type device: int
        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                assert item.get_device() == device

            elif isinstance(item, cupy.ndarray):
                assert item.device.id == device

    def check_keys(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-check-keys-cn>` | :ref:`English <ckernel-check-keys-en>`

        ----

        .. _ckernel-check-keys-cn:

        * **中文**

        检查 ``py_dict`` 的键集合是否与 ``self.cparams`` 一致，不一致时抛出异常。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel-check-keys-en:

        * **English**

        Check whether keys in ``py_dict`` exactly match keys in ``self.cparams``.
        Raise an error on mismatch.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        if py_dict.keys() != self.cparams.keys():
            missed_keys = (py_dict.keys() | self.cparams.keys()) - (
                py_dict.keys() & self.cparams.keys()
            )

            if missed_keys.__len__() > 0:
                if (missed_keys & py_dict.keys()).__len__() > 0:
                    msg = f"{missed_keys} is in py_dict but not in cparams!"
                else:
                    msg = f"{missed_keys} is in cparams but not in py_dict!"
                raise ValueError(msg)

    def check_ctypes(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-check-ctypes-cn>` |
        :ref:`English <ckernel-check-ctypes-en>`

        ----

        .. _ckernel-check-ctypes-cn:

        * **中文**

        检查 ``py_dict`` 中各参数的数据类型是否与 ``self.cparams`` 中声明的 CUDA C
        参数类型匹配（例如 ``float`` / ``half2`` / ``int``）。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel-check-ctypes-en:

        * **English**

        Validate that runtime value dtypes in ``py_dict`` match declared CUDA C
        parameter types in ``self.cparams`` (e.g., ``float``/``half2``/``int``).

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        for key, value in py_dict.items():
            ctype: str = self.cparams[key]
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float:
                    assert startswiths(ctype, ("const float", "float"))

                elif value.dtype == torch.half:
                    assert startswiths(ctype, ("const half2", "half2"))

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float32:
                    assert startswiths(ctype, ("const float", "float"))

                elif value.dtype == np.float16:
                    assert startswiths(ctype, ("const half2", "half2"))

                elif value.dtype == int or (
                    hasattr(np, "int") and value.dtype == np.int
                ):
                    assert startswiths(ctype, ("const int", "int"))

    def check_half2(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-check-half2-cn>` |
        :ref:`English <ckernel-check-half2-en>`

        ----

        .. _ckernel-check-half2-cn:

        * **中文**

        供子类按需实现的 ``half2`` 参数校验接口。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel-check-half2-en:

        * **English**

        Extension hook for subclasses to implement ``half2``-related checks.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        raise NotImplementedError

    def get_ptrs(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-get-ptrs-cn>` | :ref:`English <ckernel-get-ptrs-en>`

        ----

        .. _ckernel-get-ptrs-cn:

        * **中文**

        提取 ``py_dict`` 中每个张量参数的底层指针（或等价对象），按键排序后的参数顺序返回。

        :param py_dict: kernel 参数字典
        :type py_dict: dict
        :return: 指针元组
        :rtype: tuple

        ----

        .. _ckernel-get-ptrs-en:

        * **English**

        Collect underlying pointers (or equivalent objects) for tensor-like
        values in ``py_dict`` and return them as a tuple.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        :return: Tuple of argument pointers
        :rtype: tuple
        """
        ret_list = []
        for item in py_dict.values():
            if isinstance(item, torch.Tensor):
                ret_list.append(item.data_ptr())

            elif isinstance(item, cupy.ndarray):
                ret_list.append(item)

            else:
                raise TypeError
        return tuple(ret_list)

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-call-cn>` | :ref:`English <ckernel-call-en>`

        ----

        .. _ckernel-call-cn:

        * **中文**

        执行 CUDA kernel。调用前会完成设备一致性检查、连续化、参数类型检查和键集合校验。

        :param grid: CUDA grid 配置
        :type grid: tuple
        :param block: CUDA block 配置
        :type block: tuple
        :param py_dict: kernel 实参字典，键需与 ``self.cparams`` 一一对应
        :type py_dict: dict

        ----

        .. _ckernel-call-en:

        * **English**

        Execute the CUDA kernel after validating device consistency, contiguous
        layout, ctypes compatibility, and key alignment with ``self.cparams``.

        :param grid: CUDA grid configuration
        :type grid: tuple
        :param block: CUDA block configuration
        :type block: tuple
        :param py_dict: Runtime argument dictionary matching ``self.cparams``
        :type py_dict: dict
        """

        device = self.get_device(py_dict)
        self.check_device(device, py_dict)

        self.set_contiguous(py_dict)

        self.check_ctypes(py_dict)

        self.check_half2(py_dict)

        py_dict = dict(sorted(py_dict.items()))
        self.check_keys(py_dict)
        assert sys.version_info.major >= 3 and sys.version_info.minor >= 6
        # 需要使用有序词典
        # python >= 3.6时，字典默认是有序的

        cp_kernel = cupy.RawKernel(
            self.full_codes,
            self.kernel_name,
            options=configure.cuda_compiler_options,
            backend=configure.cuda_compiler_backend,
        )

        with cuda_utils.DeviceEnvironment(device):
            cp_kernel(grid, block, self.get_ptrs(py_dict), *args_1, **kwargs)

    def add_param(self, ctype: str, cname: str):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-add-param-cn>` | :ref:`English <ckernel-add-param-en>`

        ----

        .. _ckernel-add-param-cn:

        * **中文**

        向 ``self.cparams`` 添加一个 CUDA 形参声明。

        :param ctype: CUDA 参数类型字符串
        :type ctype: str
        :param cname: CUDA 参数名
        :type cname: str
        :raise ValueError: 当参数名重复或与保留名冲突时

        ----

        .. _ckernel-add-param-en:

        * **English**

        Add one CUDA parameter declaration to ``self.cparams``.

        :param ctype: CUDA parameter type string
        :type ctype: str
        :param cname: CUDA parameter name
        :type cname: str
        :raise ValueError: If the name already exists or conflicts with reserved names
        """
        # example: ctype = 'const float *', cname = 'x'
        if cname in self.cparams:
            raise ValueError(f"{cname} has been added to cparams!")

        if cname in self.reserved_cnames:
            raise ValueError(
                f"{cname} is the reserved cname. You should change the name of your variable to avoid conflict."
            )

        self.cparams[cname] = ctype

    @property
    def declaration(self):
        codes = f"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void {self.kernel_name}(
        """
        self.cparams = dict(sorted(self.cparams.items()))
        params_list = []
        for cname, ctype in self.cparams.items():
            params_list.append(f"{ctype} {cname}")

        codes += ", ".join(params_list)

        codes += """
        )
        """
        return codes

    @property
    def head(self):
        return "{"

    @property
    def tail(self):
        return "}"

    @property
    def full_codes(self):
        r"""
        **API Language:**
        :ref:`中文 <ckernel-full-codes-cn>` |
        :ref:`English <ckernel-full-codes-en>`

        ----

        .. _ckernel-full-codes-cn:

        * **中文**

        返回拼接后的完整 CUDA 代码字符串。

        :return: 完整 CUDA 源码
        :rtype: str

        ----

        .. _ckernel-full-codes-en:

        * **English**

        Return the full CUDA source string assembled from declaration/head/core/tail.

        :return: Full CUDA source code
        :rtype: str
        """
        return (
            wrap_with_comment(self.declaration, "declaration")
            + wrap_with_comment(self.head, "head")
            + wrap_with_comment(self.core, "core")
            + wrap_with_comment(self.tail, "tail")
        )


class CKernel1D(CKernel):
    def __init__(self, *args, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-init-cn>` | :ref:`English <ckernel1d-init-en>`

        ----

        .. _ckernel1d-init-cn:

        * **中文**

        一维（逐元素）CUDA kernel 封装类，继承自 :class:`CKernel`。该类默认添加
        ``numel`` 形参并保留 ``index`` 作为线程索引变量名。

        :param kernel_name: CUDA kernel 名称（通过 ``*args, **kwargs`` 传入基类）
        :type kernel_name: str

        ----

        .. _ckernel1d-init-en:

        * **English**

        1D (element-wise) CUDA kernel wrapper inherited from :class:`CKernel`.
        It adds ``numel`` by default and reserves ``index`` as the thread index
        variable name.

        :param kernel_name: CUDA kernel name (forwarded to the base class via
            ``*args, **kwargs``)
        :type kernel_name: str
        """
        super().__init__(*args, **kwargs)
        self.cparams["numel"] = "const int &"
        self.reserved_cnames.append("index")

    @property
    def head(self):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-head-cn>` | :ref:`English <ckernel1d-head-en>`

        ----

        .. _ckernel1d-head-cn:

        * **中文**

        返回 1D kernel 头部代码，包含线程索引计算和 ``index < numel`` 边界判断。

        :return: CUDA 头部代码片段
        :rtype: str

        ----

        .. _ckernel1d-head-en:

        * **English**

        Return the 1D kernel head code, including thread-index computation and
        the ``index < numel`` guard.

        :return: CUDA head code snippet
        :rtype: str
        """
        codes = """
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < numel)
            {
        """
        return codes

    @property
    def tail(self):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-tail-cn>` | :ref:`English <ckernel1d-tail-en>`

        ----

        .. _ckernel1d-tail-cn:

        * **中文**

        返回 1D kernel 尾部代码，用于闭合头部中的代码块。

        :return: CUDA 尾部代码片段
        :rtype: str

        ----

        .. _ckernel1d-tail-en:

        * **English**

        Return the 1D kernel tail code that closes the blocks opened in
        :attr:`head`.

        :return: CUDA tail code snippet
        :rtype: str
        """
        codes = """
            }
        }
        """
        return codes

    def check_half2(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-check-half2-cn>` |
        :ref:`English <ckernel1d-check-half2-en>`

        ----

        .. _ckernel1d-check-half2-cn:

        * **中文**

        检查 ``py_dict`` 中 ``half``/``float16`` 张量的元素个数是否为偶数，以满足
        ``half2`` 访存需求。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        .. admonition:: 注意
            :class: note

            :meth:`CKernel1D.__call__` 会在执行前自动对奇数长度 half 张量补齐，因此通常
            无需手工补齐。

        ----

        .. _ckernel1d-check-half2-en:

        * **English**

        Validate that ``half``/``float16`` tensor lengths in ``py_dict`` are
        even, which is required by ``half2`` operations.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict

        .. admonition:: Note
            :class: note

            :meth:`CKernel1D.__call__` pads odd-length half tensors before kernel
            launch, so manual padding is usually unnecessary.
        """
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    assert value.numel() % 2 == 0, (
                        f"please pad the numel of {key} to assert mod 2 == 0! (for half2)"
                    )

            if isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    assert value.size % 2 == 0, (
                        f"please pad the numel of {key} to assert mod 2 == 0! (for half2)"
                    )

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-call-cn>` | :ref:`English <ckernel1d-call-en>`

        ----

        .. _ckernel1d-call-cn:

        * **中文**

        执行 1D CUDA kernel。对于 ``half``/``float16`` 且元素个数为奇数的张量，会先补齐
        末元素后再调用基类执行，完成后再恢复原始形状与长度。

        :param grid: CUDA grid 配置
        :type grid: tuple
        :param block: CUDA block 配置
        :type block: tuple
        :param py_dict: kernel 参数字典，键应与 ``self.cparams`` 对应
        :type py_dict: dict

        ----

        .. _ckernel1d-call-en:

        * **English**

        Execute the 1D CUDA kernel. For odd-length ``half``/``float16`` tensors,
        this method pads one trailing element before delegating to the base call,
        then removes the padding and restores the original shape.

        :param grid: CUDA grid configuration
        :type grid: tuple
        :param block: CUDA block configuration
        :type block: tuple
        :param py_dict: Runtime argument dictionary aligned with ``self.cparams``
        :type py_dict: dict
        """
        # pad half2
        pad_keys = []
        pad_shapes = []
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.half:
                if value.numel() % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    value = value.flatten()

                    value = torch.cat((value, value[-1].view(1)))

                    py_dict[key] = value

            elif isinstance(value, cupy.ndarray) and value.dtype == np.float16:
                if value.size % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    value = cupy.reshape(value, -1)

                    value = cupy.concatenate((value, cupy.reshape(value[-1], 1)))

                    py_dict[key] = value

        super().__call__(grid, block, py_dict, *args_1, **kwargs)

        # move pad values
        for key, shape in zip(pad_keys, pad_shapes):
            value = py_dict[key]
            value = value[:-1]

            if isinstance(value, torch.Tensor):
                value = value.view(shape)

            elif isinstance(value, cupy.ndarray):
                value = cupy.reshape(value, shape)

            py_dict[key] = value

    def simple_call(self, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel1d-simple-call-cn>` |
        :ref:`English <ckernel1d-simple-call-en>`

        ----

        .. _ckernel1d-simple-call-cn:

        * **中文**

        ``CKernel1D.__call__`` 的简化入口。自动从 ``kwargs`` 中推导设备与 ``numel``，
        并使用配置中的线程数计算 ``blocks`` 后执行 kernel。

        :param kwargs: kernel 参数键值对（不需要手动提供 ``numel``）
        :type kwargs: dict

        ----

        .. _ckernel1d-simple-call-en:

        * **English**

        A convenience wrapper of :meth:`CKernel1D.__call__`. It infers device and
        ``numel`` from ``kwargs``, computes launch blocks with configured threads,
        and launches the kernel.

        :param kwargs: Kernel argument mapping (``numel`` is inferred automatically)
        :type kwargs: dict
        """
        py_dict = kwargs
        device = self.get_device(py_dict)
        numel = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                numel = value.numel()
            elif isinstance(value, cupy.ndarray):
                numel = value.size

        if numel is None:
            raise ValueError("No torch.Tensor or cupy.ndarray in kwargs!")

        with cuda_utils.DeviceEnvironment(device):
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            py_dict["numel"] = numel
            self.__call__((blocks,), (threads,), py_dict)


class CKernel2D(CKernel):
    def __init__(self, kernel_name: str, reverse: bool = False):
        r"""
        **API Language:**
        :ref:`中文 <ckernel2d-init-cn>` | :ref:`English <ckernel2d-init-en>`

        ----

        .. _ckernel2d-init-cn:

        * **中文**

        二维 CUDA kernel 封装，继承自 :class:`CKernel`。默认包含 ``numel`` 与 ``N``
        两个内置参数，分别表示总元素数与每个时间步的元素数。二维张量按 ``[T, N]``
        解释，其中 ``T`` 为序列长度，``N`` 为单步元素数量。

        ``reverse`` 控制时间维循环方向：
        ``True`` 时使用 ``for(int t = numel - N + index; t >= 0; t -= dt)``；
        ``False`` 时使用 ``for(int t = index; t < numel; t += dt)``。

        :param kernel_name: CUDA kernel 名称
        :type kernel_name: str
        :param reverse: 是否使用反向时间循环
        :type reverse: bool

        ----

        .. _ckernel2d-init-en:

        * **English**

        A 2D CUDA kernel wrapper derived from :class:`CKernel`. It provides built-in
        parameters ``numel`` (total element count) and ``N`` (elements per time step).
        Any 2D tensor is interpreted as ``[T, N]``, where ``T`` is sequence length.

        ``reverse`` controls the temporal loop direction:
        ``True`` uses ``for(int t = numel - N + index; t >= 0; t -= dt)``;
        ``False`` uses ``for(int t = index; t < numel; t += dt)``.

        :param kernel_name: CUDA kernel name
        :type kernel_name: str
        :param reverse: Whether to use reverse temporal traversal
        :type reverse: bool
        """
        super().__init__(kernel_name)
        self.cparams["numel"] = "const int &"
        self.reverse = reverse
        self.cparams["N"] = "const int &"
        self.reserved_cnames.append("index")
        self.reserved_cnames.append("dt")
        self.reserved_cnames.append("t")

        self._pre_core = ""
        self._post_core = ""

    @property
    def pre_core(self):
        return self._pre_core

    @pre_core.setter
    def pre_core(self, value: str):
        self._pre_core = value

    @property
    def post_core(self):
        return self._post_core

    @post_core.setter
    def post_core(self, value: str):
        self._post_core = value

    def check_shape(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel2d-check-shape-cn>` |
        :ref:`English <ckernel2d-check-shape-en>`

        ----

        .. _ckernel2d-check-shape-cn:

        * **中文**

        检查 ``py_dict`` 中的张量维度。所有 ``torch.Tensor`` 与 ``cupy.ndarray``
        的维度都必须不超过 2。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        ----

        .. _ckernel2d-check-shape-en:

        * **English**

        Validate tensor dimensionality in ``py_dict``. All ``torch.Tensor`` and
        ``cupy.ndarray`` values must have ``ndim <= 2``.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict
        """
        # all tensors should be ndim <= 2
        for value in py_dict.values():
            if isinstance(value, torch.Tensor):
                assert value.ndim <= 2

            elif isinstance(value, cupy.ndarray):
                assert value.ndim <= 2

    def check_half2(self, py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ckernel2d-check-half2-cn>` |
        :ref:`English <ckernel2d-check-half2-en>`

        ----

        .. _ckernel2d-check-half2-cn:

        * **中文**

        检查 ``py_dict`` 中半精度张量是否满足 ``half2`` 对齐要求（偶数长度）。
        对 ``torch.half`` 和 ``np.float16``：
        1D 张量要求 ``numel`` 为偶数，2D 张量要求 ``shape[1]`` 为偶数。

        :param py_dict: kernel 参数字典
        :type py_dict: dict

        .. admonition:: Note
            :class: note

            实际执行前，:meth:`CKernel2D.__call__` 会自动补齐奇数长度半精度张量；
            本函数用于约束与校验。

        ----

        .. _ckernel2d-check-half2-en:

        * **English**

        Check whether half-precision tensors in ``py_dict`` satisfy ``half2``
        alignment requirements (even length). For ``torch.half`` and
        ``np.float16`` values:
        1D tensors require even ``numel``; 2D tensors require even ``shape[1]``.

        :param py_dict: Kernel argument dictionary
        :type py_dict: dict

        .. admonition:: Note
            :class: note

            :meth:`CKernel2D.__call__` performs automatic padding for odd-sized
            half tensors before launch. This method is for validation checks.
        """
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    if value.ndim <= 1:
                        assert value.numel() % 2 == 0
                    elif value.ndim == 2:
                        assert value.shape[1] % 2 == 0

            elif isinstance(value, cupy.ndarray):
                if value.dtype == np.float16:
                    if value.ndim <= 1:
                        assert value.size % 2 == 0
                    elif value.ndim == 2:
                        assert value.shape[1] % 2 == 0

    def __call__(self, grid: tuple, block: tuple, py_dict: dict, *args_1, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel2d-call-cn>` | :ref:`English <ckernel2d-call-en>`

        ----

        .. _ckernel2d-call-cn:

        * **中文**

        执行二维 CUDA kernel。``*args_1`` 与 ``**kwargs`` 会透传给
        :class:`cupy.RawKernel` 调用。

        ``py_dict`` 中 ``key`` 必须与 ``self.cparams`` 的形参名一一对应，
        ``value`` 需为匹配数据类型的 ``torch.Tensor`` 或 ``cupy.ndarray``。
        键顺序可以任意，内部会按形参顺序重排。

        :param grid: CUDA grid 配置
        :type grid: tuple
        :param block: CUDA block 配置
        :type block: tuple
        :param py_dict: kernel 参数字典
        :type py_dict: dict

        .. admonition:: Note
            :class: note

            ``py_dict`` 中张量必须为 1D 或 2D。对于奇数长度半精度张量，
            调用前会自动补齐，执行后移除补齐并恢复原始形状。

        ----

        .. _ckernel2d-call-en:

        * **English**

        Execute the 2D CUDA kernel. ``*args_1`` and ``**kwargs`` are forwarded
        directly to :class:`cupy.RawKernel`.

        Keys in ``py_dict`` must match ``self.cparams`` one-to-one, and values must
        be ``torch.Tensor``/``cupy.ndarray`` objects with compatible dtypes. Key
        order is arbitrary because arguments are aligned internally by formal
        parameter order.

        :param grid: CUDA grid configuration
        :type grid: tuple
        :param block: CUDA block configuration
        :type block: tuple
        :param py_dict: Kernel argument dictionary
        :type py_dict: dict

        .. admonition:: Note
            :class: note

            Tensor inputs must be 1D or 2D. Odd-sized half-precision tensors are
            padded before launch, then unpadded and reshaped back afterward.
        """
        self.check_shape(py_dict)

        # pad half2
        pad_keys = []
        pad_shapes = []
        for key, value in py_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.half:
                if value.ndim <= 1:
                    # 1D tensor
                    if value.numel() % 2 != 0:
                        pad_shapes.append(value.shape)
                        pad_keys.append(key)
                        value = value.flatten()

                        value = torch.cat((value, value[-1].view(1)))

                        py_dict[key] = value

                elif value.shape[1] % 2 != 0:
                    # 2D tensor with shape = [T, N] and N % 2 != 0
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)

                    value = torch.cat((value, value[:, -1].view(-1, 1)), dim=1)
                    # [T, N] -> [T, N + 1]
                    py_dict[key] = value

            elif isinstance(value, cupy.ndarray) and value.dtype == np.float16:
                if value.ndim <= 1:
                    # 1D tensor
                    if value.size % 2 != 0:
                        pad_shapes.append(value.shape)
                        pad_keys.append(key)
                        value = cupy.reshape(value, -1)

                        value = cupy.concatenate((value, cupy.reshape(value[-1], 1)))

                        py_dict[key] = value

                elif value.shape[1] % 2 != 0:
                    pad_shapes.append(value.shape)
                    pad_keys.append(key)
                    # [T, N] -> [T, N + 1]

                    value = cupy.concatenate(
                        (value, cupy.reshape(value[:, -1], (-1, 1))), axis=1
                    )
                    py_dict[key] = value

        super().__call__(grid, block, py_dict, *args_1, **kwargs)

        # move pad values
        for i, key in enumerate(pad_keys):
            value = py_dict[key]
            shape = pad_shapes[i]
            if isinstance(value, torch.Tensor):
                if value.ndim <= 1:
                    value = value[:-1]
                    value = value.view(shape)
                else:
                    value = value[:, :-1]

            elif isinstance(value, cupy.ndarray):
                if value.ndim <= 1:
                    value = value[:, -1]
                    value = cupy.reshape(value, shape)

                else:
                    value = value[:, :-1]

            py_dict[key] = value

    @property
    def head(self):
        codes = """
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        """

        codes += wrap_with_comment(self.pre_core, "pre_core")

        if self.reverse:
            codes += """
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            """
        else:
            codes += """
                for(int t = index; t < numel; t += dt)
                {
            """
        return codes

    @property
    def tail(self):
        codes = """
                }
        """

        codes += wrap_with_comment(self.post_core, "post_core")

        codes += """
            }
        }
        """
        return codes

    def simple_call(self, **kwargs):
        r"""
        **API Language:**
        :ref:`中文 <ckernel2d-simple-call-cn>` |
        :ref:`English <ckernel2d-simple-call-en>`

        ----

        .. _ckernel2d-simple-call-cn:

        * **中文**

        :meth:`CKernel2D.__call__` 的便捷封装。该函数会从 ``kwargs`` 自动推断
        设备、``numel``、``N``，并根据配置计算 CUDA ``threads`` 与 ``blocks`` 后执行。

        :param kwargs: kernel 参数键值对（无需手动传入 ``numel`` 与 ``N``）
        :type kwargs: dict

        ----

        .. _ckernel2d-simple-call-en:

        * **English**

        A convenience wrapper of :meth:`CKernel2D.__call__`. It infers device,
        ``numel``, and ``N`` from ``kwargs``, computes launch ``threads`` and
        ``blocks`` from configuration, and then launches the kernel.

        :param kwargs: Kernel argument mapping (``numel`` and ``N`` are inferred)
        :type kwargs: dict
        """
        py_dict = kwargs
        device = self.get_device(py_dict)

        numel = None
        N = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                numel = value.numel()
                N = value.shape[1]
            elif isinstance(value, cupy.ndarray) and value.ndim == 2:
                numel = value.size
                N = value.shape[1]

        if numel is None or N is None:
            raise ValueError("No 2D torch.Tensor or cupy.ndarray in kwargs!")

        with cuda_utils.DeviceEnvironment(device):
            threads = configure.cuda_threads
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)
            py_dict["numel"] = numel
            py_dict["N"] = N
            self.__call__((blocks,), (threads,), py_dict)


class CodeTyper:
    def __init__(self, indent_num: int):
        r"""
        **API Language:**
        :ref:`中文 <codetyper-init-cn>` | :ref:`English <codetyper-init-en>`

        ----

        .. _codetyper-init-cn:

        * **中文**

        CUDA 代码缩进与拼接工具。内部维护 ``self.codes``，可逐段追加代码并按缩进格式化。

        :param indent_num: 初始缩进空格数
        :type indent_num: int

        ----

        .. _codetyper-init-en:

        * **English**

        A helper for formatting and assembling CUDA code with indentation. The
        accumulated code text is stored in ``self.codes``.

        :param indent_num: Number of spaces used for initial indentation
        :type indent_num: int
        """
        self.indent = " " * indent_num
        self.codes = "\n"

    def append(self, codes: str):
        r"""
        **API Language:**
        :ref:`中文 <codetyper-append-cn>` | :ref:`English <codetyper-append-en>`

        ----

        .. _codetyper-append-cn:

        * **中文**

        将输入 CUDA 代码片段追加到 ``self.codes``。函数按 ``;`` 分句并逐句写入，
        同时处理 ``{``/``}`` 这类块边界语句。

        :param codes: 待追加的 CUDA 代码
        :type codes: str

        ----

        .. _codetyper-append-en:

        * **English**

        Append CUDA code snippets into ``self.codes``. The method splits by ``;``
        and writes each statement with current indentation, while handling ``{`` and
        ``}`` block boundary tokens.

        :param codes: CUDA code snippet to append
        :type codes: str
        """
        codes = codes.replace("\n", "")
        codes = codes.split(";")
        for i in range(codes.__len__()):
            if codes[i].__len__() > 0:
                if codes[i] in ("{", "}"):
                    self.codes += self.indent + codes[i] + "\n"
                else:
                    self.codes += self.indent + codes[i] + ";\n"


class CodeBlock:
    def __init__(self, env: CodeTyper):
        r"""
        **API Language:**
        :ref:`中文 <codeblock-init-cn>` | :ref:`English <codeblock-init-en>`

        ----

        .. _codeblock-init-cn:

        * **中文**

        ``CodeTyper`` 的上下文管理器工具，用于自动插入代码块 ``{...}`` 并维护缩进，
        便于组织包含中间变量的多行 CUDA 逻辑。

        :param env: 目标代码环境
        :type env: CodeTyper

        ----

        .. _codeblock-init-en:

        * **English**

        A context-manager utility for ``CodeTyper`` that inserts ``{...}`` blocks
        and adjusts indentation automatically. It is useful for composing multi-line
        CUDA logic with intermediate variables.

        :param env: Target code-typing environment
        :type env: CodeTyper
        """
        self.env = env

    def __enter__(self):
        r"""
        **API Language:**
        :ref:`中文 <codeblock-enter-cn>` | :ref:`English <codeblock-enter-en>`

        ----

        .. _codeblock-enter-cn:

        * **中文**

        进入上下文时写入 ``{`` 并增加一级缩进。

        :return: 当前上下文对象（默认返回 ``None``）
        :rtype: None

        ----

        .. _codeblock-enter-en:

        * **English**

        Enter the context by appending ``{`` and increasing indentation by one level.

        :return: Current context object (defaults to ``None``)
        :rtype: None
        """
        self.env.append("{")
        self.env.indent += " "

    def __exit__(self, exc_type, exc_val, exc_tb):
        r"""
        **API Language:**
        :ref:`中文 <codeblock-exit-cn>` | :ref:`English <codeblock-exit-en>`

        ----

        .. _codeblock-exit-cn:

        * **中文**

        退出上下文时回退一级缩进并写入 ``}``。

        :param exc_type: 异常类型
        :type exc_type: type
        :param exc_val: 异常值
        :type exc_val: BaseException
        :param exc_tb: 回溯对象
        :type exc_tb: traceback
        :return: ``False``（默认行为，不屏蔽异常）
        :rtype: bool

        ----

        .. _codeblock-exit-en:

        * **English**

        Exit the context by decreasing indentation by one level and appending ``}``.

        :param exc_type: Exception type
        :type exc_type: type
        :param exc_val: Exception value
        :type exc_val: BaseException
        :param exc_tb: Traceback object
        :type exc_tb: traceback
        :return: ``False`` by default (exceptions are not suppressed)
        :rtype: bool
        """
        self.env.indent = self.env.indent[:-1]
        self.env.append("}")
