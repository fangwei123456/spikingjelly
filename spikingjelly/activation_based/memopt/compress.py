from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

__all__ = [
    "SpikeCompressor",
    "BooleanSpikeCompressor",
    "Uint8SpikeCompressor",
    "BitSpikeCompressor",
    "SparseSpikeCompressor",
]


class SpikeCompressor(ABC):
    r"""
    **API Language** - :ref:`中文 <spike-compressor-cn>` | :ref:`English <spike-compressor-en>`

    ----

    .. _spike-compressor-cn:

    * **中文**

    无状态张量压缩器的抽象基类。``compress`` 返回的 payload 必须包含
    ``decompress`` 恢复 shape、dtype 和 device 所需的信息。实现不得把单次调用的
    信息保存在实例中。

    ----

    .. _spike-compressor-en:

    * **English**

    Abstract base class for stateless tensor compressors. The payload returned by
    ``compress`` must contain everything ``decompress`` needs to restore the shape,
    dtype, and device. Implementations must not keep per-call data on the instance.
    """

    @abstractmethod
    def compress(self, x: torch.Tensor) -> object:
        r"""
        **API Language** - :ref:`中文 <spike-compressor-compress-cn>` | :ref:`English <spike-compressor-compress-en>`

        ----

        .. _spike-compressor-compress-cn:

        * **中文**

        压缩一个张量。

        :param x: 输入张量；支持的 shape、dtype 和 device 由实现决定。
        :type x: torch.Tensor
        :return: 可传给 :meth:`decompress` 的 payload。
        :rtype: object

        ----

        .. _spike-compressor-compress-en:

        * **English**

        Compress one tensor.

        :param x: Input tensor; supported shapes, dtypes, and devices are defined by
            the implementation.
        :type x: torch.Tensor
        :return: Payload accepted by :meth:`decompress`.
        :rtype: object
        """

    @abstractmethod
    def decompress(self, packed: object) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <spike-compressor-decompress-cn>` | :ref:`English <spike-compressor-decompress-en>`

        ----

        .. _spike-compressor-decompress-cn:

        * **中文**

        恢复一个张量。

        :param packed: :meth:`compress` 返回的 payload。
        :type packed: object
        :return: 恢复 shape、dtype 和 device 的张量。
        :rtype: torch.Tensor

        ----

        .. _spike-compressor-decompress-en:

        * **English**

        Restore one tensor.

        :param packed: Payload returned by :meth:`compress`.
        :type packed: object
        :return: Tensor restored to its original shape, dtype, and device.
        :rtype: torch.Tensor
        """


class BooleanSpikeCompressor(SpikeCompressor):
    def __init__(self) -> None:
        r"""
        **API Language** - :ref:`中文 <boolean-spike-compressor-cn>` | :ref:`English <boolean-spike-compressor-en>`

        ----

        .. _boolean-spike-compressor-cn:

        * **中文**

        将严格取值为 0 或 1 的脉冲张量保存为 ``bool``。支持 PyTorch 可用的
        CPU 和加速器 device，并在解压时恢复原 shape、dtype 和 device。

        ----

        .. _boolean-spike-compressor-en:

        * **English**

        Store spike tensors whose values are strictly 0 or 1 as ``bool``. It works
        on CPU and accelerator devices supported by PyTorch, and restores the
        original shape, dtype, and device.
        """

    def compress(self, x: torch.Tensor) -> object:
        r"""
        **API Language** - :ref:`中文 <boolean-spike-compress-cn>` | :ref:`English <boolean-spike-compress-en>`

        ----

        .. _boolean-spike-compress-cn:

        * **中文**

        将严格二值张量转换为 ``bool``。

        :param x: 任意 shape 和 device 的张量，元素必须为 0 或 1。
        :type x: torch.Tensor
        :return: ``bool`` 张量、原 dtype 和原 shape。
        :rtype: object

        ----

        .. _boolean-spike-compress-en:

        * **English**

        Convert a strictly binary tensor to ``bool``.

        :param x: Tensor of any shape and device whose values must be 0 or 1.
        :type x: torch.Tensor
        :return: Boolean tensor, original dtype, and original shape.
        :rtype: object
        """
        return x.to(torch.bool), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <boolean-spike-decompress-cn>` | :ref:`English <boolean-spike-decompress-en>`

        ----

        .. _boolean-spike-decompress-cn:

        * **中文**

        恢复 :meth:`compress` 生成的 payload。

        :param packed: ``bool`` 张量、原 dtype 和原 shape。
        :type packed: object
        :return: 恢复原 shape、dtype 和 device 的张量。
        :rtype: torch.Tensor

        ----

        .. _boolean-spike-decompress-en:

        * **English**

        Restore a payload produced by :meth:`compress`.

        :param packed: Boolean tensor, original dtype, and original shape.
        :type packed: object
        :return: Tensor with its original shape, dtype, and device.
        :rtype: torch.Tensor
        """
        values, dtype, shape = packed
        return values.to(dtype=dtype).reshape(shape)


class Uint8SpikeCompressor(SpikeCompressor):
    def __init__(self) -> None:
        r"""
        **API Language** - :ref:`中文 <uint8-spike-compressor-cn>` | :ref:`English <uint8-spike-compressor-en>`

        ----

        .. _uint8-spike-compressor-cn:

        * **中文**

        将取值范围为 0 到 255 的整数脉冲保存为 ``uint8``，并在解压时恢复原
        shape、dtype 和 device。

        ----

        .. _uint8-spike-compressor-en:

        * **English**

        Store integer-valued spikes in the range 0 to 255 as ``uint8``, then restore
        their original shape, dtype, and device.
        """

    def compress(self, x: torch.Tensor) -> object:
        r"""
        **API Language** - :ref:`中文 <uint8-spike-compress-cn>` | :ref:`English <uint8-spike-compress-en>`

        ----

        .. _uint8-spike-compress-cn:

        * **中文**

        将整数值张量转换为 ``uint8``。

        :param x: 任意 shape 和 device 的张量，元素必须是 ``[0, 255]`` 内的整数。
        :type x: torch.Tensor
        :return: ``uint8`` 张量、原 dtype 和原 shape。
        :rtype: object

        ----

        .. _uint8-spike-compress-en:

        * **English**

        Convert an integer-valued tensor to ``uint8``.

        :param x: Tensor of any shape and device with integer values in ``[0, 255]``.
        :type x: torch.Tensor
        :return: Uint8 tensor, original dtype, and original shape.
        :rtype: object
        """
        return x.to(torch.uint8), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <uint8-spike-decompress-cn>` | :ref:`English <uint8-spike-decompress-en>`

        ----

        .. _uint8-spike-decompress-cn:

        * **中文**

        恢复 :meth:`compress` 生成的 payload。

        :param packed: ``uint8`` 张量、原 dtype 和原 shape。
        :type packed: object
        :return: 恢复原 shape、dtype 和 device 的张量。
        :rtype: torch.Tensor

        ----

        .. _uint8-spike-decompress-en:

        * **English**

        Restore a payload produced by :meth:`compress`.

        :param packed: Uint8 tensor, original dtype, and original shape.
        :type packed: object
        :return: Tensor with its original shape, dtype, and device.
        :rtype: torch.Tensor
        """
        values, dtype, shape = packed
        return values.to(dtype=dtype).reshape(shape)


class BitSpikeCompressor(SpikeCompressor):
    def __init__(self) -> None:
        r"""
        **API Language** - :ref:`中文 <bit-spike-compressor-cn>` | :ref:`English <bit-spike-compressor-en>`

        ----

        .. _bit-spike-compressor-cn:

        * **中文**

        将 8 个严格取值为 0 或 1 的脉冲打包进一个字节。CPU 和 CUDA 使用相同的
        PyTorch 张量运算，且该路径可被 ``torch.compile`` 捕获。解压时恢复原
        shape、dtype 和 device。

        ----

        .. _bit-spike-compressor-en:

        * **English**

        Pack eight spikes whose values are strictly 0 or 1 into one byte. CPU and
        CUDA use the same PyTorch tensor operations, and the path can be captured by
        ``torch.compile``. Decompression restores the original shape, dtype, and
        device.
        """

    def compress(self, x: torch.Tensor) -> object:
        r"""
        **API Language** - :ref:`中文 <bit-spike-compress-cn>` | :ref:`English <bit-spike-compress-en>`

        ----

        .. _bit-spike-compress-cn:

        * **中文**

        将严格二值张量按每 8 个元素一组打包。

        :param x: 任意 shape 和 device 的张量，元素必须为 0 或 1。
        :type x: torch.Tensor
        :return: 打包后的字节张量、原 dtype 和原 shape。
        :rtype: object

        ----

        .. _bit-spike-compress-en:

        * **English**

        Pack a strictly binary tensor in groups of eight values.

        :param x: Tensor of any shape and device whose values must be 0 or 1.
        :type x: torch.Tensor
        :return: Packed byte tensor, original dtype, and original shape.
        :rtype: object
        """
        flat = x.to(torch.uint8).reshape(-1)
        padding = (-flat.numel()) % 8
        if padding:
            flat = F.pad(flat, (0, padding))
        shifts = torch.arange(8, dtype=torch.uint8, device=x.device)
        values = (flat.reshape(-1, 8) << shifts).sum(dim=1, dtype=torch.int16)
        return values.to(torch.uint8), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <bit-spike-decompress-cn>` | :ref:`English <bit-spike-decompress-en>`

        ----

        .. _bit-spike-decompress-cn:

        * **中文**

        恢复 :meth:`compress` 生成的 payload。

        :param packed: 字节张量、原 dtype 和原 shape。
        :type packed: object
        :return: 恢复原 shape、dtype 和 device 的张量。
        :rtype: torch.Tensor

        ----

        .. _bit-spike-decompress-en:

        * **English**

        Restore a payload produced by :meth:`compress`.

        :param packed: Byte tensor, original dtype, and original shape.
        :type packed: object
        :return: Tensor with its original shape, dtype, and device.
        :rtype: torch.Tensor
        """
        values, dtype, shape = packed
        shifts = torch.arange(8, dtype=torch.uint8, device=values.device)
        flat = ((values.unsqueeze(1) >> shifts) & 1).reshape(-1)
        return flat[: torch.Size(shape).numel()].to(dtype=dtype).reshape(shape)


class SparseSpikeCompressor(SpikeCompressor):
    def __init__(self, dtype: torch.dtype = torch.int64) -> None:
        r"""
        **API Language** - :ref:`中文 <sparse-spike-compressor-cn>` | :ref:`English <sparse-spike-compressor-en>`

        ----

        .. _sparse-spike-compressor-cn:

        * **中文**

        只保存严格二值脉冲中非零元素的一维索引。适合稀疏输入；解压时恢复原
        shape、dtype 和 device。

        :param dtype: 保存索引的整数 dtype。
        :type dtype: torch.dtype

        ----

        .. _sparse-spike-compressor-en:

        * **English**

        Store only the flattened indices of nonzero elements in a strictly binary
        spike tensor. This suits sparse inputs; decompression restores the original
        shape, dtype, and device.

        :param dtype: Integer dtype used for stored indices.
        :type dtype: torch.dtype
        """
        self.dtype = dtype

    def compress(self, x: torch.Tensor) -> object:
        r"""
        **API Language** - :ref:`中文 <sparse-spike-compress-cn>` | :ref:`English <sparse-spike-compress-en>`

        ----

        .. _sparse-spike-compress-cn:

        * **中文**

        保存严格二值张量中非零元素的一维索引。

        :param x: 任意 shape 和 device 的张量，元素必须为 0 或 1。
        :type x: torch.Tensor
        :return: 非零索引、原 dtype 和原 shape。
        :rtype: object

        ----

        .. _sparse-spike-compress-en:

        * **English**

        Store the flattened indices of nonzero values in a strictly binary tensor.

        :param x: Tensor of any shape and device whose values must be 0 or 1.
        :type x: torch.Tensor
        :return: Nonzero indices, original dtype, and original shape.
        :rtype: object
        """
        indices = torch.nonzero(x.reshape(-1), as_tuple=False).reshape(-1)
        return indices.to(self.dtype), x.dtype, x.shape

    def decompress(self, packed: object) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <sparse-spike-decompress-cn>` | :ref:`English <sparse-spike-decompress-en>`

        ----

        .. _sparse-spike-decompress-cn:

        * **中文**

        恢复 :meth:`compress` 生成的 payload。

        :param packed: 非零索引、原 dtype 和原 shape。
        :type packed: object
        :return: 恢复原 shape、dtype 和 device 的张量。
        :rtype: torch.Tensor

        ----

        .. _sparse-spike-decompress-en:

        * **English**

        Restore a payload produced by :meth:`compress`.

        :param packed: Nonzero indices, original dtype, and original shape.
        :type packed: object
        :return: Tensor with its original shape, dtype, and device.
        :rtype: torch.Tensor
        """
        indices, dtype, shape = packed
        flat = torch.zeros(
            torch.Size(shape).numel(), dtype=dtype, device=indices.device
        )
        return flat.scatter_(0, indices.to(torch.int64), 1).reshape(shape)
