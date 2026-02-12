import abc
import torch
import logging


__all__ = [
    "BaseSpikeCompressor",
    "NullSpikeCompressor",
    "BooleanSpikeCompressor",
    "Uint8SpikeCompressor",
    "BitSpikeCompressor",
    "SparseSpikeCompressor",
]

try:
    import triton
except BaseException as e:
    triton = None

if triton is not None:
    logging.info(
        "spikingjelly.activation_based.triton_kernel.compress: "
        "Use Triton backend for bit spike compression"
    )
    from ..triton_kernel import bit_spike_compress, bit_spike_decompress
else:
    logging.info(
        "spikingjelly.activation_based.triton_kernel.compress: "
        "Use PyTorch backend for bit spike compression"
    )

    def bit_spike_compress(s_seq: torch.Tensor) -> torch.Tensor:
        s_seq = s_seq.to(dtype=torch.bool).reshape(-1)
        compressed_shape = (s_seq.numel() + 7) // 8
        s_seq_compressed = torch.zeros(
            compressed_shape, dtype=torch.uint8, device=s_seq.device
        )
        for i in range(8):
            sliced = s_seq[i::8].to(dtype=torch.uint8)
            sliced_len = sliced.numel()
            if sliced_len > 0:
                s_seq_compressed[:sliced_len] |= sliced << i
        return s_seq_compressed

    def bit_spike_decompress(s_seq_compressed: torch.Tensor, shape) -> torch.Tensor:
        decompressed_len = shape.numel()
        s_seq_decompressed = torch.zeros(
            decompressed_len, dtype=torch.bool, device=s_seq_compressed.device
        )
        for i in range(8):
            sliced_len = (decompressed_len - i + 7) // 8
            sliced = ((s_seq_compressed >> i) & 1)[:sliced_len]
            s_seq_decompressed[i::8] = sliced
        return s_seq_decompressed.reshape(shape)


class BaseSpikeCompressor(abc.ABC):
    requires_strictly_binary = False

    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <BaseSpikeCompressor.__init__-cn>` | :ref:`English <BaseSpikeCompressor.__init__-en>`

        ----

        .. _BaseSpikeCompressor.__init__-cn:

        * **中文**

        脉冲压缩器的抽象基类。欲实现脉冲压缩器，需继承该抽象基类并实现 ``_compress`` 和 ``_decompress`` 方法。

        ----

        .. _BaseSpikeCompressor.__init__-en:

        * **English**

        Abstract base class for spike compressors.
        To implement a spike compressor, you need to inherit this abstract base class
        and implement the ``_compress`` and ``_decompress`` methods.
        """
        pass

    @abc.abstractmethod
    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        pass

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language:**
        :ref:`中文 <BaseSpikeCompressor.compress-cn>` | :ref:`English <BaseSpikeCompressor.compress-en>`

        ----

        .. _BaseSpikeCompressor.compress-cn:

        * **中文**

        压缩缩脉冲序列。

        :param s_seq: 输入脉冲序列
        :type s_seq: torch.Tensor

        :return: 压缩后的脉冲序列
        :rtype: torch.Tensor

        ----

        .. _BaseSpikeCompressor.compress-en:

        * **English**

        Compress spike sequence.

        :param s_seq: input spike sequence
        :type s_seq: torch.Tensor

        :return: compressed spike sequence
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self._compress(s_seq)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        r"""
        **API Language:**
        :ref:`中文 <BaseSpikeCompressor.decompress-cn>` | :ref:`English <BaseSpikeCompressor.decompress-en>`

        ----

        .. _BaseSpikeCompressor.decompress-cn:

        * **中文**

        解压缩脉冲序列。

        :param s_seq: 压缩的脉冲序列
        :type s_seq: torch.Tensor

        :param shape: 原始形状
        :type shape: tuple or torch.Size

        :return: 解压缩后的脉冲序列
        :rtype: torch.Tensor

        ----

        .. _BaseSpikeCompressor.decompress-en:

        * **English**

        Decompress spike sequence.

        :param s_seq: compressed spike sequence
        :type s_seq: torch.Tensor

        :param shape: original shape
        :type shape: tuple or torch.Size

        :return: decompressed spike sequence
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self._decompress(s_seq, shape)


class NullSpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = False

    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <NullSpikeCompressor.__init__-cn>` | :ref:`English <NullSpikeCompressor.__init__-en>`

        ----

        .. _NullSpikeCompressor.__init__-cn:

        * **中文**

        空脉冲压缩器。压缩和解压缩过程都是恒等映射。

        ``NullSpikeCompressor`` 是唯一能够无损处理非二进制张量的"脉冲压缩器"模块。例如，SNN的输入层
        应该始终使用 ``NullSpikeCompressor`` ，因为其输入是浮点张量而不是二值张量。

        ----

        .. _NullSpikeCompressor.__init__-en:

        * **English**

        Null spike compressor. The compression and decompression process are identity mapping.

        ``NullSpikeCompressor`` is the only compressor module that can deal with non-binary
        tensors losslessly. For instance, the input layer should always use
        ``NullSpikeCompressor``, as its input is a float tensor rather than a binary tensor.

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            from spikingjelly.activation_based.memopt.compress import NullSpikeCompressor

            compressor = NullSpikeCompressor()
            x = torch.randn(32, 10)
            compressed = compressor.compress(x)
            decompressed = compressor.decompress(compressed, x.shape)
        """
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq


class BooleanSpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = True

    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <BooleanSpikeCompressor.__init__-cn>` | :ref:`English <BooleanSpikeCompressor.__init__-en>`

        ----

        .. _BooleanSpikeCompressor.__init__-cn:

        * **中文**

        布尔脉冲压缩器。

        将脉冲序列转换为布尔类型以节省内存。要求输入必须是严格的二进制脉冲。

        ----

        .. _BooleanSpikeCompressor.__init__-en:

        * **English**

        Boolean spike compressor.

        Convert spike sequences to boolean type to save memory.
        Requires input to be strictly binary spikes.

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            from spikingjelly.activation_based.memopt.compress import BooleanSpikeCompressor

            compressor = BooleanSpikeCompressor()
            spikes = torch.randint(0, 2, (32, 100)).float()
            compressed = compressor.compress(spikes)
            decompressed = compressor.decompress(compressed, spikes.shape)
        """
        super().__init__()
        self.s_seq_dtype = torch.float32

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        self.s_seq_dtype = s_seq.dtype
        return s_seq.to(dtype=torch.bool)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq.to(dtype=self.s_seq_dtype).reshape(shape)


class Uint8SpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = False

    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <Uint8SpikeCompressor.__init__-cn>` | :ref:`English <Uint8SpikeCompressor.__init__-en>`

        ----

        .. _Uint8SpikeCompressor.__init__-cn:

        * **中文**

        Uint8脉冲压缩器。

        将脉冲序列转换为uint8类型以节省内存。可以处理非二进制整数数值。

        ----

        .. _Uint8SpikeCompressor.__init__-en:

        * **English**

        Uint8 spike compressor.

        Convert spike sequences to uint8 type to save memory. Can handle non-binary integer values.

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            from spikingjelly.activation_based.memopt.compress import Uint8SpikeCompressor

            compressor = Uint8SpikeCompressor()
            x = torch.randn(32, 10)
            compressed = compressor.compress(x)
            decompressed = compressor.decompress(compressed, x.shape)
        """
        super().__init__()
        self.s_seq_dtype = torch.float32

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        self.s_seq_dtype = s_seq.dtype
        return s_seq.to(dtype=torch.uint8)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq.to(dtype=self.s_seq_dtype).reshape(shape)


class BitSpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = True

    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <BitSpikeCompressor.__init__-cn>` | :ref:`English <BitSpikeCompressor.__init__-en>`

        ----

        .. _BitSpikeCompressor.__init__-cn:

        * **中文**

        比特脉冲压缩器。

        使用位压缩技术将8个二进制脉冲压缩到一个字节中，实现极高的内存压缩比。
        要求输入必须是严格的二进制脉冲（0或1）。

        ----

        .. _BitSpikeCompressor.__init__-en:

        * **English**

        Bit-level spike compressor.

        Use bit compression technique to compress 8 binary spikes into one byte,
        achieving high memory compression ratio.
        Requires input to be strictly binary spikes (0 or 1).

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            from spikingjelly.activation_based.memopt.compress import BitSpikeCompressor

            compressor = BitSpikeCompressor()
            spikes = torch.randint(0, 2, (32, 1000)).float()
            compressed = compressor.compress(spikes)
            decompressed = compressor.decompress(compressed, spikes.shape)
        """
        super().__init__()
        self.s_seq_dtype = torch.float32

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        # s_seq: float32
        return bit_spike_compress(s_seq)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        s_seq_decompressed = bit_spike_decompress(s_seq, shape)
        return s_seq_decompressed.to(dtype=self.s_seq_dtype)


class SparseSpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = True

    def __init__(self, dtype=torch.int64):
        r"""
        **API Language:**
        :ref:`中文 <SparseSpikeCompressor.__init__-cn>` | :ref:`English <SparseSpikeCompressor.__init__-en>`

        ----

        .. _SparseSpikeCompressor.__init__-cn:

        * **中文**

        稀疏脉冲压缩器。

        只存储非零脉冲的位置索引，适用于稀疏脉冲序列。
        要求输入必须是严格的二进制脉冲（0或1）。

        :param dtype: 索引数据类型，默认为 ``torch.int64``
        :type dtype: torch.dtype

        ----

        .. _SparseSpikeCompressor.__init__-en:

        * **English**

        Sparse spike compressor.

        Only store the position indices of non-zero spikes, suitable for sparse spike sequences.
        Requires input to be strictly binary spikes (0 or 1).

        :param dtype: index data type. Default to ``torch.int64``
        :type dtype: torch.dtype

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            from spikingjelly.activation_based.memopt.compress import SparseSpikeCompressor

            compressor = SparseSpikeCompressor()
            spikes = (torch.rand(32, 1000) < 0.04).float()
            compressed = compressor.compress(spikes)
            decompressed = compressor.decompress(compressed, spikes.shape)
        """
        super().__init__()
        self.dtype = dtype
        self.s_seq_dtype = torch.float32

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        indices = torch.nonzero(s_seq.reshape(-1))
        self.s_seq_dtype = s_seq.dtype
        return indices.to(dtype=self.dtype)

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        s_seq_decompressed = torch.zeros(
            shape.numel(), dtype=self.s_seq_dtype, device=s_seq.device
        )
        s_seq_decompressed = s_seq_decompressed.scatter_(
            dim=0,
            index=s_seq.to(dtype=torch.int64).reshape(-1),
            value=1,
        )
        return s_seq_decompressed.reshape(shape)
