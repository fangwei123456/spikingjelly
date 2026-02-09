import abc
import torch
import logging


try:
    import triton
except BaseException as e:
    triton = None

if triton is not None:
    logging.info(
        "spikingjelly.activation_based.triton_kernel.compress: "
        "Use Triton backend for bit spike compression"
    )
    from ...triton_kernel import bit_spike_compress, bit_spike_decompress
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

    def bit_spike_decompress(
        s_seq_compressed: torch.Tensor, shape
    ) -> torch.Tensor:
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
        pass

    @abc.abstractmethod
    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        pass

    def compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._compress(s_seq)

    def decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        with torch.no_grad():
            return self._decompress(s_seq, shape)


class NullSpikeCompressor(BaseSpikeCompressor):
    """Similar to IdentitySpikeCompressor, but the decompressed tensor must have
    the same dtype as the original one.

    NullSpikeCompressor is used for dealing with non-binary tensors. It is the
    only "spike compressor" module that can deal with non-binary tensors
    losslessly (actually, we shouldn't call is a "spike" compressor). For
    instance, the input layer should always use NullSpikeCompressor, as its
    input is a float tensor rather than a spike tensor.
    """

    requires_strictly_binary = False

    def __init__(self):
        super().__init__()

    def _compress(self, s_seq: torch.Tensor) -> torch.Tensor:
        return s_seq

    def _decompress(self, s_seq: torch.Tensor, shape) -> torch.Tensor:
        return s_seq


class BooleanSpikeCompressor(BaseSpikeCompressor):
    requires_strictly_binary = True

    def __init__(self):
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
