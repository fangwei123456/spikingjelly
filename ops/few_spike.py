"""Private FP32 inference fusion for the existing few-spike table decoders."""

import torch

from spikingjelly.logger import logger

try:
    import triton
    import triton.language as tl
except (ImportError, OSError) as error:
    triton = None
    logger.info("Few-spike Triton kernel unavailable: {}", error)


if triton is not None:

    @triton.jit
    def _few_spike_inference(
        X,
        THETA,
        H,
        D,
        OTHER_THETA,
        OTHER_H,
        OTHER_D,
        GATES,
        OUT,
        N,
        K: tl.constexpr,
        ROUTE: tl.constexpr,
        REGIONS: tl.constexpr,
        SPLIT: tl.constexpr,
        CLAMP: tl.constexpr,
        SEQUENCE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < N
        v = tl.load(X + index, valid, 0)
        if ROUTE == 1:
            if CLAMP is not None:
                v = tl.where(v < -CLAMP, -CLAMP, tl.where(v > CLAMP, CLAMP, v))
            sign = tl.where(v > 0, 1.0, tl.where(v < 0, -1.0, 0.0))
            v = tl.abs(v)
            normal = v <= SPLIT
        if ROUTE == 2:
            region = tl.full((BLOCK,), 0, tl.int32)
            for boundary in range(REGIONS - 1):
                region += (v > tl.load(GATES + boundary)).to(tl.int32)
            region = tl.where(v != v, REGIONS - 1, region)
        readout = tl.full((BLOCK,), 0, tl.float32)
        for k in range(K):
            if ROUTE == 2:
                offset = region * K + k
                theta = tl.load(THETA + offset, valid, 0)
                h = tl.load(H + offset, valid, 0)
                d = tl.load(D + offset, valid, 0)
            else:
                theta, h, d = tl.load(THETA + k), tl.load(H + k), tl.load(D + k)
                if ROUTE == 1:
                    theta = tl.where(normal, theta, tl.load(OTHER_THETA + k))
                    h = tl.where(normal, h, tl.load(OTHER_H + k))
                    d = tl.where(normal, d, tl.load(OTHER_D + k))
            spike = ((v - theta) >= 0).to(tl.float32)
            value = d * spike
            if SEQUENCE:
                if ROUTE == 1:
                    value = value * sign
                tl.store(OUT + k * N + index, value, valid)
            else:
                readout = readout + value
            v = v - h * spike
        if not SEQUENCE:
            if ROUTE == 1:
                readout = readout * sign
            tl.store(OUT + index, readout, valid)


def _forward(
    gate,
    theta,
    h,
    d,
    return_sequence,
    *,
    outlier=None,
    gates=None,
    split_threshold=0.0,
    clamp_value=None,
):
    if triton is None:
        return None
    k = theta.shape[-1]
    output = torch.empty(
        (k, *gate.shape) if return_sequence else gate.shape,
        dtype=gate.dtype,
        device=gate.device,
    )
    other_theta, other_h, other_d = outlier if outlier is not None else (theta, h, d)
    with torch.cuda.device(gate.device):
        _few_spike_inference[(triton.cdiv(gate.numel(), 256),)](
            gate,
            theta,
            h,
            d,
            other_theta,
            other_h,
            other_d,
            gates if gates is not None else theta,
            output,
            gate.numel(),
            k,
            2 if gates is not None else (1 if outlier is not None else 0),
            theta.shape[0] if gates is not None else 1,
            split_threshold,
            clamp_value,
            return_sequence,
            256,
            enable_fp_fusion=False,
        )
    return output
