import torch

try:
    import triton
    import triton.language as tl
except (AttributeError, RuntimeError) as error:
    raise ImportError(f"Optional Triton initialization failed: {error}") from error


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
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    N = tl.cast(N, tl.int64)
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
            offset = region.to(tl.int64) * K + k
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


def _forward_impl(
    gate,
    theta,
    h,
    d,
    return_sequence,
    other_theta=None,
    other_h=None,
    other_d=None,
    gates=None,
    split_threshold=0.0,
    clamp_value=None,
    *,
    _kernel_wrapper=None,
):
    k = theta.shape[-1]
    output = torch.empty(
        (k, *gate.shape) if return_sequence else gate.shape,
        dtype=gate.dtype,
        device=gate.device,
    )
    outlier = other_theta is not None
    if not outlier:
        other_theta, other_h, other_d = theta, h, d
    kernel = (
        _few_spike_inference
        if _kernel_wrapper is None
        else _kernel_wrapper(_few_spike_inference)
    )
    with torch.cuda.device(gate.device):
        kernel[(triton.cdiv(gate.numel(), 256),)](
            gate,
            theta,
            h,
            d,
            other_theta,
            other_h,
            other_d,
            gates if gates is not None else theta,  # Unused outside HG (ROUTE 2).
            output,
            gate.numel(),
            k,
            2 if gates is not None else (1 if outlier else 0),
            theta.shape[0] if gates is not None else 1,
            split_threshold,
            clamp_value,
            return_sequence,
            256,
            enable_fp_fusion=False,
        )
    return output
