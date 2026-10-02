"""Allocation and iteration order for strided point-neuron tensors."""

import torch


def _empty_like(x, *, dtype=None, sequence=True, shape=None, align_time_slice=False):
    dense = x.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense.default(x)
    if shape is None and dense and not align_time_slice:
        return torch.empty_like(x, dtype=dtype)
    output_shape = x.shape if shape is None else shape
    order = sorted(range(x.ndim), key=x.stride().__getitem__)
    if sequence and not dense:
        order = [d for d in order if d != 0] + [0]
    strides = [0] * x.ndim
    size = 1
    for d in order:
        strides[d] = size
        size *= max(output_shape[d], 1)
    if align_time_slice:
        # CuPy returns result[1:]; align that view for Inductor's external-op ABI.
        element_size = (x.dtype if dtype is None else dtype).itemsize
        prefix = -strides[0] % (16 // element_size)
        if prefix:
            return x.new_empty(size + prefix, dtype=dtype).as_strided(
                output_shape, strides, prefix
            )
    return torch.empty_strided(
        output_shape,
        strides,
        dtype=x.dtype if dtype is None else dtype,
        device=x.device,
    )


def _layout_args(reference, *tensors, sequence=True):
    # Triton metadata are constexpr: specialize nested symbolic strides too.
    reference_strides = tuple(int(s) for s in reference.stride())
    order = sorted(
        range(int(sequence), reference.ndim), key=reference_strides.__getitem__
    )
    shape = reference.shape
    sizes = tuple(int(shape[d]) for d in order)
    layouts = []
    for x in tensors:
        strides = tuple(int(s) for s in x.stride())
        if sequence and x.ndim != reference.ndim:
            strides = (0,) + strides
        layouts.append(
            (strides[0] if sequence else 0,) + tuple(strides[d] for d in order)
        )
    return sizes, tuple(layouts)
