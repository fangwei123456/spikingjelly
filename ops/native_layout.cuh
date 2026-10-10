#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <algorithm>
#include <array>
#include <c10/cuda/CUDAException.h>
#include <cstring>
#include <vector>

// Native kernels fully overwrite these outputs; avoid redispatching allocation.
inline at::Tensor sj_empty_like(const at::Tensor &x, at::TensorOptions options = {}) {
    if (x.is_non_overlapping_and_dense())
        return at::Tensor(at::detail::empty_strided_cuda(
            x.sizes(), x.strides(), x.options().merge_in(options)));
    return at::empty_like(x, options);
}

inline at::Tensor sj_empty_state_like(const at::Tensor &sequence) {
    if (sequence.is_non_overlapping_and_dense() &&
        (sequence.size(0) == 1 ||
         sequence.stride(0) == sequence.numel() / sequence.size(0)))
        return at::Tensor(at::detail::empty_strided_cuda(
            sequence.sizes().slice(1), sequence.strides().slice(1), sequence.options()));
    return sj_empty_like(sequence[0]);
}

struct SjLinearOffsets {
    int64_t n;
    __device__ int64_t operator[](int) const { return n; }
};

struct SjLinearLayout {
    using Offsets = SjLinearOffsets;
    __device__ Offsets spatial(int64_t n, int64_t) const { return {n}; }
    template <int Slot>
    __device__ int64_t index(int64_t linear, int64_t, const Offsets &) const {
        return linear;
    }
};

template <int Buffers> struct SjOffsets {
    int64_t values[Buffers + 1];
    __device__ int64_t operator[](int slot) const { return values[slot]; }
};

template <int Buffers, int Dims> struct SjStridedLayout {
    using Offsets = SjOffsets<Buffers>;
    int64_t sizes[Dims]{};
    int64_t strides[Buffers + 1][Dims]{};
    int64_t time_strides[Buffers]{};
    int rank{};
    int spatial_tile_shift{};
    bool linear[Buffers + 1]{};

    template <class Index> __device__ Offsets spatial_impl(Index n) const {
        if (spatial_tile_shift) {
            // Split each warp 8x4 to coalesce buffers with different spatial orders.
            const Index c = Index(1) << spatial_tile_shift;
            n = (n & ~(c * 4 - 1)) + (n & 7) + ((n >> 3) & 3) * c +
                ((n >> 5) & (c / 8 - 1)) * 8;
        }
        Offsets out{};
#pragma unroll
        for (int b = 0; b <= Buffers; ++b)
            out.values[b] = linear[b] ? n : 0;
        for (int d = 0; d < rank; ++d) {
            const Index size = Index(sizes[d]);
            Index coordinate;
            if ((size & (size - 1)) == 0) {
                coordinate = n & (size - 1);
                n >>= __ffsll(static_cast<long long>(size)) - 1;
            } else {
                coordinate = n % size;
                n /= size;
            }
#pragma unroll
            for (int b = 0; b <= Buffers; ++b)
                if (!linear[b])
                    out.values[b] += int64_t(coordinate) * strides[b][d];
        }
        return out;
    }
    __device__ Offsets spatial(int64_t n, int64_t neurons) const {
        return neurons <= UINT32_MAX ? spatial_impl(uint32_t(n))
                                     : spatial_impl(uint64_t(n));
    }
    template <int Slot>
    __device__ int64_t index(int64_t, int64_t t, const Offsets &offsets) const {
        return offsets[Slot] + t * time_strides[Slot];
    }
};

template <int Buffers> struct SjDeviceLayout {
    using Offsets = SjOffsets<Buffers>;
    const SjStridedLayout<Buffers, 64> *data;
    __device__ Offsets spatial(int64_t n, int64_t neurons) const {
        return data->spatial(n, neurons);
    }
    template <int Slot>
    __device__ int64_t index(int64_t linear, int64_t t, const Offsets &offsets) const {
        return data->template index<Slot>(linear, t, offsets);
    }
};

struct SjLayoutChunk {
    unsigned char bytes[1024];
    int count;
};

static __global__ void sj_write_layout(unsigned char *destination, int offset,
                                       SjLayoutChunk chunk) {
    for (int i = threadIdx.x; i < chunk.count; i += blockDim.x)
        destination[offset + i] = chunk.bytes[i];
}

template <int Buffers, class Launch>
C10_NOINLINE void sj_launch_strided_layout(
    const at::Tensor &reference,
    const std::array<const at::Tensor *, Buffers> &tensors, Launch launch) {
    std::vector<int> order;
    for (int d = 1; d < reference.dim(); ++d)
        if (reference.size(d) > 1)
            order.push_back(d);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return reference.stride(a) < reference.stride(b);
    });
    // An int64-sized nonempty tensor has at most 62 non-unit dimensions.
    SjStridedLayout<Buffers, 64> layout;
    for (int b = 0; b < Buffers; ++b)
        layout.time_strides[b] =
            tensors[b]->dim() == reference.dim() ? tensors[b]->stride(0) : 0;
    for (int d : order) {
        int64_t current[Buffers + 1];
        for (int b = 0; b < Buffers; ++b)
            current[b] = tensors[b]->stride(d - (tensors[b]->dim() != reference.dim()));
        current[Buffers] = 1;
        for (int j = d + 1; j < reference.dim(); ++j)
            current[Buffers] *= reference.size(j);
        bool merge = layout.rank > 0;
        for (int b = 0; b <= Buffers && merge; ++b)
            merge = current[b] ==
                    layout.strides[b][layout.rank - 1] * layout.sizes[layout.rank - 1];
        if (merge) {
            layout.sizes[layout.rank - 1] *= reference.size(d);
        } else {
            layout.sizes[layout.rank] = reference.size(d);
            for (int b = 0; b <= Buffers; ++b)
                layout.strides[b][layout.rank] = current[b];
            ++layout.rank;
        }
    }
    for (int b = 0; b <= Buffers; ++b) {
        int64_t span = 1;
        layout.linear[b] = true;
        for (int d = 0; d < layout.rank; ++d) {
            layout.linear[b] &= layout.strides[b][d] == span;
            span *= layout.sizes[d];
        }
    }
    if (layout.rank > 1 && layout.sizes[0] >= 16 && layout.sizes[0] <= 64 &&
        (layout.sizes[0] & (layout.sizes[0] - 1)) == 0 &&
        (reference.numel() / reference.size(0)) % (layout.sizes[0] * 4) == 0) {
        for (int b = 0; b < Buffers && !layout.spatial_tile_shift; ++b)
            for (int d = 1; d < layout.rank; ++d)
                if (layout.strides[b][d] > 0 &&
                    layout.strides[b][d] < layout.strides[b][0]) {
                    for (int64_t c = layout.sizes[0]; c > 1; c /= 2)
                        ++layout.spatial_tile_shift;
                    break;
                }
    }
    // Inline metadata fits the CUDA 11 kernel-argument limit, including 8 buffers.
    if (layout.rank <= 32) {
        SjStridedLayout<Buffers, 32> compact;
        compact.rank = layout.rank;
        compact.spatial_tile_shift = layout.spatial_tile_shift;
        for (int d = 0; d < layout.rank; ++d)
            compact.sizes[d] = layout.sizes[d];
        for (int b = 0; b <= Buffers; ++b) {
            compact.linear[b] = layout.linear[b];
            for (int d = 0; d < layout.rank; ++d)
                compact.strides[b][d] = layout.strides[b][d];
        }
        for (int b = 0; b < Buffers; ++b)
            compact.time_strides[b] = layout.time_strides[b];
        launch(compact);
    } else {
        // Initialize on the captured stream: no host buffer must outlive this call.
        auto storage =
            at::empty({int64_t(sizeof(layout))}, reference.options().dtype(at::kByte));
        auto stream = at::cuda::getCurrentCUDAStream(reference.get_device());
        for (size_t offset = 0; offset < sizeof(layout); offset += 1024) {
            SjLayoutChunk chunk{};
            chunk.count = std::min<size_t>(1024, sizeof(layout) - offset);
            std::memcpy(chunk.bytes, reinterpret_cast<const char *>(&layout) + offset,
                        chunk.count);
            sj_write_layout<<<1, 256, 0, stream>>>(storage.data_ptr<unsigned char>(),
                                                   offset, chunk);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        launch(SjDeviceLayout<Buffers>{
            reinterpret_cast<const SjStridedLayout<Buffers, 64> *>(
                storage.data_ptr())});
    }
}

template <int Buffers, class Launch>
inline void sj_launch_layout(const at::Tensor &reference,
                            const std::array<const at::Tensor *, Buffers> &tensors,
                            Launch launch) {
    if (std::all_of(tensors.begin(), tensors.end(),
                   [](const at::Tensor *x) { return x->is_contiguous(); })) {
        launch(SjLinearLayout{});
    } else {
        sj_launch_strided_layout<Buffers>(reference, tensors, launch);
    }
}
