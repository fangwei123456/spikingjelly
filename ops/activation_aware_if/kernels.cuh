#include "../_cuda.cuh"
#include "../native_layout.cuh"
struct SjChannelLayout {
    int64_t sizes[64]{};
    int64_t strides[64]{};
    int rank = 0;
    explicit SjChannelLayout(const at::Tensor &tensor) {
        for (int d = tensor.dim() - 1; d >= 0; --d) {
            if (tensor.size(d) > 1) {
                sizes[rank] = tensor.size(d);
                strides[rank++] = tensor.stride(d);
            }
        }
    }
    __device__ int64_t operator()(int64_t index) const {
        int64_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            offset += (index % sizes[d]) * strides[d];
            index /= sizes[d];
        }
        return offset;
    }
};

struct SjContiguousChannelLayout {
    __device__ int64_t operator()(int64_t index) const { return index; }
};

template <typename scalar_t, class Layout, class ChannelLayout>
__global__ void activation_aware_if_forward(
    scalar_t const *x, float const *v, float const *threshold, float const *offset,
    scalar_t *out, float *vo, long long T, long long N, long long channels,
    long long inner, bool scalar_threshold, bool scalar_offset, float reset, bool soft,
    bool trace, Layout layout, ChannelLayout threshold_layout,
    ChannelLayout offset_layout) {
    for (long long n = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         n < N; n += static_cast<long long>(blockDim.x) * gridDim.x) {
        const auto offsets = layout.spatial(n, N);
        const long long c = offsets[4] / inner % channels;
        const float th = threshold[scalar_threshold ? 0 : threshold_layout(c)],
                    off = offset[scalar_offset ? 0 : offset_layout(c)];
        float voltage = v[offsets[1]];
        for (long long t = 0; t < T; ++t) {
            const long long i = t * N + n;
            const float h = voltage + float(x[layout.template index<0>(i, t, offsets)]);
            const float spike = float(h + off >= th);
            voltage = soft ? h - spike * th : spike * reset + (1.f - spike) * h;
            out[layout.template index<2>(i, t, offsets)] = scalar_t(spike);
            if (trace)
                vo[layout.template index<3>(i, t, offsets)] = voltage;
        }
        if (!trace)
            vo[offsets[3]] = voltage;
    }
}
