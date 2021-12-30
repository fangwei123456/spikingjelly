#include <torch/extension.h>
#include <vector>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("cudnn_convolution", py::overload_cast<const at::Tensor&, const at::Tensor&,
    // at::IntArrayRef, at::IntArrayRef, at::IntArrayRef,
    // int64_t, bool, bool, bool>(&at::cudnn_convolution));
    // /*
    // aten/src/ATen/native/cudnn/ConvPlaceholders.cpp

    // at::Tensor cudnn_convolution(
    //     const at::Tensor& input, const at::Tensor& weight,
    //     IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    //     int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)

    // There are two overloaded C++ methods `cudnn_convolution`. So, we need to use an alternative syntax to cast the overloaded function.
    // Refer to https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods and https://github.com/pytorch/pytorch/issues/39518 for more details.
    // */
    
    m.def("cudnn_convolution_backward", &at::cudnn_convolution_backward);
    /*
    aten/src/ATen/native/cudnn/ConvShared.cpp

    Tensor cudnn_convolution_forward(
        CheckedFrom c,
        const TensorArg& input, const TensorArg& weight,
        IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
        bool benchmark, bool deterministic, bool allow_tf32)
    */

    m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input);
   /*
   aten/src/ATen/native/cudnn/ConvShared.cpp

   at::Tensor cudnn_convolution_backward_input(
        IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
        IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
        bool benchmark, bool deterministic, bool allow_tf32)
   */

    m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight);
    /*
   aten/src/ATen/native/cudnn/ConvShared.cpp

   at::Tensor cudnn_convolution_backward_weight(
        IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
        IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
        bool benchmark, bool deterministic, bool allow_tf32)
    */
}

