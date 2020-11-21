#include <iostream>
#include <torch/extension.h>
#include <math.h>
void alpha_atan_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size);
void alpha_sigmoid_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size);

torch::Tensor alpha_backward_base(torch::Tensor & grad_output, torch::Tensor & x, const torch::Tensor & alpha, 
    void alpha_backward_cuda(const float* , const float* , const float & , float* , const int & )
)
{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
    if (! x.is_contiguous())
    {
        x = x.contiguous();
    }
    if (! grad_output.is_contiguous())
    {
        grad_output = grad_output.contiguous();
    }
    auto grad_x = torch::zeros_like(x.data());
    alpha_backward_cuda(grad_output.data_ptr<float>(), x.data_ptr<float>(), alpha.item<float>(), grad_x.data_ptr<float>(), x.numel());
    return grad_x;   
}


torch::Tensor alpha_sigmoid_backward(torch::Tensor & grad_output, torch::Tensor & x, const torch::Tensor & alpha)
{
    if (x.get_device() < 0)
    {
        // CPU
        return alpha * torch::sigmoid_backward(grad_output, torch::sigmoid(x * alpha));
    }
    else
    {
        // GPU
        return alpha_backward_base(grad_output, x, alpha, alpha_sigmoid_backward_cuda);
    }}


torch::Tensor alpha_atan_backward(torch::Tensor & grad_output, torch::Tensor & x, const torch::Tensor & alpha)
{   
    if (x.get_device() < 0)
    {
        // CPU
        return alpha / 2.0f / (1.0f + (M_PI_2 * alpha * x).pow_(2)) * grad_output;
    }
    else
    {
        // GPU
        return alpha_backward_base(grad_output, x, alpha, alpha_atan_backward_cuda);
    }
}
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_backward", &torch::sigmoid_backward);
    m.def("alpha_sigmoid_backward", &alpha_sigmoid_backward);
    m.def("alpha_atan_backward", &alpha_atan_backward);
}