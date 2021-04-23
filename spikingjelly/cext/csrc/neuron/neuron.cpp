#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include "neuron_def.h"

//LIF hard reset----------------------------------------------------
std::vector<at::Tensor> LIF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> LIF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

//IF hard reset----------------------------------------------------
std::vector<at::Tensor> IF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset);

std::vector<at::Tensor> IF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index);


//LIF hard reset fptt----------------------------------------------------
std::vector<at::Tensor> LIF_hard_reset_fptt(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & reciprocal_tau, const bool & detach_x);
    
std::vector<at::Tensor> LIF_hard_reset_fptt_with_grad(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

//IF hard reset fptt----------------------------------------------------
std::vector<at::Tensor> IF_hard_reset_fptt(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset);

std::vector<at::Tensor> IF_hard_reset_fptt_with_grad(torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index);

//LIF bp----------------------------------------------------
std::vector<at::Tensor> LIF_backward(
    torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h,
    const float & reciprocal_tau, const bool & detach_x);

//IF bp----------------------------------------------------
std::vector<at::Tensor> IF_backward(
    torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h);

//LIF bptt----------------------------------------------------
std::vector<at::Tensor> LIF_bptt(
    torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
    torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h,
    const float & reciprocal_tau, const bool & detach_x);

//IF bptt----------------------------------------------------
std::vector<at::Tensor> IF_bptt(
    torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
    torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h);

//PLIF bptt----------------------------------------------------
std::vector<at::Tensor> ParametricLIF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_hard_reset_fptt_with_grad(
    torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset,
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_backward(
    torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_bptt(
    torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
    torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
    const float & reciprocal_tau, const bool & detach_x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
    m.def("LIF_hard_reset_forward_with_grad", &LIF_hard_reset_forward_with_grad);
    m.def("LIF_hard_reset_fptt", &LIF_hard_reset_fptt);
    m.def("LIF_hard_reset_fptt_with_grad", &LIF_hard_reset_fptt_with_grad);
    m.def("LIF_backward", &LIF_backward);
    m.def("LIF_bptt", &LIF_bptt);
    m.def("IF_hard_reset_forward", &IF_hard_reset_forward);
    m.def("IF_hard_reset_forward_with_grad", &IF_hard_reset_forward_with_grad);
    m.def("IF_hard_reset_fptt", &IF_hard_reset_fptt);
    m.def("IF_hard_reset_fptt_with_grad", &IF_hard_reset_fptt_with_grad);
    m.def("IF_backward", &IF_backward);
    m.def("IF_bptt", &IF_bptt);
    m.def("ParametricLIF_hard_reset_forward_with_grad", &ParametricLIF_hard_reset_forward_with_grad);
    m.def("ParametricLIF_backward", &ParametricLIF_backward);
    m.def("ParametricLIF_hard_reset_fptt_with_grad", &ParametricLIF_hard_reset_fptt_with_grad);
    m.def("ParametricLIF_bptt", &ParametricLIF_bptt);
}


