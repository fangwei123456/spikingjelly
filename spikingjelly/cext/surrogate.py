import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
import math
use_fast_math = True
extra_cuda_cflags = []
if use_fast_math:
    extra_cuda_cflags.append('-use_fast_math')
cext_surrogate = cpp_extension.load(name='surrogate', sources=['./spikingjelly/cext/csrc/surrogate/surrogate.cpp', './spikingjelly/cext/csrc/surrogate/surrogate.cu'],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return x.ge(0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = cext_surrogate.alpha_sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1])
        return grad_x, None

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return x.ge(0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = cext_surrogate.alpha_atan_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1])
        return grad_x, None
