import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from _C_surrogate import surrogate as cext_surrogate 

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
