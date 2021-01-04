import torch
import torch.nn as nn
import torch.nn.functional as F
import _C_gemm

class sparse_mm_dense_atf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse: torch.Tensor, dense: torch.Tensor):
        # sparse: [M, N]  dense: [N, P]  y:[M, P]
        if sparse.requires_grad or dense.requires_grad:
            ctx.save_for_backward(sparse, dense)
        y = torch.zeros(size=[sparse.shape[0], dense.shape[1]], dtype=torch.float, device=sparse.device)
        _C_gemm.sparse_mm_dense_cusparse(sparse, dense, y)
        # y = torch.mm(sparse, dense)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [M, P]
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = grad_output.mm(dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.zeros_like(dense.data)
            _C_gemm.sparse_mm_dense_cusparse(sparse.t(), grad_output, grad_dense)
            # grad_dense = sparse.t().mm(grad_output)
        return grad_sparse, grad_dense


def sparse_mm_dense(sparse: torch.Tensor, dense: torch.Tensor):
    '''
    * :ref:`API in English <sparse_mm_dense-en>`

    .. _sparse_mm_dense-cn:

    :param sparse: 稀疏2D tensor
    :type sparse: torch.Tensor
    :param dense: 稠密2D tensor
    :type dense: torch.Tensor
    :return: sparse 和 dense 的矩阵乘
    :rtype: torch.Tensor

    对输入的稀疏的二维矩阵 ``sparse`` 和稠密的二维矩阵 ``dense`` 进行矩阵乘法。

    .. warning::

        代码内部的实现方式是，首先将 ``sparse`` 转换为稀疏矩阵格式，然后再调用相关库进行运算。如果 ``sparse`` 不够稀疏，则该函数的速度会比普通矩阵乘法 ``torch.mm`` 慢很多。


    .. warning::

        稀疏矩阵的乘法存在一定的计算误差，但误差并不显著，或可忽略。

    .. warning::

        本函数不支持CPU。

    * :ref:`中文API <sparse_mm_dense-cn>`

    .. _sparse_mm_dense-en:

    :param sparse: a 2D sparse tensor
    :type sparse: torch.Tensor
    :param dense: a 2D dense tensor
    :type dense: torch.Tensor
    :return: a matrix multiplication of the matrices ``dense`` and ``sparse``
    :rtype: torch.Tensor

    Performs a matrix multiplication of the matrices ``dense`` and ``sparse``.

    .. admonition:: Warning
        :class: warning

        This function is implemented by converting ``sparse`` to a sparse format and doing a sparse matrix multiplication. If the sparsity of ``sparse`` is not high enough, the speed of this function will be slower than ``torch.mm``.


    .. admonition:: Warning
        :class: warning

        There are some numeral errors when doing the sparse matrix multiplication. But the errors are not significant.

    .. admonition:: Warning
        :class: warning

        This function does not support to run on cpu.
    '''
    return sparse_mm_dense_atf.apply(sparse, dense)




