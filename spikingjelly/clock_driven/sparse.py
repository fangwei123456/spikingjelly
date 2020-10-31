import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

def to_sparse(x: torch.Tensor):
    # x.shape=[B, d1, d2, ..., dn]
    # 返回的coords.shape=[M, n], feats.shape=[M, 1]
    # M 是非0元素的数量
    '''
    .. code-block:: python

        import MinkowskiEngine as ME
        spike = torch.rand([2, 3, 2]) > 0.5
        print(spike.int())
        coords, feats = to_sparse(spike)
        x = ME.SparseTensor(coords=coords, feats=feats)
        print(x)
        print(x.dense(min_coords=torch.IntTensor([0, 0]), max_coords=torch.IntTensor([2, 1])))
    '''
    assert x.dim() >= 3, 'must ensure x.dim() >= 3'

    if x.dtype == torch.bool:
        mask = x
    else:
        mask = x != 0
    coords = []

    for i in range(x.dim()):
        index_shape = [1] * x.dim()
        index_shape[i] = x.shape[i]

        index_tensor = torch.arange(0, x.shape[i], device=x.device).view(index_shape)

        repeat_shape = list(x.shape)
        repeat_shape[i] = 1
        coords.append(index_tensor.repeat(repeat_shape)[mask].unsqueeze_(0))
    # 若x为 tensor([[ True, False,  True],
    #                  [ True, False, False]])
    # 运行到此处，coords为 [tensor([[0, 0, 1]]), tensor([[0, 2, 0]])]

    coords = torch.cat(coords).t().int()
    if x.dtype == torch.bool:
        feats = torch.ones(size=[coords.shape[0], 1], dtype=torch.float, device=x.device)
    else:
        feats = x[mask].unsqueeze_(1)
    return coords, feats

class ToSparse(nn.Module):
    def forward(self, x: torch.Tensor):
        coords, feats = to_sparse(x)
        return ME.SparseTensor(coords=coords, feats=feats)

class SparseMaxPool3d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=4)

class SparseMaxPool2d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=3)

class SparseMaxPool1d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=2)


class SparseSumPool3d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=4)

class SparseSumPool2d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=3)

class SparseSumPool1d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=2)


class SparseConv3d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=4)

class SparseConv2d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=3)

class SparseConv1d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=2)
