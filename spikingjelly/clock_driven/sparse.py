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
        coords.append(index_tensor.repeat(repeat_shape)[mask].unsqueeze_(1))

    coords = torch.cat(coords, dim=1).int()
    if x.dtype == torch.bool:
        feats = torch.ones(size=[coords.shape[0], 1], dtype=torch.float, device=x.device)
    else:
        feats = x[mask].unsqueeze_(1)
    return coords, feats


class ToSparse(nn.Module):
    '''
    .. code-block:: python

        spike = torch.rand([2, 3, 2, 2]) > 0.5
        print('spike\n', spike.int())

        net = nn.Sequential(
            ToSparse(),
            ToDense([3, 2, 2])
        )

        x_d = net(spike)
        print('dense\n', x_d.shape)
        print(x_d)
    '''
    def forward(self, x: torch.Tensor):
        coords, feats = to_sparse(x)
        return ME.SparseTensor(coords=coords, feats=feats)


class ToDense(nn.Module):
    def __init__(self, dense_shape: torch.Size or list or tuple):
        '''
        .. code-block:: python

            spike = torch.rand([2, 3, 2, 2]) > 0.5
            print('spike\n', spike.int())

            net = nn.Sequential(
                ToSparse(),
                ToDense([3, 2, 2])
            )

            x_d = net(spike)
            print('dense\n', x_d.shape)
            print(x_d)
        '''
        super().__init__()
        self.max_coords = []
        self.min_coords = []
        for i in dense_shape:
            self.max_coords.append(i - 1)
            self.min_coords.append(0)
        self.max_coords = torch.IntTensor(self.max_coords)
        self.min_coords = torch.IntTensor(self.min_coords)

    def forward(self, x: ME.SparseTensor):
        return x.dense(self.min_coords, self.max_coords)[0].squeeze_(1)


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
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=4)


class SparseConv2d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=3)


class SparseConv1d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=2)
