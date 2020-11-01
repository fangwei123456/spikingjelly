import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


def get_coord_feat(x: torch.Tensor):
    # x.shape=[N, C, d1, d2, ..., dn]
    # 返回的coords.shape=[M, n], feats.shape=[M, C]
    # M 是非全0特征的数量
    '''
    .. code-block:: python

        spike = torch.rand([2, 3, 2]) > 0.9
        print(spike)
        coords, feats = get_coord_feat(spike)
        print(coords, feats)
    '''
    # x.shape = [N, C, ...]
    assert x.dim() >= 2, 'error: x.dim() < 2'

    with torch.no_grad():
        # xs是x在通道这一维度上的累加，通道这一维度被视作是特征所在的维度，例如对于彩色图像，其特征有3维
        if x.dtype == torch.bool:
            xs = x.float().sum(dim=1)
        else:
            xs = x.abs().sum(dim=1)
        mask = xs != 0  # 记录不为0的元素的位置

    coords = []
    for i in range(xs.dim()):
        index_shape = [1] * xs.dim()
        index_shape[i] = xs.shape[i]

        index_tensor = torch.arange(0, xs.shape[i], device=xs.device).view(index_shape)

        repeat_shape = list(xs.shape)
        repeat_shape[i] = 1
        coords.append(index_tensor.repeat(repeat_shape)[mask].unsqueeze_(1))

    # 将通道移动到最后一维
    axis_order = list(range(x.dim()))
    axis_order.pop(1)
    axis_order.append(1)
    xp = x.permute(axis_order)

    coords = torch.cat(coords, dim=1).int()

    feats = xp[mask]
    return coords, feats

def to_sparse(x: torch.Tensor):
    coords, feats = get_coord_feat(x)
    return ME.SparseTensor(coords=coords, feats=feats)

class ToSparse(nn.Module):
    '''
    .. code-block:: python

        spike = torch.rand([2, 3, 4, 5]) > 0.5
        print('spike\n', spike)

        net = nn.Sequential(
            ToSparse(),
            ToDense([4, 5])
        )

        x_d = net(spike)
        print('dense\n', x_d.shape)
        print(x_d)
    '''
    def forward(self, x: torch.Tensor):
        return to_sparse(x)


class ToDense(nn.Module):
    def __init__(self, coord_shape: torch.Size or list or tuple):
        '''
        .. code-block:: python

            spike = torch.rand([2, 3, 4, 5]) > 0.5
            print('spike\n', spike)

            net = nn.Sequential(
                ToSparse(),
                ToDense([4, 5])
            )

            x_d = net(spike)
            print('dense\n', x_d.shape)
            print(x_d)
        '''
        super().__init__()
        self.max_coords = []
        self.min_coords = []
        for i in coord_shape:
            self.max_coords.append(i - 1)
            self.min_coords.append(0)
        self.max_coords = torch.IntTensor(self.max_coords)
        self.min_coords = torch.IntTensor(self.min_coords)

    def forward(self, x: ME.SparseTensor):
        return x.dense(self.min_coords, self.max_coords)[0]


class SparseMaxPool3d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=3)


class SparseMaxPool2d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=2)


class SparseMaxPool1d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=1)


class SparseSumPool3d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=3)


class SparseSumPool2d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=2)


class SparseSumPool1d(ME.MinkowskiSumPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=1)


class SparseConv3d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=3)


class SparseConv2d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        '''
        .. code-block:: python

            spike = (torch.rand([2, 3, 8, 8]) > 0.9).float()
            print('spike\n', spike)

            net = nn.Sequential(
                ToSparse(),
                SparseConv2d(3, 64, 3),
                ToDense([8, 8])
            )

            y = net(spike)
            print(y.shape)
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=2)


class SparseConv1d(ME.MinkowskiConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, has_bias=bias, dimension=1)
