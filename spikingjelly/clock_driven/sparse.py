import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

def to_sparse(x: torch.Tensor):
    '''
    .. code-block:: python

        import MinkowskiEngine as ME
        spike = torch.rand([2, 3, 2]) > 0.5
        print(spike.int())
        coords, feats = to_sparse(spike)
        x = ME.SparseTensor(coords=coords, feats=feats)
        print(x)
        print(x.dense())
    '''
    assert x.dim() >= 2, 'x must be a tensor with more than 1 dim'

    if x.dtype == torch.bool:
        mask = x
    else:
        mask = x > 0
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
        feats = x[mask]
    return coords, feats

class SparseConverter:
    @ staticmethod
    def str_tensor_shape(x: torch.Tensor):
        # 将x的尺寸转换为字符串，例如'2-3'
        shape_list = list(x.shape)
        for i in range(shape_list.__len__()):
            shape_list[i] = str(shape_list[i])
        return '-'.join(shape_list)

    def __init__(self):
        self.index_buffer = {}  # 保存生成索引的tensor的缓冲区

    def to_sparse(self, x: torch.Tensor):
        assert x.dim() >= 2, 'x must be a tensor with more than 1 dim'
        if x.dtype == torch.bool:
            mask = x
        else:
            mask = x > 0

        str_shape = SparseConverter.str_tensor_shape(x)

        coords = []
        for i in range(x.dim()):
            # 生成脉冲数据的第i维的索引
            str_shape_i = str_shape + '_' + str(i)

            if str_shape_i not in self.index_buffer:
                index_shape = [1] * x.dim()
                index_shape[i] = x.shape[i]

                index_tensor = torch.arange(0, x.shape[i], device=x.device).view(index_shape)

                repeat_shape = list(x.shape)
                repeat_shape[i] = 1
                self.index_buffer[str_shape_i] = index_tensor.repeat(repeat_shape)

            coords.append(self.index_buffer[str_shape_i][mask].unsqueeze_(0))

            # 若x为 tensor([[ True, False,  True],
            #                  [ True, False, False]])
            # 运行到此处，coords为 [tensor([[0, 0, 1]]), tensor([[0, 2, 0]])]

        coords = torch.cat(coords).t().int()
        coords = torch.cat(coords).t().int()
        if x.dtype == torch.bool:
            feats = torch.ones(size=[coords.shape[0], 1], dtype=torch.float, device=x.device)
        else:
            feats = x[mask]
        return coords, feats

class SparseMaxPool3d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=4)

class SparseMaxPool2d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=3)

class SparseMaxPool1d(ME.MinkowskiMaxPooling):
    def __init__(self, kernel_size, stride=1, dilation=1):
        super().__init__(kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=2)

