import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F


class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, fc=False):
        super(TimeAttention, self).__init__()
        if fc ==True:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

            self.sharedMLP = nn.Sequential(
                nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False),
            )
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

            self.sharedMLP = nn.Sequential(
                nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "f b c h w -> b c f h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c f h w -> f b c h w")
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "f b c h w -> b (f c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = rearrange(x, "b f c h w -> f b c h w")

        return self.sigmoid(x)


class TCSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, c_ratio=16, t_ratio=16):
        super(TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(timeWindows, t_ratio)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TCA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, c_ratio=16, t_ratio=16):
        super(TCA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(timeWindows, t_ratio)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out


class CSA(nn.Module):
    def __init__(self, channels, stride=1, c_ratio=16):
        super(CSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, t_ratio=16):
        super(TSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows, t_ratio)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, t_ratio=16, fc=False):
        super(TA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows, t_ratio, fc)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out


class CA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, c_ratio=16):
        super(CA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        # self.ta = TimeAttention(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out


class SA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1):
        super(SA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(channels)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(x) * x  # 广播机制

        out = self.relu(out)
        return out