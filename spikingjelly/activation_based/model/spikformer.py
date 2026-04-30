import torch
import torch.nn as nn

from .. import base, functional, layer, neuron
from ..layer.attention import SpikingSelfAttention

__all__ = [
    "Spikformer",
    "SpikformerBlock",
    "SpikformerConv2dBN",
    "SpikformerConv2dBNLIF",
    "SpikformerMLP",
    "SpikformerPatchStem",
    "spikformer_s",
    "spikformer_ti",
]


class SpikformerConv2dBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block = layer.SeqToANNContainer(*layers)

    def forward(self, x_seq: torch.Tensor):
        return self.block(x_seq)

    def __spatial_split__(self):
        return tuple(self.block.children())


class SpikformerConv2dBNLIF(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.conv_bn = SpikformerConv2dBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool=pool,
        )
        self.neuron = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    def forward(self, x_seq: torch.Tensor):
        return self.neuron(self.conv_bn(x_seq))

    def __spatial_split__(self):
        return self.conv_bn, self.neuron


class SpikformerPatchStem(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 256,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        if patch_size != 16:
            raise ValueError(
                "The current SpikformerPatchStem uses a fixed 4-stage /16 patch pipeline; "
                f"expected patch_size=16, but got {patch_size}."
            )
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.embed_dims = embed_dims

        stage_dims = [embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims]
        layers = []
        in_c = in_channels
        for out_c in stage_dims:
            layers.append(
                SpikformerConv2dBNLIF(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool=True,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
            )
            in_c = out_c
        self.stages = nn.Sequential(*layers)
        self.positional_encoding = SpikformerConv2dBNLIF(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=False,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )
        self.grid_size = (img_size_h // patch_size, img_size_w // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x_seq: torch.Tensor):
        x_seq = self.stages(x_seq)
        residual = x_seq
        x_seq = self.positional_encoding(x_seq)
        return x_seq + residual


class SpikformerMLP(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.fc1 = layer.SeqToANNContainer(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
        )
        self.neuron1 = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )
        self.fc2 = layer.SeqToANNContainer(
            nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_features),
        )
        self.neuron2 = neuron.LIFNode(
            tau=tau,
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    def forward(self, x_seq: torch.Tensor):
        x_seq = self.neuron1(self.fc1(x_seq))
        x_seq = self.neuron2(self.fc2(x_seq))
        return x_seq

    def __spatial_split__(self):
        return self.fc1, self.neuron1, self.fc2, self.neuron2


class SpikformerBlock(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(dim=dim, num_heads=num_heads, backend=backend)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SpikformerMLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )

    def forward(self, x_seq: torch.Tensor):
        if x_seq.ndim != 5:
            raise ValueError(
                f"expected 5D input with shape [T, N, C, H, W], but got {x_seq.shape}"
            )
        T, N, C, H, W = x_seq.shape
        x_tokens = x_seq.flatten(3)
        x_tokens = x_tokens + self.attn(x_tokens)
        x_tokens = x_tokens + self.mlp(x_tokens)
        return x_tokens.reshape(T, N, C, H, W).contiguous()


class Spikformer(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dims: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depths: int = 4,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.depths = depths
        self.patch_embed = SpikformerPatchStem(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
                for _ in range(depths)
            ]
        )
        self.head = layer.Linear(embed_dims, num_classes, step_mode="m")
        self._init_weights()
        functional.set_step_mode(self, "m")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _to_sequence(self, x: torch.Tensor):
        if x.ndim == 4:
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if x.ndim == 5:
            return x
        raise ValueError(
            f"expected 4D image input [N, C, H, W] or 5D sequence input [T, N, C, H, W], but got {x.shape}"
        )

    def forward_features(self, x_seq: torch.Tensor):
        x_seq = self.patch_embed(x_seq)
        for block in self.blocks:
            x_seq = block(x_seq)
        return x_seq.flatten(3).mean(dim=-1)

    def forward(self, x: torch.Tensor):
        x_seq = self._to_sequence(x)
        x_seq = self.forward_features(x_seq)
        return self.head(x_seq)


def spikformer_ti(
    T: int = 4,
    in_channels: int = 3,
    img_size_h: int = 224,
    img_size_w: int = 224,
    num_classes: int = 1000,
    backend: str = "torch",
) -> Spikformer:
    return Spikformer(
        T=T,
        in_channels=in_channels,
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        num_classes=num_classes,
        embed_dims=256,
        num_heads=8,
        mlp_ratio=4.0,
        depths=4,
        backend=backend,
    )


def spikformer_s(
    T: int = 4,
    in_channels: int = 3,
    img_size_h: int = 224,
    img_size_w: int = 224,
    num_classes: int = 1000,
    backend: str = "torch",
) -> Spikformer:
    return Spikformer(
        T=T,
        in_channels=in_channels,
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        num_classes=num_classes,
        embed_dims=384,
        num_heads=12,
        mlp_ratio=4.0,
        depths=6,
        backend=backend,
    )
