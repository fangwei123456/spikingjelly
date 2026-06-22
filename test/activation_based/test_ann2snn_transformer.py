import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDScaledDotProductAttention,
)


class TinyTDTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = TDLayerNorm(embed_dim)
        self.q_proj = TDLinear(embed_dim, embed_dim)
        self.k_proj = TDLinear(embed_dim, embed_dim)
        self.v_proj = TDLinear(embed_dim, embed_dim)
        self.attn = TDScaledDotProductAttention()
        self.out_proj = TDLinear(embed_dim, embed_dim)
        self.norm2 = TDLayerNorm(embed_dim)
        self.fc1 = TDLinear(embed_dim, mlp_dim)
        self.act = TDGELU()
        self.fc2 = TDLinear(mlp_dim, embed_dim)

    def _split_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        t, batch_size, seq_len, _ = x_seq.shape
        x_seq = x_seq.reshape(t, batch_size, seq_len, self.num_heads, self.head_dim)
        return x_seq.transpose(2, 3)

    def _merge_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        t, batch_size, _, seq_len, _ = x_seq.shape
        x_seq = x_seq.transpose(2, 3).contiguous()
        return x_seq.reshape(t, batch_size, seq_len, self.embed_dim)

    @staticmethod
    def _ann_linear(module: TDLinear, x_cum: torch.Tensor) -> torch.Tensor:
        return F.linear(x_cum, module.weight, module.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        norm_seq = self.norm1(x_seq)
        q_seq = self._split_heads(self.q_proj(norm_seq))
        k_seq = self._split_heads(self.k_proj(norm_seq))
        v_seq = self._split_heads(self.v_proj(norm_seq))
        attn_seq = self._merge_heads(self.attn(q_seq, k_seq, v_seq))
        x_seq = x_seq + self.out_proj(attn_seq)

        mlp_seq = self.fc1(self.norm2(x_seq))
        mlp_seq = self.act(mlp_seq)
        mlp_seq = self.fc2(mlp_seq)
        return x_seq + mlp_seq

    def ann_reference(self, x_cum: torch.Tensor) -> torch.Tensor:
        norm_cum = F.layer_norm(
            x_cum,
            self.norm1.normalized_shape,
            self.norm1.weight,
            self.norm1.bias,
            self.norm1.eps,
        )
        q_cum = self._split_heads(self._ann_linear(self.q_proj, norm_cum))
        k_cum = self._split_heads(self._ann_linear(self.k_proj, norm_cum))
        v_cum = self._split_heads(self._ann_linear(self.v_proj, norm_cum))
        attn_cum = F.scaled_dot_product_attention(
            q_cum,
            k_cum,
            v_cum,
            dropout_p=0.0,
        )
        x_cum = x_cum + self._ann_linear(self.out_proj, self._merge_heads(attn_cum))

        mlp_cum = F.layer_norm(
            x_cum,
            self.norm2.normalized_shape,
            self.norm2.weight,
            self.norm2.bias,
            self.norm2.eps,
        )
        mlp_cum = self._ann_linear(self.fc1, mlp_cum)
        mlp_cum = F.gelu(mlp_cum, approximate=self.act.approximate)
        mlp_cum = self._ann_linear(self.fc2, mlp_cum)
        return x_cum + mlp_cum


def test_tiny_transformer_block_matches_ann_reference_on_cumulative_input():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(5, 2, 4, 8)

    y_seq = block(x_seq)
    expected = block.ann_reference(x_seq.cumsum(dim=0))

    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_tiny_transformer_block_final_cumulative_output_matches_total_input():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(4, 2, 5, 8)

    y_seq = block(x_seq)
    expected = block.ann_reference(x_seq.cumsum(dim=0))

    assert torch.allclose(
        y_seq.cumsum(dim=0)[-1],
        expected[-1],
        atol=1e-5,
        rtol=1e-5,
    )


def test_tiny_transformer_block_uses_affine_td_linear_layers():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)

    linear_layers = [
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.out_proj,
        block.fc1,
        block.fc2,
    ]

    assert all(layer.bias is not None for layer in linear_layers)


def test_tiny_transformer_block_autograd_smoke():
    block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)

    y_seq = block(x_seq)
    y_seq.square().sum().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert block.q_proj.bias.grad is not None
    assert torch.isfinite(block.q_proj.bias.grad).all()
