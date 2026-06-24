import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.ann2snn import Converter
from spikingjelly.activation_based.ann2snn.operators import (
    TDGELU,
    TDLayerNorm,
    TDLinear,
    TDMultiheadAttention,
    TDScaledDotProductAttention,
)


def _activation_aware_calibration_channel_last(
    activation: torch.Tensor,
    threshold_std_scale: float = 3.0,
    eps: float = 1e-6,
):
    reduce_dims = tuple(range(activation.dim() - 1))
    offset = activation.mean(dim=reduce_dims)
    threshold = activation.std(dim=reduce_dims, unbiased=False) * threshold_std_scale
    threshold = torch.clamp(threshold, min=eps)
    return threshold.detach(), offset.detach()


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


class TinyActivationAwareTDTransformerBlock(TinyTDTransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        v_threshold: torch.Tensor,
        v_offset: torch.Tensor,
        surrogate_function: surrogate.SurrogateFunctionBase,
    ) -> None:
        super().__init__(embed_dim, num_heads, mlp_dim)
        self.activation_neuron = neuron.ActivationAwareIFNode(
            v_threshold=v_threshold,
            v_offset=v_offset,
            channel_dim=-1,
            surrogate_function=surrogate_function,
            step_mode="m",
        )
        self.hidden_spike_seq = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        norm_seq = self.norm1(x_seq)
        q_seq = self._split_heads(self.q_proj(norm_seq))
        k_seq = self._split_heads(self.k_proj(norm_seq))
        v_seq = self._split_heads(self.v_proj(norm_seq))
        attn_seq = self._merge_heads(self.attn(q_seq, k_seq, v_seq))
        x_seq = x_seq + self.out_proj(attn_seq)

        mlp_seq = self.fc1(self.norm2(x_seq))
        mlp_seq = self.act(mlp_seq)
        mlp_seq = self.activation_neuron(mlp_seq)
        self.hidden_spike_seq = mlp_seq
        mlp_seq = self.fc2(mlp_seq)
        return x_seq + mlp_seq


class TinyANNFunctionalSDPATransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-3, -2).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self.norm1(x)
        q = self._split_heads(self.q_proj(norm))
        k = self._split_heads(self.k_proj(norm))
        v = self._split_heads(self.v_proj(norm))
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        x = x + self.out_proj(self._merge_heads(attn))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class TinyANNMHATransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self.norm1(x)
        attn, _ = self.attn(
            norm,
            norm,
            norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


def _apply_ann_to_cumulative(
    block: nn.Module,
    x_seq: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    x_cum = x_seq.cumsum(dim=0)
    return torch.stack([block(x_cum[t], **kwargs) for t in range(x_seq.shape[0])])


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


def test_activation_aware_calibration_for_tiny_transformer_hidden_activation():
    torch.manual_seed(0)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.randn(5, 2, 4, 8)
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(x_seq)))

    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)

    assert threshold.shape == (16,)
    assert offset.shape == (16,)
    assert torch.isfinite(threshold).all()
    assert torch.isfinite(offset).all()
    assert (threshold > 0).all()


def test_activation_aware_tiny_transformer_block_forward_and_spike_sanity():
    torch.manual_seed(1)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    x_seq = torch.linspace(-1.0, 1.0, steps=5 * 2 * 4 * 8).view(5, 2, 4, 8)
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(x_seq)))
    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)
    block = TinyActivationAwareTDTransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        v_threshold=threshold,
        v_offset=offset,
        surrogate_function=surrogate.DeterministicPass(),
    )

    y_seq = block(x_seq)

    assert y_seq.shape == x_seq.shape
    assert torch.isfinite(y_seq).all()
    assert torch.isfinite(y_seq.cumsum(dim=0)).all()
    assert block.activation_neuron.v_threshold.shape == (16,)
    assert block.activation_neuron.v_offset.shape == (16,)
    assert block.hidden_spike_seq is not None
    assert torch.isfinite(block.hidden_spike_seq).all()
    assert block.hidden_spike_seq.sum() > 0

    block.activation_neuron.reset()
    y_seq_after_reset = block(x_seq)
    assert torch.allclose(y_seq, y_seq_after_reset)


def test_activation_aware_tiny_transformer_block_autograd_smoke():
    torch.manual_seed(2)
    probe_block = TinyTDTransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    calibration_seq = torch.randn(4, 2, 3, 8)
    hidden_seq = probe_block.act(
        probe_block.fc1(probe_block.norm2(calibration_seq))
    )
    threshold, offset = _activation_aware_calibration_channel_last(hidden_seq)
    block = TinyActivationAwareTDTransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        v_threshold=threshold,
        v_offset=offset,
        surrogate_function=surrogate.Sigmoid(),
    )
    x_seq = torch.randn(4, 2, 3, 8, requires_grad=True)

    y_seq = block(x_seq)
    y_seq.square().mean().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_converter_functional_sdpa_transformer_block_matches_ann_reference():
    block = TinyANNFunctionalSDPATransformerBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
    )
    block.eval()
    converted = Converter(dataloader=[]).replace_by_td_operators(block)
    modules = dict(converted.named_modules())
    x_seq = torch.randn(5, 2, 4, 8)
    attn_mask = torch.tensor(
        [
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    y_seq = converted(x_seq, attn_mask)
    expected = block(x_seq.cumsum(dim=0), attn_mask=attn_mask)

    assert isinstance(modules["norm1"], TDLayerNorm)
    assert isinstance(modules["q_proj"], TDLinear)
    assert isinstance(modules["act"], TDGELU)
    assert any(
        isinstance(module, TDScaledDotProductAttention)
        for module in modules.values()
    )
    assert not any(
        node.op == "call_function"
        and node.target is F.scaled_dot_product_attention
        for node in converted.graph.nodes
    )
    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_converter_mha_transformer_block_matches_ann_reference():
    block = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    block.eval()
    converted = Converter(dataloader=[]).replace_by_td_operators(block)
    modules = dict(converted.named_modules())
    x_seq = torch.randn(4, 2, 5, 8)
    attn_mask = torch.zeros(5, 5)
    attn_mask[:, -1] = float("-inf")

    y_seq = converted(x_seq, attn_mask)
    expected = _apply_ann_to_cumulative(block, x_seq, attn_mask=attn_mask)

    assert isinstance(modules["norm1"], TDLayerNorm)
    assert isinstance(modules["attn"], TDMultiheadAttention)
    assert isinstance(modules["fc1"], TDLinear)
    assert isinstance(modules["act"], TDGELU)
    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_converter_transformer_block_autograd_smoke():
    block = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    converted = Converter(dataloader=[]).replace_by_td_operators(block)
    x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)

    y_seq = converted(x_seq)
    y_seq.square().sum().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in converted.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
