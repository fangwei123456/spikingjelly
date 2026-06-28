import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.ann2snn import (
    Converter,
    STATransformerRecipe,
    TransformerSpikeEquivalentRecipe,
)
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


class TinyImageTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(8)
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x.mean(dim=1)


class TinyConstantTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias_token = nn.Parameter(torch.randn(1, 1, 8))
        self.norm = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias_token
        return self.fc(self.norm(x)).mean(dim=1)


class TinyKeywordTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(pixel_values)).mean(dim=1)


class TinyDictOutputTransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.fc(self.norm(x)).mean(dim=1)
        return {"logits": logits}


class TinyConvPaddingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            3,
            4,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).mean(dim=(2, 3))
        return self.fc(x)


class TinyHeadClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.body = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.norm(x))).mean(dim=1)


class TinyMHAWeightsBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            8,
            2,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=True,
        )


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
    hidden_seq = probe_block.act(probe_block.fc1(probe_block.norm2(calibration_seq)))
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
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
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
        isinstance(module, TDScaledDotProductAttention) for module in modules.values()
    )
    assert not any(
        node.op == "call_function" and node.target is F.scaled_dot_product_attention
        for node in converted.graph.nodes
    )
    assert torch.allclose(y_seq.cumsum(dim=0), expected, atol=1e-5, rtol=1e-5)


def test_converter_mha_transformer_block_matches_ann_reference():
    block = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16)
    block.eval()
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
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
    converted = Converter(recipe=TransformerSpikeEquivalentRecipe()).convert(block)
    x_seq = torch.randn(3, 2, 4, 8, requires_grad=True)

    y_seq = converted(x_seq)
    y_seq.square().sum().backward()

    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad).all()
    for parameter in converted.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sta_transformer_recipe_converts_affine_layers_and_runs_inner_steps():
    torch.manual_seed(3)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long))]
    recipe = STATransformerRecipe(
        dataloader=calibration,
        time_steps=4,
        threshold_mode="mse",
    )

    converted = Converter(recipe=recipe).convert(model)
    modules = dict(converted.named_modules())
    x = torch.randn(2, 3, 16, 16)
    y = converted(x)

    assert converted.time_steps == 4
    assert not hasattr(modules["model.patch"], "v_threshold")
    assert not hasattr(modules["model.fc1"], "v_threshold")
    assert not hasattr(modules["model.fc2"], "v_threshold")
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_repeated_forward_resets_state():
    torch.manual_seed(31)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 16, 16)

    y0 = converted(x)
    y1 = converted(x)

    assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-6)


def test_sta_transformer_recipe_preserves_final_wrapper_training_flag():
    model = TinyImageTransformerClassifier().train()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)

    assert converted.training is False


def test_sta_transformer_recipe_equivalent_mode_skips_calibration_loop():
    class CountingCalibration:
        def __iter__(self):
            raise AssertionError("equivalent mode should not iterate calibration")

    model = TinyImageTransformerClassifier().eval()

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=CountingCalibration(),
            time_steps=4,
            mode="equivalent",
        )
    ).convert(model)

    assert torch.isfinite(converted(torch.randn(2, 3, 16, 16))).all()


def test_sta_transformer_recipe_preserves_nonzero_conv_padding_mode():
    torch.manual_seed(39)
    model = TinyConvPaddingClassifier().eval()
    calibration = [(torch.randn(2, 3, 8, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 3, 8, 8)

    assert torch.allclose(converted(x), model(x), atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_equivalent_mha_matches_ann():
    torch.manual_seed(32)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    modules = dict(converted.named_modules())
    x = torch.randn(2, 4, 8)

    assert torch.allclose(converted(x), model(x), atol=1e-5, rtol=1e-5)
    assert modules["model.attn"].attn.batch_first is True


def test_sta_transformer_recipe_preserves_static_attention_mask_kwargs():
    torch.manual_seed(40)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 8)
    attn_mask = torch.zeros(4, 4)
    attn_mask[:, -1] = float("-inf")

    assert torch.allclose(
        converted(x, attn_mask=attn_mask),
        model(x, attn_mask=attn_mask),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sta_transformer_recipe_returns_attention_weight_deltas():
    torch.manual_seed(41)
    model = TinyMHAWeightsBlock().eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 8)
    attn_mask = torch.zeros(4, 4)
    attn_mask[:, -1] = float("-inf")

    converted_output, converted_weights = converted(x, attn_mask=attn_mask)
    expected_output, expected_weights = model(x, attn_mask=attn_mask)

    assert torch.allclose(converted_output, expected_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(converted_weights, expected_weights, atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_spiking_encoder_mode_encodes_nonlinear_outputs():
    torch.manual_seed(8)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
            threshold_mode="mse",
        )
    ).convert(model)
    modules = dict(converted.named_modules())
    y = converted(torch.randn(2, 3, 16, 16))

    assert not hasattr(modules["model.fc1"], "v_threshold")
    assert not hasattr(modules["model.fc2"], "v_threshold")
    assert hasattr(modules["model.norm"].encoder, "v_threshold")
    assert hasattr(modules["model.act"].encoder, "v_threshold")
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_spiking_encoder_mode_encodes_mha_output():
    torch.manual_seed(33)
    model = TinyANNMHATransformerBlock(embed_dim=8, num_heads=2, mlp_dim=16).eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_encoder",
        )
    ).convert(model)
    modules = dict(converted.named_modules())
    y = converted(torch.randn(2, 4, 8))

    assert hasattr(modules["model.attn"].encoder, "v_threshold")
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_threshold_mode_max_path():
    torch.manual_seed(34)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
            threshold_mode="max",
        )
    ).convert(model)
    threshold = dict(converted.named_modules())["model.fc1"].v_threshold
    y = converted(torch.randn(2, 3, 16, 16))

    assert torch.isfinite(threshold).all()
    assert (threshold > 0).all()
    assert torch.isfinite(y).all()


def test_sta_transformer_recipe_time_steps_affects_mse_thresholds():
    torch.manual_seed(4)
    model_t2 = TinyImageTransformerClassifier().eval()
    model_t8 = TinyImageTransformerClassifier().eval()
    model_t8.load_state_dict(model_t2.state_dict())
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted_t2 = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=2,
            mode="spiking_affine",
            threshold_mode="mse",
        )
    ).convert(model_t2)
    converted_t8 = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=8,
            mode="spiking_affine",
            threshold_mode="mse",
        )
    ).convert(model_t8)

    threshold_t2 = dict(converted_t2.named_modules())["model.fc1"].v_threshold
    threshold_t8 = dict(converted_t8.named_modules())["model.fc1"].v_threshold

    assert not torch.allclose(threshold_t2, threshold_t8)


def test_sta_transformer_recipe_threshold_scale_rescales_thresholds():
    torch.manual_seed(5)
    model_base = TinyImageTransformerClassifier().eval()
    model_scaled = TinyImageTransformerClassifier().eval()
    model_scaled.load_state_dict(model_base.state_dict())
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted_base = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
            threshold_mode="mse",
        )
    ).convert(model_base)
    converted_scaled = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
            threshold_mode="mse",
            threshold_scale=0.5,
        )
    ).convert(model_scaled)

    threshold_base = dict(converted_base.named_modules())["model.fc1"].v_threshold
    threshold_scaled = dict(converted_scaled.named_modules())["model.fc1"].v_threshold

    assert torch.allclose(threshold_scaled, threshold_base * 0.5)


def test_sta_transformer_recipe_num_calibration_batches_limits_observer_updates():
    torch.manual_seed(35)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),) for _ in range(3)]
    recipe = STATransformerRecipe(
        dataloader=calibration,
        time_steps=4,
        mode="spiking_affine",
        num_calibration_batches=1,
    )

    Converter(recipe=recipe).convert(model)

    assert recipe._observers
    assert all(
        observer.num_batches_tracked == 1 for observer in recipe._observers.values()
    )


def test_sta_transformer_recipe_dict_batch_and_dict_output_are_supported():
    torch.manual_seed(36)
    model = TinyDictOutputTransformerClassifier().eval()
    calibration = [(torch.randn(2, 4, 4),)]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)

    x = torch.randn(2, 4, 4)
    output = converted(x)

    assert set(output.keys()) == {"logits"}
    assert torch.allclose(output["logits"], model(x)["logits"], atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_kwargs_calibration_path():
    torch.manual_seed(37)
    model = TinyKeywordTransformerClassifier().eval()
    calibration = [{"pixel_values": torch.randn(2, 4, 4)}]
    converted = Converter(
        recipe=STATransformerRecipe(dataloader=calibration, time_steps=4)
    ).convert(model)
    x = torch.randn(2, 4, 4)

    assert torch.allclose(converted(pixel_values=x), model(x), atol=1e-5, rtol=1e-5)


def test_sta_transformer_recipe_spike_classifier_flag_controls_head_conversion():
    torch.manual_seed(38)
    model_default = TinyHeadClassifier().eval()
    model_with_head = TinyHeadClassifier().eval()
    model_with_head.load_state_dict(model_default.state_dict())
    calibration = [(torch.randn(2, 4, 4),)]

    converted_default = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
        )
    ).convert(model_default)
    converted_with_head = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
            spike_classifier=True,
        )
    ).convert(model_with_head)

    assert not hasattr(
        dict(converted_default.named_modules())["model.head"], "v_threshold"
    )
    assert hasattr(
        dict(converted_with_head.named_modules())["model.head"], "v_threshold"
    )


def test_sta_transformer_recipe_can_spike_conv2d_explicitly():
    torch.manual_seed(6)
    model = TinyImageTransformerClassifier().eval()
    calibration = [(torch.randn(2, 3, 16, 16),)]

    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            mode="spiking_affine",
            threshold_mode="mse",
            spike_conv2d=True,
        )
    ).convert(model)
    modules = dict(converted.named_modules())

    assert hasattr(modules["model.patch"], "v_threshold")
    assert modules["model.patch"].v_threshold.shape == (8,)


def test_sta_transformer_recipe_wraps_tensor_constants_as_first_step_inputs():
    torch.manual_seed(7)
    model = TinyConstantTransformerClassifier().eval()
    calibration = [(torch.randn(2, 4, 8),)]
    converted = Converter(
        recipe=STATransformerRecipe(
            dataloader=calibration,
            time_steps=4,
            threshold_mode="mse",
        )
    ).convert(model)
    x = torch.randn(2, 4, 8)
    tensor_get_attrs = [
        node
        for node in converted.model.graph.nodes
        if node.op == "get_attr"
        and torch.is_tensor(
            STATransformerRecipe._get_attr_value(converted.model, node.target)
        )
    ]

    assert tensor_get_attrs == []
    assert torch.allclose(converted(x), model(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"time_steps": 0}, "time_steps"),
        ({"mode": "missing"}, "mode"),
        ({"threshold_mode": "missing"}, "threshold_mode"),
        ({"threshold_scale": 0.0}, "threshold_scale"),
        ({"momentum": 1.5}, "momentum"),
        ({"num_calibration_batches": 0}, "num_calibration_batches"),
    ],
)
def test_sta_transformer_recipe_validate_errors(kwargs, match):
    model = TinyImageTransformerClassifier().eval()
    recipe = STATransformerRecipe(
        dataloader=[(torch.randn(2, 3, 16, 16),)],
        **kwargs,
    )

    with pytest.raises(ValueError, match=match):
        Converter(recipe=recipe).convert(model)
