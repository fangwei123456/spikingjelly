import pytest
import torch

try:
    import transformers
except ImportError:
    transformers = None

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import (
    ModuleConverter,
    Qwen2SNNCalibration,
    Qwen2SNNConfig,
    Qwen2SNNRecipe,
    calibrate_qwen2_snn,
)


def _tiny_qwen2(*, tie_word_embeddings: bool = False):
    if transformers is None:
        pytest.skip("transformers is not installed")
    torch.manual_seed(20260725)
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=tie_word_embeddings,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def _inputs():
    return {
        "input_ids": torch.tensor([[0, 3, 4, 5], [0, 0, 6, 7]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]),
    }


def _convert(model):
    config = Qwen2SNNConfig(
        time_steps=4,
        calibration_levels=2,
        calibration_reservoir_size=16,
    )
    calibration = calibrate_qwen2_snn(model, [_inputs()], config)
    converted = ModuleConverter(Qwen2SNNRecipe(calibration, config)).convert(model)
    return calibration, converted


def test_qwen2_calibration_round_trip_and_conversion_preserve_contract():
    calibration, converted = _convert(_tiny_qwen2(tie_word_embeddings=True))

    restored = Qwen2SNNCalibration.from_state_dict(calibration.state_dict())

    assert restored.valid_token_count == 5
    assert restored.time_steps == calibration.time_steps
    assert restored.calibration_levels == calibration.calibration_levels
    assert restored.calibration_quantile == calibration.calibration_quantile
    assert restored.calibration_reservoir_size == calibration.calibration_reservoir_size
    assert restored.calibration_seed == calibration.calibration_seed
    assert torch.equal(restored.input_scale, calibration.input_scale)
    assert tuple(restored.layer_scales[0]) == ("query", "key", "value", "mlp")
    assert all(
        torch.equal(restored.layer_scales[0][name], calibration.layer_scales[0][name])
        for name in restored.layer_scales[0]
    )
    assert any(parameter.requires_grad for parameter in converted.parameters())
    assert converted.lm_head.weight is converted.embed_tokens.weight
    assert converted.device == converted.embed_tokens.weight.device
    assert converted.get_input_embeddings() is converted.embed_tokens
    assert converted.get_output_embeddings() is converted.lm_head
    assert converted.structure_summary()["converted_decoder_count"] == 1
    assert [record["name"] for record in converted.encoder_statistics()] == [
        "model.input",
        "layer.0.query",
        "layer.0.key",
        "layer.0.value",
        "layer.0.mlp",
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("time_steps", 1.5),
        ("time_steps", True),
        ("calibration_levels", 1.5),
        ("calibration_reservoir_size", 1.5),
        ("calibration_seed", 1.5),
    ),
)
def test_qwen2_config_rejects_non_integer_discrete_values(name, value):
    with pytest.raises(TypeError, match=name):
        Qwen2SNNConfig(**{name: value})


def test_qwen2_config_rejects_unknown_neuron_backend():
    with pytest.raises(ValueError, match="neuron_backend"):
        Qwen2SNNConfig(neuron_backend="unknown")


def test_qwen2_exact_td_matches_hugging_face_with_left_padding():
    model = _tiny_qwen2()
    _, converted = _convert(model)
    inputs = _inputs()

    with torch.inference_mode():
        expected = model(**inputs).logits
        actual = converted(**inputs, encoding_mode="exact_td").logits

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen2_exact_td_preserves_lm_head_bias():
    model = _tiny_qwen2()
    model.lm_head = torch.nn.Linear(8, 32, bias=True)
    _, converted = _convert(model)
    inputs = _inputs()

    with torch.inference_mode():
        expected = model(**inputs).logits
        actual = converted(**inputs, encoding_mode="exact_td").logits

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen2_cached_decode_matches_full_sequence():
    model = _tiny_qwen2()
    _, converted = _convert(model)
    prompt = torch.tensor([[1, 2, 3]])
    next_token = torch.tensor([[4]])

    with torch.inference_mode():
        functional.reset_net(converted)
        prefill = converted(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            encoding_mode="exact_td",
            use_cache=True,
        )
        functional.reset_net(converted)
        cached = converted(
            input_ids=next_token,
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            encoding_mode="exact_td",
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        functional.reset_net(converted)
        full = converted(
            input_ids=torch.cat((prompt, next_token), dim=1),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            encoding_mode="exact_td",
        )

    assert torch.allclose(
        cached.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-5
    )


def test_qwen2_rejects_cache_when_cache_use_is_disabled():
    _, converted = _convert(_tiny_qwen2())
    prompt = torch.tensor([[1, 2, 3]])

    with torch.inference_mode():
        prefill = converted(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            encoding_mode="exact_td",
            use_cache=True,
        )
        cache = prefill.past_key_values
        original_length = cache.get_seq_length()
        with pytest.raises(ValueError, match="requires use_cache=True"):
            converted(
                input_ids=torch.tensor([[4]]),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                encoding_mode="exact_td",
                past_key_values=cache,
            )

    assert cache.get_seq_length() == original_length


def test_qwen2_signed_if_replays_after_reset():
    _, converted = _convert(_tiny_qwen2())
    inputs = _inputs()

    with torch.inference_mode():
        first = converted(**inputs, encoding_mode="signed_if").logits
        functional.reset_net(converted)
        second = converted(**inputs, encoding_mode="signed_if").logits

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_qwen2_qcfs_sg_allows_autograd():
    _, converted = _convert(_tiny_qwen2())
    inputs = _inputs()

    output = converted(**inputs, encoding_mode="qcfs_sg")
    loss = output.logits.float().sum()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in converted.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_qwen2_calibration_failure_does_not_leave_hooks_installed():
    model = _tiny_qwen2()
    config = Qwen2SNNConfig(
        time_steps=4,
        calibration_levels=2,
        calibration_reservoir_size=16,
    )

    with pytest.raises(TypeError, match="tensor input_ids"):
        calibrate_qwen2_snn(
            model,
            [{"input_ids": "invalid", "attention_mask": torch.ones(1, 1)}],
            config,
        )

    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1))
    assert output.logits.shape == (1, 1, 32)


def test_qwen2_rejects_mrope():
    model = _tiny_qwen2()
    model.config.use_mrope = True

    with pytest.raises(ValueError, match="MRoPE"):
        calibrate_qwen2_snn(model, [_inputs()], Qwen2SNNConfig())
