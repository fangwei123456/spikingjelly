import copy
import os

import pytest
import torch

from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.model import Spikformer
from spikingjelly.activation_based.precision import (
    Float8TransformerEnginePolicy,
    PrecisionConfig,
    TransformerEngineDotProductAttentionAdapter,
    build_capability_report,
    prepare_model_for_precision,
    validate_capability,
)


FP8_MODES = ("fp8-torchao", "fp8-te")


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(candidate.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _cosine_similarity(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(candidate, reference, dim=0))


def _assert_gradient_quality(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    min_cosine: float,
    max_relative_l2: float,
) -> None:
    reference_norm = float(torch.linalg.vector_norm(reference.float()).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate.float()).item())
    if reference_norm < 1e-8:
        assert candidate_norm < 1e-5
        return
    assert _cosine_similarity(candidate, reference) >= min_cosine
    assert _relative_l2(candidate, reference) <= max_relative_l2


def _reset_model(model: torch.nn.Module) -> None:
    functional.reset_net(model)


def _required_backends() -> set[str]:
    value = os.environ.get("SPIKINGJELLY_REQUIRE_FP8_BACKENDS", "")
    required = {item.strip() for item in value.split(",") if item.strip()}
    return set(FP8_MODES) if "all" in required else required


def _required_recipes() -> set[str]:
    value = os.environ.get("SPIKINGJELLY_REQUIRE_FP8_RECIPES", "")
    return {item.strip() for item in value.split(",") if item.strip()}


def _unavailable_backend(mode: str, reason: str) -> None:
    if mode in _required_backends():
        pytest.fail(f"Required FP8 backend {mode!r} is unavailable: {reason}")
    pytest.skip(reason)


def _cuda_device_or_skip(mode: str) -> torch.device:
    if not torch.cuda.is_available():
        _unavailable_backend(mode, "CUDA is required for FP8 hardware validation.")
    return torch.device("cuda", 0)


def _require_backend_capability(
    model: torch.nn.Module,
    device: torch.device,
    mode: str,
    recipe: str = "auto",
) -> dict:
    report = build_capability_report(model, device, mode)
    try:
        validate_capability(report)
    except RuntimeError as exc:
        _unavailable_backend(mode, str(exc))
    if recipe in {"block", "mxfp8"}:
        availability = report.get("te_recipe_availability") or {}
        if not availability.get(recipe):
            reason = f"Transformer Engine recipe {recipe!r} is unavailable."
            if recipe in _required_recipes():
                pytest.fail(f"Required FP8 recipe {recipe!r} is unavailable.")
            pytest.skip(reason)
    return report


def _assert_train_and_infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    mode: str,
    recipe: str = "auto",
    output_max_relative_l2: float = 0.10,
    gradient_min_cosine: float = 0.98,
    gradient_max_relative_l2: float = 0.20,
) -> None:
    device = x.device
    reference_model = copy.deepcopy(model).eval()
    reload_model = copy.deepcopy(model)
    _reset_model(reference_model)
    with torch.inference_mode():
        reference_output = reference_model(x)
    reference_model.train()
    _reset_model(reference_model)
    reference_training_output = reference_model(x)
    reference_loss = (reference_training_output.float() - 0.25).square().mean()
    reference_loss.backward()
    reference_gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in reference_model.parameters()
    ]
    _require_backend_capability(model, device, mode, recipe)
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(
            mode=mode,
            strictness="strict",
            fp8_recipe=recipe,
            device=str(device),
        ),
    )
    model = artifacts.model
    assert artifacts.effective_config.mode == mode
    assert artifacts.policy.conversion_report()["converted_modules"]

    model.eval()
    _reset_model(model)
    with torch.inference_mode(), artifacts.autocast_context():
        initial_fp8_output = model(x)
    torch.testing.assert_close(
        initial_fp8_output.float(),
        reference_output.float(),
        atol=0.25,
        rtol=0.25,
    )
    assert _relative_l2(initial_fp8_output, reference_output) <= output_max_relative_l2

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    _reset_model(model)
    with artifacts.autocast_context():
        output = model(x)
        loss = (output.float() - 0.25).square().mean()
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, reference_loss, atol=0.1, rtol=0.25)
    assert _relative_l2(loss, reference_loss) <= 0.10
    artifacts.backward(loss, optimizer, parameters=model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    candidate_parameters = list(model.parameters())
    for parameter, reference_gradient in zip(
        candidate_parameters, reference_gradients, strict=True
    ):
        if reference_gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            candidate_gradient = parameter.grad.float()
            reference_gradient = reference_gradient.float()
            if (
                reference_gradient.ndim == candidate_gradient.ndim + 1
                and reference_gradient.shape[-1] == 1
            ):
                reference_gradient = reference_gradient.squeeze(-1)
            assert candidate_gradient.shape == reference_gradient.shape
            torch.testing.assert_close(
                candidate_gradient,
                reference_gradient,
                atol=0.25,
                rtol=0.35,
            )
            _assert_gradient_quality(
                candidate_gradient,
                reference_gradient,
                min_cosine=gradient_min_cosine,
                max_relative_l2=gradient_max_relative_l2,
            )
    assert any(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, model.parameters(), strict=True)
    )

    model.eval()
    _reset_model(model)
    with torch.inference_mode(), artifacts.autocast_context():
        trained_source_output = model(x)
    state_dict = model.state_dict()
    reload_artifacts = prepare_model_for_precision(
        reload_model,
        device,
        PrecisionConfig(
            mode=mode,
            strictness="strict",
            fp8_recipe=recipe,
            device=str(device),
        ),
    )
    assert reload_artifacts.effective_config.mode == mode
    assert reload_artifacts.policy.conversion_report()["converted_modules"]
    reload_model = reload_artifacts.model
    reload_model.load_state_dict(state_dict, strict=True)
    reload_model.eval()
    _reset_model(reload_model)
    with torch.inference_mode(), reload_artifacts.autocast_context():
        inference_output = reload_model(x)
    _reset_model(reload_model)
    with torch.inference_mode(), reload_artifacts.autocast_context():
        repeated_inference_output = reload_model(x)
    assert inference_output.shape == output.shape
    assert torch.isfinite(inference_output).all()
    torch.testing.assert_close(
        inference_output.float(),
        trained_source_output.float(),
        atol=0.25,
        rtol=0.25,
    )
    assert (
        _relative_l2(inference_output, trained_source_output) <= output_max_relative_l2
    )
    torch.testing.assert_close(
        repeated_inference_output.float(),
        inference_output.float(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_trains_updates_parameters_and_runs_inference(mode):
    device = _cuda_device_or_skip(mode)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
    ).to(device)
    x = torch.randn(32, 64, device=device)

    _assert_train_and_infer(model, x, mode)


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_pointwise_conv1d_trains_and_runs_multistep_inference(mode):
    device = _cuda_device_or_skip(mode)
    model = layer.Conv1d(16, 32, kernel_size=1, step_mode="m").to(device)
    x = torch.randn(2, 16, 16, 8, device=device)

    _assert_train_and_infer(model, x, mode)


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_stateful_snn_trains_resets_and_runs_inference(mode):
    device = _cuda_device_or_skip(mode)
    torch.manual_seed(20260719)
    model = torch.nn.Sequential(
        layer.Linear(64, 64, step_mode="m"),
        neuron.LIFNode(step_mode="m"),
        layer.Linear(64, 32, step_mode="m"),
    ).to(device)
    x = torch.randn(4, 16, 64, device=device) * 0.1

    _assert_train_and_infer(
        model,
        x,
        mode,
        output_max_relative_l2=0.15,
        gradient_min_cosine=0.95,
        gradient_max_relative_l2=0.30,
    )


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_snn_boundary_spike_mismatch_is_bounded(mode):
    device = _cuda_device_or_skip(mode)
    torch.manual_seed(20260719)
    model = torch.nn.Sequential(
        layer.Linear(64, 64, step_mode="m"),
        neuron.LIFNode(step_mode="m", v_reset=None, store_v_seq=True),
    ).to(device)
    reference_model = copy.deepcopy(model).eval()
    x = torch.randn(8, 32, 64, device=device)
    _require_backend_capability(model, device, mode)
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=mode, strictness="strict", device=str(device)),
    )
    assert artifacts.effective_config.mode == mode
    assert artifacts.policy.conversion_report()["converted_modules"]
    model = artifacts.model.eval()

    _reset_model(reference_model)
    _reset_model(model)
    with torch.inference_mode():
        reference_spike = reference_model(x)
    with torch.inference_mode(), artifacts.autocast_context():
        candidate_spike = model(x)

    reference_h = (
        reference_model[1].v_seq + reference_spike * reference_model[1].v_threshold
    )
    near_threshold = (reference_h - reference_model[1].v_threshold).abs() < 0.05
    away_from_threshold = ~near_threshold
    mismatch = candidate_spike != reference_spike
    assert near_threshold.any()
    assert away_from_threshold.any()
    assert not mismatch[away_from_threshold].any()
    mismatch_ratio = float(mismatch.float().mean().item())
    assert mismatch_ratio <= 0.01


@pytest.mark.parametrize("mode", FP8_MODES)
def test_tiny_spikformer_fp8_trains_and_runs_inference(mode):
    device = _cuda_device_or_skip(mode)
    torch.manual_seed(20260719)
    model = Spikformer(
        T=2,
        in_channels=3,
        img_size_h=64,
        img_size_w=64,
        num_classes=16,
        embed_dims=64,
        num_heads=4,
        depths=2,
        backend="torch",
    ).to(device)
    reference_model = copy.deepcopy(model).train()
    _require_backend_capability(model, device, mode)
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=mode, strictness="strict", device=str(device)),
    )
    assert artifacts.effective_config.mode == mode
    model = artifacts.model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = torch.randn(16, 3, 64, 64, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    target = torch.randint(0, 16, (16,), device=device)

    _reset_model(reference_model)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_output = reference_model(reference_x)
        reference_loss = torch.nn.functional.cross_entropy(
            reference_output.mean(0).float(), target
        )
    reference_loss.backward()

    model.train()
    _reset_model(model)
    optimizer.zero_grad(set_to_none=True)
    with artifacts.autocast_context():
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output.mean(0).float(), target)
    artifacts.backward(loss, optimizer, parameters=model.parameters())
    assert output.shape == (2, 16, 16)
    assert torch.isfinite(loss)
    assert _relative_l2(output, reference_output) <= 0.35
    assert _relative_l2(loss, reference_loss) <= 0.10
    assert x.grad is not None and reference_x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(reference_x.grad).all()
    assert model.head.weight.grad is not None
    assert reference_model.head.weight.grad is not None
    _assert_gradient_quality(
        model.head.weight.grad,
        reference_model.head.weight.grad,
        min_cosine=0.90,
        max_relative_l2=0.45,
    )
    assert artifacts.policy.conversion_report()["converted_modules"]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    _reset_model(model)
    with torch.inference_mode(), artifacts.autocast_context():
        inference_output = model(x)
    assert inference_output.shape == output.shape
    assert torch.isfinite(inference_output).all()


def _train_regression_curve(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> list[float]:
    device = x.device
    _require_backend_capability(model, device, mode)
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=mode, strictness="strict", device=str(device)),
    )
    assert artifacts.effective_config.mode == mode
    if mode in FP8_MODES:
        assert artifacts.policy.conversion_report()["converted_modules"]
    model = artifacts.model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        with artifacts.autocast_context():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output.float(), target)
        assert torch.isfinite(loss)
        artifacts.backward(loss, optimizer, parameters=model.parameters())
        losses.append(float(loss.detach().item()))
    return losses


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_training_converges_comparably_to_bf16(mode):
    device = _cuda_device_or_skip(mode)
    torch.manual_seed(20260719)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 64),
    ).to(device)
    x = torch.randn(64, 64, device=device)
    teacher = torch.randn(64, 64, device=device) / 8.0
    target = x @ teacher

    bf16_losses = _train_regression_curve(copy.deepcopy(model), x, target, "bf16")
    fp8_losses = _train_regression_curve(copy.deepcopy(model), x, target, mode)

    assert bf16_losses[-1] <= bf16_losses[0] * 0.90
    assert fp8_losses[-1] <= fp8_losses[0] * 0.90
    assert fp8_losses[-1] <= bf16_losses[-1] * 1.20


@pytest.mark.parametrize("mode", FP8_MODES)
def test_fp8_backend_runs_inference_on_second_cuda_device(mode):
    _cuda_device_or_skip(mode)
    if torch.cuda.device_count() < 2:
        pytest.skip("A second CUDA device is required for this smoke test.")
    device = torch.device("cuda", 1)
    model = torch.nn.Linear(64, 64).to(device).eval()
    x = torch.randn(16, 64, device=device)
    _require_backend_capability(model, device, mode)
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=mode, strictness="strict", device=str(device)),
    )
    assert artifacts.effective_config.mode == mode
    assert artifacts.policy.conversion_report()["converted_modules"]
    with torch.inference_mode(), artifacts.autocast_context():
        output = artifacts.model(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("recipe", ["auto", "delayed", "current", "block", "mxfp8"])
def test_fp8_te_recipes_train_and_run_inference(recipe):
    device = _cuda_device_or_skip("fp8-te")
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(64),
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
    ).to(device)
    x = torch.randn(32, 64, device=device)

    _assert_train_and_infer(model, x, "fp8-te", recipe)


@pytest.mark.parametrize(
    ("model", "x"),
    [
        (
            torch.nn.LayerNorm(64),
            torch.randn(32, 64),
        ),
        (
            torch.nn.Sequential(
                torch.nn.LayerNorm(64),
                torch.nn.Linear(64, 64),
            ),
            torch.randn(32, 64),
        ),
        (
            layer.Linear(64, 64, step_mode="s"),
            torch.randn(32, 64),
        ),
    ],
    ids=("layer-norm", "layer-norm-linear", "layer-linear-single-step"),
)
def test_fp8_te_conversion_surfaces_match_fp32_on_hardware(model, x):
    device = _cuda_device_or_skip("fp8-te")
    _assert_train_and_infer(model.to(device), x.to(device), "fp8-te")


def test_fp8_te_attention_adapter_propagates_gradients_and_runs_inference():
    device = _cuda_device_or_skip("fp8-te")
    adapter = TransformerEngineDotProductAttentionAdapter(
        num_attention_heads=4,
        head_dim=16,
    ).to(device)
    _require_backend_capability(adapter, device, "fp8-te")
    policy = Float8TransformerEnginePolicy(fp8_recipe="auto")
    try:
        policy.check_capability(adapter, device)
    except RuntimeError as exc:
        _unavailable_backend("fp8-te", str(exc))
    query = torch.randn(2, 4, 32, 16, device=device, requires_grad=True)
    key = torch.randn(2, 4, 32, 16, device=device, requires_grad=True)
    value = torch.randn(2, 4, 32, 16, device=device, requires_grad=True)
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        reference_query,
        reference_key,
        reference_value,
    )
    reference_loss = reference_output.float().square().mean()
    reference_loss.backward()

    adapter.train()
    with policy.autocast_context():
        output = adapter(query, key, value)
        loss = output.float().square().mean()
    loss.backward()
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert _relative_l2(output, reference_output) <= 0.15
    assert _relative_l2(loss, reference_loss) <= 0.10
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    )
    for candidate, reference in zip(
        (query, key, value),
        (reference_query, reference_key, reference_value),
        strict=True,
    ):
        _assert_gradient_quality(
            candidate.grad,
            reference.grad,
            min_cosine=0.95,
            max_relative_l2=0.30,
        )

    adapter.eval()
    with torch.inference_mode(), policy.autocast_context():
        inference_output = adapter(query.detach(), key.detach(), value.detach())
    assert inference_output.shape == query.shape
    assert torch.isfinite(inference_output).all()
