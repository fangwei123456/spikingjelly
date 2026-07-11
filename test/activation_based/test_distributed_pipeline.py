# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_cifar10dvs_vgg_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    x = torch.randn(1, 2, 2, 48, 48)
    pipeline_module = _build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=x,
    ).eval()
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_module_matches_baseline():
    torch.manual_seed(0)
    baseline = spikformer_ti(
        T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
    ).eval()
    x = torch.randn(2, 3, 64, 64)
    pipeline_module = _build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=3,
        example_input=x,
    ).eval()
    functional.reset_net(baseline)
    reference = baseline(x)
    functional.reset_net(pipeline_module)
    result = pipeline_module(x)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_measure_module_cost_uses_autograd_inside_no_grad():
    module = nn.Linear(3, 2)
    x = torch.randn(4, 3)

    with torch.no_grad():
        _output, cost = _measure_module_cost(module, x)

    assert cost > 0
    assert module.weight.grad is None


def test_cifar10dvs_vgg_pipeline_runtime_supports_interleaved_single_rank():
    with single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        x = torch.randn(2, 2, 2, 48, 48)
        runtime = configure_cifar10dvs_vgg_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="interleaved",
            pp_virtual_stages=2,
        )
        assert runtime.schedule_kind == "interleaved"
        assert runtime.virtual_pipeline_size == 2
        assert len(runtime.stage_modules) == 2


def test_spikformer_pipeline_runtime_supports_zero_bubble_single_rank():
    with single_rank_process_group():
        model = spikformer_ti(
            T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
        ).eval()
        x = torch.randn(2, 3, 64, 64)
        runtime = configure_spikformer_pipeline(
            copy.deepcopy(model),
            example_input=x,
            device=torch.device("cpu"),
            n_microbatches=2,
            pp_schedule="zero_bubble",
            pp_virtual_stages=2,
            pp_delay_wgrad=True,
        )
        assert runtime.schedule_kind == "zero_bubble"
        assert runtime.delayed_wgrad is True
        assert len(runtime.stage_modules) == 2


def test_recommend_pipeline_memopt_stages_prefers_heavy_stages():
    selected = recommend_pipeline_memopt_stages(
        (1.0, 8.0, 3.0, 7.0), stage_budget_ratio=0.5
    )
    assert selected == (1, 3)


def test_apply_pipeline_stage_memopt_only_wraps_selected_heavy_stage():
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )
    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
    )
    assert applied is True
    assert optimize_ms >= 0.0
    assert runtime.memopt_selected_stage_indices == (0,)
    assert isinstance(runtime.stage_module.inner.features[0], GCContainer)


def test_apply_pipeline_stage_memopt_supports_legacy_memopt_signature(monkeypatch):
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("cpu"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )

    import spikingjelly.activation_based.memopt as memopt

    calls = {"count": 0}

    def fake_memory_optimization(
        module, target_types, dummy_input, compress_x, level, verbose
    ):
        calls["count"] += 1
        return module

    monkeypatch.setattr(memopt, "memory_optimization", fake_memory_optimization)

    runtime, optimize_ms, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
        use_plan_cache=True,
    )

    assert applied is True
    assert optimize_ms >= 0.0
    assert calls["count"] == 1


def test_apply_pipeline_stage_memopt_moves_dummy_input_to_runtime_device(monkeypatch):
    torch.manual_seed(0)
    model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    stage = _CIFAR10DVSVGGPipelineStage(
        feature_modules=[copy.deepcopy(model.features[0])],
        classifier=None,
        transpose_input=True,
    ).eval()
    wrapped_stage = _MicrobatchResetStage(stage)
    runtime = SNNPipelineRuntime(
        schedule=None,
        stage_module=wrapped_stage,
        stage_modules=(wrapped_stage,),
        local_stage_indices=(0,),
        stage_index=0,
        num_stages=2,
        device=torch.device("meta"),
        n_microbatches=2,
        model_family="cifar10dvs_vgg",
        split_points=("stages.1",),
        stage_costs=(10.0, 1.0),
        stage_input_examples=(torch.randn(1, 2, 2, 48, 48),),
    )

    import spikingjelly.activation_based.memopt as memopt

    def fake_memory_optimization(
        module, target_types, dummy_input, compress_x, level, verbose
    ):
        assert next(module.parameters()).device.type == "meta"
        assert dummy_input[0].device.type == "meta"
        return module

    monkeypatch.setattr(memopt, "memory_optimization", fake_memory_optimization)

    _, _, applied = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=1,
        compress_x=False,
        stage_budget_ratio=0.5,
        use_plan_cache=False,
    )

    assert applied is True


def test_parse_pipeline_layout_validates_counts():
    counts = parse_pipeline_layout("1|2|3", 3, 6)
    assert counts == (1, 2, 3)
    with pytest.raises(ValueError, match="requires 6 units"):
        parse_pipeline_layout("1|2|2", 3, 6)


def test_resolve_pipeline_schedule_kind_rules():
    assert resolve_pipeline_schedule_kind("auto", 1, False) == "1f1b"
    assert resolve_pipeline_schedule_kind("auto", 2, False) == "interleaved"
    assert resolve_pipeline_schedule_kind("auto", 2, True) == "zero_bubble"
    with pytest.raises(ValueError, match="requires pp_virtual_stages >= 2"):
        resolve_pipeline_schedule_kind("interleaved", 1, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        resolve_pipeline_schedule_kind("gpipe", 2, False)
    with pytest.raises(ValueError, match="does not support pp_virtual_stages=2"):
        resolve_pipeline_schedule_kind("1f1b", 2, False)


def test_make_pipeline_outputs_contiguous_clones_views():
    base = torch.randn(2, 3, 4)
    view = base.transpose(0, 1)
    out = _make_pipeline_outputs_contiguous(view)
    torch.testing.assert_close(out, view)
    assert out.data_ptr() != view.data_ptr()


def test_cifar_pipeline_transposes_on_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
    example = torch.randn(1, 2, 2, 48, 48)
    pipeline = _build_cifar10dvs_vgg_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(list(baseline.features.children())) + 1),
    )
    assert pipeline.stages[0].transpose_input is False
    assert pipeline.stages[1].transpose_input is True
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)


def test_spikformer_pipeline_attaches_patch_embed_to_first_non_empty_stage():
    torch.manual_seed(0)
    baseline = spikformer_ti(
        T=2,
        img_size_h=64,
        img_size_w=64,
        num_classes=11,
        backend="torch",
    ).eval()
    example = torch.randn(1, 3, 64, 64)
    pipeline = _build_spikformer_pipeline_module(
        copy.deepcopy(baseline),
        num_logical_stages=2,
        example_input=example,
        layout_counts=(0, len(baseline.blocks) + 2),
    )
    assert pipeline.stages[0].patch_embed is None
    assert pipeline.stages[1].patch_embed is not None
    functional.reset_net(baseline)
    reference = baseline(example)
    functional.reset_net(pipeline)
    result = pipeline(example)
    torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
