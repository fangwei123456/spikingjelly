# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_build_eager_config_expands_hybrid_policy():
    config = build_eager_config(
        mode="fsdp2_tp",
        device_type="cpu",
        mesh_shape=(2, 2),
        tp_mesh_dim=1,
        dp_mesh_dim=0,
        policy=EagerParallelPolicy(
            linear_tensor_parallel_roots=("classifier",),
            conv_tensor_parallel_roots=("features",),
            fsdp_shard_roots=("features", "classifier"),
            fsdp2_tp_shard_roots=("features",),
            fsdp2_tp_shard_module_root=False,
        ),
    )

    assert config.enable_fsdp2 is True
    assert config.enable_data_parallel is False
    assert config.auto_tensor_parallel is True
    assert config.tensor_parallel_roots == ["classifier"]
    assert config.experimental_conv_tensor_parallel is True
    assert config.conv_tensor_parallel_roots == ["features"]
    assert config.fsdp_shard_roots == ["features"]
    assert config.fsdp_shard_module_root is False
    assert config.tp_mesh_dim == 1
    assert config.dp_mesh_dim == 0


def test_build_eager_config_fsdp2_tp_falls_back_to_root_shard_flag():
    config = build_eager_config(
        mode="fsdp2_tp",
        device_type="cpu",
        mesh_shape=(1, 1),
        tp_mesh_dim=1,
        dp_mesh_dim=0,
        policy=EagerParallelPolicy(
            linear_tensor_parallel_roots=("classifier",),
            fsdp_shard_roots=("features", "classifier"),
            fsdp_shard_module_root=True,
        ),
    )

    assert config.fsdp_shard_roots == ["features", "classifier"]
    assert config.fsdp_shard_module_root is True


def test_build_eager_config_allows_disabling_linear_tp_only():
    config = build_eager_config(
        mode="tp",
        device_type="cpu",
        mesh_shape=(4,),
        policy=EagerParallelPolicy(
            linear_tensor_parallel_roots=("classifier",),
            conv_tensor_parallel_roots=("features",),
        ),
        enable_linear_tensor_parallel=False,
    )

    assert config.auto_tensor_parallel is False
    assert config.tensor_parallel_roots is None
    assert config.experimental_conv_tensor_parallel is True
    assert config.conv_tensor_parallel_roots == ["features"]


def test_configure_snn_distributed_conv_only_tp_does_not_build_linear_plan(
    monkeypatch: pytest.MonkeyPatch,
):
    import spikingjelly.activation_based.distributed.execution as execution

    with single_rank_process_group():

        def _unexpected_auto_plan(*args, **kwargs):
            raise AssertionError("linear tensor parallel plan should not be built")

        monkeypatch.setattr(
            execution, "auto_build_tensor_parallel_plan", _unexpected_auto_plan
        )
        model = ToyDistributedSNN()
        configured_model, mesh, _ = configure_snn_distributed(
            model,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                experimental_conv_tensor_parallel=True,
                conv_tensor_parallel_roots=["features"],
                enable_data_parallel=False,
            ),
        )

        assert configured_model is model
        assert mesh is not None


def test_apply_returns_unified_runtime_single_rank():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        analysis = analyze(model, roots=["features"])
        distributed_plan = plan(
            analysis=analysis,
            objective="speed",
            topology={"dp": 1},
            backend="inductor",
            batch_size=4,
            model_family="toy_snn",
        )
        runtime = apply(model=model, plan=distributed_plan, device_type="cpu")
        assert isinstance(runtime, SNNDistributedRuntime)
        assert runtime.kind == "eager"
        assert runtime.mesh is None  # mode is "none", no parallel strategy active
        assert runtime.plan.mode == distributed_plan.mode


def test_configure_snn_distributed_mesh_shape_alone_without_strategy_returns_none_mesh():
    """mesh_shape alone doesn't trigger mesh creation — a parallel strategy must also be enabled."""
    with single_rank_process_group():
        model = ToyDistributedSNN()
        configured_model, mesh, analysis = configure_snn_distributed(
            model,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                enable_data_parallel=False,
                enable_fsdp2=False,
            ),
        )
        assert configured_model is model
        assert mesh is None
        assert isinstance(analysis, distributed_dtensor.SNNDistributedAnalysis)


def test_configure_snn_distributed_noop_does_not_require_device_mesh():
    model = ToyDistributedSNN()
    configured_model, mesh, analysis = configure_snn_distributed(
        model,
        SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=None,
            auto_tensor_parallel=False,
            enable_data_parallel=False,
            enable_fsdp2=False,
        ),
    )
    assert configured_model is model
    assert mesh is None
    assert isinstance(analysis, distributed_dtensor.SNNDistributedAnalysis)


def test_apply_rejects_device_mesh_world_size_mismatch():
    model = ToyDistributedSNN()
    analysis = analyze(model, roots=["features"])
    distributed_plan = plan(
        analysis=analysis,
        objective="speed",
        topology={"dp": 1},
        backend="inductor",
        batch_size=4,
        model_family="toy_snn",
    )

    class FakeMesh:
        def __init__(self):
            self.mesh = torch.arange(2)

    with pytest.raises(ValueError, match="device_mesh spans 2 ranks"):
        apply(
            model=model,
            plan=distributed_plan,
            device_type="cpu",
            device_mesh=FakeMesh(),
        )


def test_apply_rejects_pipeline_mode_without_example_input():
    distributed_plan = SNNDistributedPlan(
        mode="pp",
        objective="capacity",
        topology=SNNDistributedTopology.from_mapping({"pp": 2}),
        model_family="toy_snn",
        backend="inductor",
        batch_size=8,
        optimizer_strategy="none",
        memopt_level=1,
        rationale=(),
        notes=(),
        experimental_features=DistributedFeatureSet(),
    )
    with pytest.raises(NotImplementedError, match="Pipeline parallelism"):
        apply(model=ToyDistributedSNN(), plan=distributed_plan, device_type="cpu")


def test_configure_snn_distributed_supports_data_parallel_only():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=0,
        )
        distributed_model, mesh, _ = configure_snn_distributed(model, config)

        assert isinstance(distributed_model, DistributedDataParallel)
        assert mesh.ndim == 1

        x = torch.randn(3, 2, 8)
        _reset_net(distributed_model.module)
        y = distributed_model(x)
        assert y.shape == (3, 2, 4)


def test_configure_snn_distributed_rejects_ddp_plus_tp():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=True,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        with pytest.raises(NotImplementedError, match="FSDP2 \\+ TP"):
            configure_snn_distributed(model, config)


def test_configure_snn_distributed_rejects_ddp_plus_experimental_conv_tp():
    with single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            auto_tensor_parallel=False,
            experimental_conv_tensor_parallel=True,
            conv_tensor_parallel_roots=["features"],
            enable_data_parallel=True,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        with pytest.raises(NotImplementedError, match="FSDP2 \\+ TP"):
            configure_snn_distributed(model, config)
        assert "ChannelShardConv2d" not in type(model.features[0].proj_bn[-2]).__name__


def test_configure_snn_distributed_requires_dp_mesh_dim_for_multidim_data_parallel():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=None,
        )
        with pytest.raises(ValueError, match="dp_mesh_dim must be specified"):
            configure_snn_distributed(model, config)


def test_configure_snn_distributed_rejects_out_of_range_mesh_dims():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        with pytest.raises(ValueError, match="tp_mesh_dim=2 is out of range"):
            configure_snn_distributed(
                model,
                SNNDistributedConfig(
                    device_type="cpu",
                    mesh_shape=(1, 1),
                    tensor_parallel_roots=["features"],
                    auto_tensor_parallel=True,
                    tp_mesh_dim=2,
                    dp_mesh_dim=0,
                ),
            )
        with pytest.raises(ValueError, match="dp_mesh_dim=2 is out of range"):
            configure_snn_distributed(
                model,
                SNNDistributedConfig(
                    device_type="cpu",
                    mesh_shape=(1, 1),
                    auto_tensor_parallel=False,
                    enable_data_parallel=True,
                    tp_mesh_dim=1,
                    dp_mesh_dim=2,
                ),
            )


def test_configure_snn_distributed_rejects_overlapping_tp_and_dp_mesh_dims():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        with pytest.raises(ValueError, match="tp_mesh_dim and dp_mesh_dim"):
            configure_snn_distributed(
                model,
                SNNDistributedConfig(
                    device_type="cpu",
                    mesh_shape=(1, 1),
                    enable_fsdp2=True,
                    tensor_parallel_roots=["features"],
                    auto_tensor_parallel=True,
                    tp_mesh_dim=1,
                    dp_mesh_dim=1,
                ),
            )


def test_configure_snn_distributed_allows_same_dim_for_data_parallel_only():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        distributed_model, mesh, _ = configure_snn_distributed(
            model,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                auto_tensor_parallel=False,
                enable_data_parallel=True,
                tp_mesh_dim=0,
                dp_mesh_dim=0,
            ),
        )

        assert isinstance(distributed_model, DistributedDataParallel)
        assert mesh.ndim == 1


def test_high_level_cifar10dvs_vgg_helper():
    with single_rank_process_group():
        model = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        _distributed_model, mesh, analysis = configure_snn_distributed(
            model,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1,),
                tensor_parallel_roots=["classifier"],
                auto_tensor_parallel=True,
                experimental_conv_tensor_parallel=True,
                conv_tensor_parallel_roots=["features"],
                enable_data_parallel=False,
            ),
        )
        assert mesh.ndim == 1
        assert analysis.tensor_parallel_candidate_names == ("classifier.0",)


def test_cifar10dvs_vgg_fsdp2_single_rank_smoke():
    with single_rank_process_group():
        torch.manual_seed(0)
        candidate = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        analysis = analyze(candidate)
        distributed_plan = plan(
            analysis=analysis,
            objective="memory",
            topology={"dp": 1},
            backend="torch",
            batch_size=1,
            model_family="cifar10dvs_vgg",
            mode="fsdp2",
        )
        runtime = apply(model=candidate, plan=distributed_plan, device_type="cpu")
        assert runtime.mesh is not None
        assert runtime.analysis is not None
        assert runtime.analysis.tensor_parallel_roots == ("classifier",)


def test_cifar10dvs_vgg_fsdp2_tp_helper_single_rank():
    with single_rank_process_group():
        torch.manual_seed(0)
        baseline = CIFAR10DVSVGG(dropout=0.0, backend="torch").eval()
        candidate = copy.deepcopy(baseline).eval()
        distributed_model, mesh, _ = configure_snn_distributed(
            candidate,
            SNNDistributedConfig(
                device_type="cpu",
                mesh_shape=(1, 1),
                enable_fsdp2=True,
                fsdp_shard_roots=["features"],
                fsdp_shard_module_root=False,
                tensor_parallel_roots=["classifier"],
                auto_tensor_parallel=True,
                experimental_conv_tensor_parallel=True,
                conv_tensor_parallel_roots=["features"],
                tp_mesh_dim=1,
                dp_mesh_dim=0,
            ),
        )
        x = torch.randn(1, 2, 2, 48, 48)
        _reset_net(baseline)
        reference = baseline(x)
        _reset_net(distributed_model)
        result = materialize_dtensor_output(distributed_model(x))
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
        assert mesh.ndim == 2
