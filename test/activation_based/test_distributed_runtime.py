# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_collect_reset_modules_and_reset_collected_modules():
    net = nn.Sequential(_ToyResetCounter(), nn.ReLU(), _ToyResetCounter())
    modules = collect_reset_modules(net)
    assert len(modules) == 2
    reset_collected_modules(modules)
    assert net[0].reset_calls == 1
    assert net[2].reset_calls == 1

def test_collect_reset_modules_ignores_non_callable_reset_attributes():
    net = nn.Sequential(_ToyNonCallableReset(), _ToyResetCounter())
    modules = collect_reset_modules(net)
    assert modules == (net[1],)

def test_prepare_metrics_classification_output_reduces_time_major_logits():
    logits = torch.randn(5, 4, 10)
    labels = torch.eye(10)[torch.tensor([0, 1, 2, 3])]
    prepared = prepare_classification_output(
        logits,
        labels,
        require_full_logits=True,
    )
    assert isinstance(prepared, PreparedModelOutput)
    torch.testing.assert_close(prepared.logits, logits.mean(dim=0))
    assert torch.equal(prepared.target, torch.tensor([0, 1, 2, 3]))

def test_prepare_metrics_classification_output_preserves_singleton_index_targets():
    logits = torch.randn(5, 4, 10)
    labels = torch.tensor([[0], [1], [2], [3]])
    prepared = prepare_classification_output(
        logits,
        labels,
        require_full_logits=True,
    )
    assert isinstance(prepared, PreparedModelOutput)
    torch.testing.assert_close(prepared.logits, logits.mean(dim=0))
    assert torch.equal(prepared.target, torch.tensor([0, 1, 2, 3]))

def test_prepare_metrics_classification_output_reduces_last_axis_targets():
    logits = torch.randn(5, 4, 10)
    labels = torch.eye(10)[torch.tensor([0, 1, 2, 3])].unsqueeze(1)
    prepared = prepare_classification_output(logits, labels)
    assert torch.equal(prepared.target, torch.tensor([0, 1, 2, 3]))

def test_prepare_metrics_classification_output_preserves_target_device():
    logits = torch.randn(2, 4)
    labels = torch.tensor([1, 3])
    prepared = prepare_classification_output(
        logits,
        labels,
        require_full_logits=True,
    )
    assert prepared.target.device == prepared.logits.device

def test_runtime_from_legacy_supports_eager_primitives():
    model = ToyDistributedSNN()
    runtime = SNNDistributedRuntime.from_legacy(
        kind="eager",
        model=model,
        mesh=None,
        analysis=None,
        mode="none",
    )
    assert runtime.plan is not None
    assert runtime.plan.mode == "none"
    optimizer = runtime.build_optimizer(
        optimizer_cls=torch.optim.SGD,
        lr=0.1,
    )
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(5, 2, 8)
    y = torch.tensor([0, 1])
    optimizer.zero_grad(set_to_none=True)
    outputs = runtime.model(x.float())
    outputs, labels = runtime.prepare_classification_output(outputs, y)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    runtime.reset_state()
    assert outputs.shape == (2, 4)
    assert labels.shape == (2,)
    assert torch.is_tensor(loss)

def test_runtime_from_legacy_preserves_mesh_shape_metadata():
    fake_mesh = SimpleNamespace(shape=(2, 2))
    runtime = SNNDistributedRuntime.from_legacy(
        kind="eager",
        model=ToyDistributedSNN(),
        mesh=fake_mesh,
        analysis=None,
        mode="fsdp2_tp",
    )
    assert runtime.plan is not None
    assert runtime.plan.topology.mesh_shape == (2, 2)

def test_runtime_from_legacy_uses_mode_for_single_axis_topology():
    fake_mesh = SimpleNamespace(shape=(2,))
    runtime = SNNDistributedRuntime.from_legacy(
        kind="eager",
        model=ToyDistributedSNN(),
        mesh=fake_mesh,
        analysis=None,
        mode="tp",
    )
    assert runtime.plan is not None
    assert runtime.plan.topology.dims == {"tp": 2}

def test_runtime_reset_state_uses_pipeline_stage_when_available():
    stage = _ToyResetCounter()
    pipeline_wrapper = SimpleNamespace(stage_module=stage)
    runtime = SNNDistributedRuntime.from_legacy(
        kind="pipeline",
        model=nn.Identity(),
        mesh=None,
        analysis=None,
        mode="pp",
        pipeline_runtime=pipeline_wrapper,
    )
    runtime.reset_state()
    assert stage.reset_calls == 1

def test_runtime_forward_loss_rejects_pipeline_runtime():
    stage = nn.Linear(3, 2)
    model = nn.Linear(5, 4)
    runtime = SNNDistributedRuntime.from_legacy(
        kind="pipeline",
        model=model,
        mesh=None,
        analysis=None,
        mode="pp",
        pipeline_runtime=SimpleNamespace(stage_module=stage),
    )
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1])
    with pytest.raises(NotImplementedError, match="does not execute pipeline runtimes"):
        runtime.forward_loss(criterion, x, y)

def test_runtime_prepare_classification_output_can_return_metadata():
    model = ToyDistributedSNN()
    runtime = SNNDistributedRuntime.from_legacy(
        kind="eager",
        model=model,
        mesh=None,
        analysis=None,
        mode="none",
    )
    logits = torch.randn(5, 2, 4)
    labels = torch.tensor([0, 1])
    prepared = runtime.prepare_classification_output(
        logits,
        labels,
        return_metadata=True,
    )
    assert isinstance(prepared, PreparedModelOutput)
    assert prepared.logits.shape == (2, 4)
    assert prepared.target.shape == (2,)

def test_runtime_prepare_dataloader_tolerates_missing_plan():
    runtime = SNNDistributedRuntime(
        kind="eager",
        model=nn.Identity(),
        mesh=None,
        analysis=None,
        plan=None,
        mode="none",
    )
    loader = runtime.prepare_dataloader(
        dataset=TensorDataset(torch.randn(2, 3), torch.tensor([0, 1])),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    assert len(list(loader)) == 2

def test_prepare_metrics_classification_output_squeezes_singleton_label_dim():
    logits = torch.randn(2, 4)
    labels = torch.tensor([[1], [3]])
    prepared = prepare_classification_output(
        logits,
        labels,
        require_full_logits=True,
    )
    assert torch.equal(prepared.target, torch.tensor([1, 3]))

def test_build_snn_optimizer_supports_zero_for_dp():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=0,
        )
        distributed_model, _, _ = configure_snn_distributed(model, config)
        optimizer = build_snn_optimizer(
            distributed_model,
            mode="dp",
            lr=1e-3,
            optimizer_sharding="zero",
        )
        assert "Zero" in type(optimizer).__name__
        x = torch.randn(3, 2, 8)
        y = torch.randn(3, 2, 4)
        _reset_net(distributed_model.module)
        loss = torch.nn.functional.mse_loss(distributed_model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

def test_build_snn_optimizer_rejects_zero_outside_dp():
    model = ToyDistributedSNN()
    with pytest.raises(ValueError, match="pure 'dp' mode only"):
        build_snn_optimizer(
            model,
            mode="tp",
            lr=1e-3,
            optimizer_sharding="zero",
        )
