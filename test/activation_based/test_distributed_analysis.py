# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_analyze_snn_distributed_capability_finds_state_and_linear_targets():
    model = ToyDistributedSNN()
    analysis = analyze_snn_distributed_capability(
        model, tensor_parallel_roots=["features"]
    )

    assert "features.1" in analysis.memory_module_names
    assert analysis.tensor_parallel_candidate_names == ("features.0", "features.2")
    assert analysis.unsupported_tensor_parallel_names == ()
    assert any(
        "Stateful neuron modules remain local/replicated" in note
        for note in analysis.notes
    )


def test_analyze_rejects_empty_tensor_parallel_root():
    with pytest.raises(ValueError, match="non-empty module paths"):
        analyze_snn_distributed_capability(
            ToyDistributedSNN(), tensor_parallel_roots=[""]
        )


def test_analyze_reports_unsupported_convs_outside_selected_roots():
    model = nn.Module()
    model.features = nn.Sequential(nn.Conv2d(2, 4, kernel_size=1))
    model.classifier = nn.Sequential(layer.Linear(4, 2))

    analysis = analyze_snn_distributed_capability(
        model, tensor_parallel_roots=["classifier"]
    )

    assert analysis.tensor_parallel_candidate_names == ("classifier.0",)
    assert analysis.unsupported_tensor_parallel_names == ("features.0",)


def test_adapter_registry_lists_known_families():
    names = list_adapters()
    assert "cifar10dvs_vgg" in names
    assert "spikformer" in names


def test_resolve_adapter_for_known_models():
    vgg = CIFAR10DVSVGG(dropout=0.0, backend="torch")
    assert resolve_adapter(vgg, None) is not None
    spikformer = spikformer_ti(
        T=2, img_size_h=64, img_size_w=64, num_classes=11, backend="torch"
    )
    assert resolve_adapter(spikformer, None) is not None


def test_infer_model_family_unwraps_module_attribute():
    wrapped = SimpleNamespace(module=CIFAR10DVSVGG(dropout=0.0, backend="torch"))
    from spikingjelly.activation_based.distributed.adapters.base import (
        infer_model_family,
    )

    assert infer_model_family(wrapped) == "cifar10dvs_vgg"


def test_infer_model_family_ignores_non_module_module_attribute():
    wrapped = SimpleNamespace(module="not-a-module")
    from spikingjelly.activation_based.distributed.adapters.base import (
        infer_model_family,
    )

    assert infer_model_family(wrapped) is None


def test_analyze_stays_generic_without_model_family_specific_adapter():
    vgg = CIFAR10DVSVGG(dropout=0.0, backend="torch")
    generic = analyze(vgg)
    direct = analyze_snn_distributed_capability(vgg)
    assert generic.memory_module_names == direct.memory_module_names
    assert (
        generic.tensor_parallel_candidate_names
        == direct.tensor_parallel_candidate_names
    )
