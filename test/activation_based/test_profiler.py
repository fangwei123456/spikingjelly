import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based.profiler import (
    CategoryMemoryProfiler,
    LayerWiseMemoryProfiler,
    LayerWiseFPCUDATimeProfiler,
)


def _create_test_model():
    net = nn.Sequential(
        layer.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        neuron.IFNode(),
        layer.AdaptiveAvgPool2d((1, 1)),
        layer.Flatten(),
        layer.Linear(64, 10),
        neuron.IFNode(),
    )
    functional.set_step_mode(net, "m")
    return net


def test_context_manager_basic():
    net = _create_test_model().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    with CategoryMemoryProfiler((net,), (optimizer,), log_path="test.prof.txt") as prof:
        x = torch.randn(5, 32, 1, 28, 28).cuda()
        y = net(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        functional.reset_net(net)
        results = prof.export()
    return results


def test_layer_wise_profiling():
    net = _create_test_model().cuda()
    with LayerWiseMemoryProfiler(
        (net,),
        model_names=("test_net",),
        search_mode=("submodules",),
        instances=(nn.Module,),
        device="cuda",
        log_path="test.prof.txt",
        data_path="test.prof.pt",
    ) as prof:
        x = torch.randn(15, 6, 1, 28, 28).cuda()
        y = net(x)
        loss = y.sum()
        loss.backward()
        functional.reset_net(net)
    results = prof.export(output=True)


def test_time_profiling():
    net = _create_test_model().cuda()
    with LayerWiseFPCUDATimeProfiler(
        (net,),
        model_names=("test_net",),
        search_mode=("submodules",),
        instances=(nn.Module,),
        warmup=5,
        log_path="test.prof.txt",
    ) as prof:
        net.eval()
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(7, 8, 1, 28, 28).cuda()
                _ = net(x)
                functional.reset_net(net)
    results = prof.export(output=True)


def test_exception_safety():
    net = _create_test_model().cuda()

    with pytest.raises(RuntimeError):
        with LayerWiseFPCUDATimeProfiler(
            (net,),
            model_names=("test_net",),
            search_mode=("submodules",),
            instances=(nn.Module,),
            warmup=5,
            log_path="test.prof.txt",
        ) as prof:
            x = torch.randn(8, 32, 1, 28, 28).cuda()
            y = net(x)
            raise RuntimeError("Simulate an exception.")
            loss = y.sum()
            loss.backward()
    print(f"len(prof.hooks)={len(prof.hooks)}")
    assert len(prof.hooks) == 0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit(0)
    test_context_manager_basic()
    test_layer_wise_profiling()
    test_time_profiling()
    test_exception_safety()
    print("Done!")
