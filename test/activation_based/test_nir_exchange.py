import re
from pathlib import Path

import pytest

nir = pytest.importorskip("nir")
pytest.importorskip("nirtorch")

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.activation_based import base, functional, layer, neuron, nir_exchange


@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_existing_nodes_round_trip(tmp_path: Path, step_mode: str):
    torch.manual_seed(0)
    net = nn.Sequential(
        layer.Conv2d(2, 3, 3, padding=1),
        neuron.IFNode(v_threshold=0.5, v_reset=0.0),
        layer.AvgPool2d(2),
        layer.Flatten(),
        layer.Linear(3 * 4 * 4, 4),
        neuron.ParametricLIFNode(init_tau=2.0, v_reset=0.0),
    )
    functional.set_step_mode(net, step_mode)
    shape = (2, 2, 8, 8) if step_mode == "s" else (3, 2, 2, 8, 8)
    x = torch.rand(shape)
    path = tmp_path / "model.nir"

    nir_exchange.export_to_nir(net, x, path)
    restored = nir_exchange.import_from_nir(path, step_mode=step_mode)

    functional.reset_net(net)
    expected = net(x)
    actual, _ = restored(x)
    assert torch.equal(actual, expected)

    restored_conv = next(m for m in restored.modules() if isinstance(m, layer.Conv2d))
    assert restored_conv.in_channels == 2
    assert restored_conv.out_channels == 3


@pytest.mark.parametrize("step_mode", ["s", "m"])
def test_conv1d_cuba_lif_round_trip(tmp_path: Path, step_mode: str):
    torch.manual_seed(1)
    net = nn.Sequential(
        layer.Conv1d(2, 3, 3, padding=1),
        neuron.CUBALIFNode(
            c_decay=0.5,
            v_decay=0.75,
            v_threshold=0.5,
            v_reset=0.0,
        ),
    )
    functional.set_step_mode(net, step_mode)
    shape = (2, 2, 8) if step_mode == "s" else (4, 2, 2, 8)
    x = torch.rand(shape)

    path = tmp_path / "conv1d-cuba-lif.nir"
    nir_exchange.export_to_nir(net, x, path)
    graph = nir.read(path)
    restored = nir_exchange.import_from_nir(path, step_mode=step_mode)

    functional.reset_net(net)
    expected = net(x)
    actual, _ = restored(x)
    assert torch.equal(actual, expected)
    assert any(isinstance(node, nir.Conv1d) for node in graph.nodes.values())
    assert any(isinstance(node, nir.CubaLIF) for node in graph.nodes.values())


def test_linear_without_bias_round_trip():
    net = nn.Sequential(layer.Linear(3, 2, bias=False))
    x = torch.rand(4, 3)

    graph = nir_exchange.export_to_nir(net, x)
    restored = nir_exchange.import_from_nir(graph)

    actual, _ = restored(x)
    assert torch.equal(actual, net(x))
    assert any(isinstance(node, nir.Linear) for node in graph.nodes.values())


def test_imported_state_is_explicit():
    dt = 1e-4
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.array([2])}),
            "if": nir.IF(
                r=np.full(2, 1.0 / dt),
                v_threshold=np.ones(2),
                v_reset=np.zeros(2),
            ),
            "output": nir.Output({"output": np.array([2])}),
        },
        edges=[("input", "if"), ("if", "if"), ("if", "output")],
        type_check=False,
    )
    model = nir_exchange.import_from_nir(graph, dt=dt, dtype=torch.float64)
    x = torch.full((1, 2), 0.6, dtype=torch.float64)

    first, state = model(x)
    continued, _ = model(x, state)
    fresh, _ = model(x)

    assert torch.count_nonzero(first) == 0
    assert torch.count_nonzero(continued) == continued.numel()
    assert torch.equal(fresh, first)
    assert first.dtype == torch.float64
    assert list(base.memories(model)) == [0.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_plif_cuda_round_trip():
    net = nn.Sequential(neuron.ParametricLIFNode(v_reset=0.0)).cuda()
    x = torch.rand(2, 3, device="cuda")

    graph = nir_exchange.export_to_nir(net, x)
    restored = nir_exchange.import_from_nir(graph, device="cuda")

    functional.reset_net(net)
    expected = net(x)
    actual, _ = restored(x)
    assert torch.equal(actual, expected)


def test_export_preserves_memories():
    net = nn.Sequential(neuron.LIFNode(tau=2.0, v_reset=0.0))
    x = torch.full((1, 2), 0.75)
    net(x)
    before = [value.clone() for value in base.memories(net)]

    nir_exchange.export_to_nir(net, x)

    after = list(base.memories(net))
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, after, strict=True)
    )


def test_export_restores_memories_when_tracing_fails():
    class FailingNet(base.MemoryModule):
        def __init__(self):
            super().__init__()
            self.register_memory("trace_count", torch.tensor(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.trace_count.add_(1)
            raise RuntimeError("trace failed")

    net = FailingNet()
    with pytest.raises(RuntimeError, match="trace failed"):
        nir_exchange.export_to_nir(net, torch.rand(1, 2))
    assert torch.equal(net.trace_count, torch.tensor(0))


def test_rejects_unrepresentable_models():
    with pytest.raises(NotImplementedError, match="soft reset"):
        nir_exchange.export_to_nir(
            nn.Sequential(neuron.LIFNode(v_reset=None)), torch.rand(1, 2)
        )

    with pytest.raises(NotImplementedError, match="grouped convolutions"):
        nir_exchange.export_to_nir(
            nn.Sequential(layer.Conv2d(2, 2, 3, groups=2)),
            torch.rand(1, 2, 5, 5),
        )

    with pytest.raises(NotImplementedError, match="AvgPool2d"):
        nir_exchange.export_to_nir(
            nn.Sequential(layer.AvgPool2d(2, padding=1, count_include_pad=False)),
            torch.rand(1, 2, 5, 5),
        )


def test_rejects_invalid_nir_parameters():
    graph = nir.NIRGraph.from_list(
        nir.IF(
            r=np.full(2, 1e4),
            v_threshold=np.array([1.0, 2.0]),
            v_reset=np.zeros(2),
        )
    )
    with pytest.raises(ValueError, match="uniform"):
        nir_exchange.import_from_nir(graph)

    heterogeneous_r = nir.NIRGraph.from_list(
        nir.IF(
            r=np.array([1e4, 1e4 + 0.01]),
            v_threshold=np.ones(2),
            v_reset=np.zeros(2),
        )
    )
    with pytest.raises(ValueError, match=re.escape("nir.IF.r must be uniform")):
        nir_exchange.import_from_nir(heterogeneous_r)

    incompatible_r = nir.NIRGraph.from_list(
        nir.IF(
            r=np.full(2, 1e4 + 0.05),
            v_threshold=np.ones(2),
            v_reset=np.zeros(2),
        )
    )
    with pytest.raises(ValueError, match=re.escape("nir.IF.r must equal")):
        nir_exchange.import_from_nir(incompatible_r)

    invalid_lif = nir.NIRGraph.from_list(
        nir.LIF(
            tau=np.full(2, 0.5e-4),
            r=np.ones(2),
            v_leak=np.zeros(2),
            v_threshold=np.ones(2),
            v_reset=np.zeros(2),
        )
    )
    with pytest.raises(ValueError, match="greater than 1"):
        nir_exchange.import_from_nir(invalid_lif)

    infinite_lif = nir.NIRGraph.from_list(
        nir.LIF(
            tau=np.full(2, np.inf),
            r=np.full(2, np.inf),
            v_leak=np.zeros(2),
            v_threshold=np.ones(2),
            v_reset=np.zeros(2),
        )
    )
    with pytest.raises(ValueError, match="finite"):
        nir_exchange.import_from_nir(infinite_lif)

    with pytest.raises(ValueError, match="positive"):
        nir_exchange.import_from_nir(graph, dt=0.0)
    with pytest.raises(ValueError, match="step_mode"):
        nir_exchange.import_from_nir(graph, step_mode="invalid")
    with pytest.raises(ValueError, match="positive"):
        nir_exchange.export_to_nir(nn.Identity(), torch.rand(1, 2), dt=0.0)

    recurrent = nir.NIRGraph(
        nodes={"lif": graph.nodes["if"]},
        edges=[("lif", "lif")],
        type_check=False,
    )
    with pytest.raises(NotImplementedError, match="step_mode='s'"):
        nir_exchange.import_from_nir(recurrent, step_mode="m")


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("r", np.array([4.0, 4.0 + 1e-6])),
        ("w_in", np.array([2.0, 2.0 + 1e-6])),
        ("v_leak", np.array([0.0, 1e-12])),
    ],
)
def test_rejects_heterogeneous_cuba_lif_parameters(field: str, values: np.ndarray):
    node = nir.CubaLIF(
        tau_syn=np.full(2, 2e-4),
        tau_mem=np.full(2, 4e-4),
        r=np.full(2, 4.0),
        v_leak=np.zeros(2),
        v_threshold=np.ones(2),
        v_reset=np.zeros(2),
        w_in=np.full(2, 2.0),
    )
    setattr(node, field, values)

    with pytest.raises(
        ValueError, match=re.escape(f"nir.CubaLIF.{field} must be uniform")
    ):
        nir_exchange.import_from_nir(nir.NIRGraph.from_list(node))


def test_rejects_incompatible_cuba_lif_parameters():
    incompatible = nir.CubaLIF(
        tau_syn=np.full(2, 2e-4),
        tau_mem=np.full(2, 4e-4),
        r=np.full(2, 4.0 + 2e-5),
        v_leak=np.zeros(2),
        v_threshold=np.ones(2),
        v_reset=np.zeros(2),
        w_in=np.full(2, 2.0),
    )
    with pytest.raises(ValueError, match=re.escape("nir.CubaLIF.r must equal")):
        nir_exchange.import_from_nir(nir.NIRGraph.from_list(incompatible))

    non_finite = nir.CubaLIF(
        tau_syn=np.full(2, np.inf),
        tau_mem=np.full(2, 4e-4),
        r=np.full(2, 4.0),
        v_leak=np.zeros(2),
        v_threshold=np.ones(2),
        v_reset=np.zeros(2),
        w_in=np.full(2, np.inf),
    )
    with pytest.raises(ValueError, match="finite"):
        nir_exchange.import_from_nir(nir.NIRGraph.from_list(non_finite))
