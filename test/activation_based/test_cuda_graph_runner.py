import pytest
import torch

from spikingjelly.activation_based._cuda_graph import StaticCudaGraph
from spikingjelly.activation_based import functional, neuron


def test_static_cuda_graph_requires_warmup():
    with pytest.raises(ValueError, match="positive"):
        StaticCudaGraph(lambda x: x, warmup_steps=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_cuda_graph_replays_new_inputs_and_falls_back_for_new_shape():
    runner = StaticCudaGraph(lambda x: (x * 2, x + 1), warmup_steps=1)

    warmup = runner(torch.tensor([1.0, 2.0], device="cuda"))
    captured = runner(torch.tensor([3.0, 4.0], device="cuda"))
    captured_value = captured[0].clone()
    replayed = runner(torch.tensor([5.0, 6.0], device="cuda"))
    fallback = runner(torch.tensor([[7.0, 8.0]], device="cuda"))

    torch.testing.assert_close(warmup[0], torch.tensor([2.0, 4.0], device="cuda"))
    torch.testing.assert_close(captured_value, torch.tensor([6.0, 8.0], device="cuda"))
    torch.testing.assert_close(replayed[0], torch.tensor([10.0, 12.0], device="cuda"))
    torch.testing.assert_close(fallback[0], torch.tensor([[14.0, 16.0]], device="cuda"))
    assert runner.stats.warmup_runs == 1
    assert runner.stats.captures == 1
    assert runner.stats.replays == 1
    assert runner.stats.eager_fallbacks == 1


def _lif_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        neuron.LIFNode(tau=2.0, step_mode="m", backend="torch"),
    ).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_cuda_graph_preserves_lif_reset_semantics():
    torch.manual_seed(7)
    eager = _lif_model().eval()
    captured_model = _lif_model().eval()
    captured_model.load_state_dict(eager.state_dict())
    runner = StaticCudaGraph(captured_model, warmup_steps=1)

    for scale in (0.25, 0.5, 1.0):
        x = torch.randn(3, 2, 4, device="cuda") * scale
        expected = eager(x).clone()
        actual = runner(x).clone()
        functional.reset_net(eager)
        functional.reset_net(captured_model)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_cuda_graph_training_matches_eager_parameter_updates():
    torch.manual_seed(11)
    eager = _lif_model().train()
    captured_model = _lif_model().train()
    captured_model.load_state_dict(eager.state_dict())
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=0.1)
    captured_optimizer = torch.optim.SGD(captured_model.parameters(), lr=0.1)

    def captured_step(x):
        output = captured_model(x)
        loss = output.square().mean()
        loss.backward()
        return loss, output

    runner = StaticCudaGraph(captured_step, warmup_steps=1)
    for scale in (0.25, 0.5, 1.0):
        x = torch.randn(3, 2, 4, device="cuda") * scale
        eager_optimizer.zero_grad(set_to_none=False)
        eager_loss = eager(x).square().mean()
        eager_loss.backward()
        eager_optimizer.step()
        functional.reset_net(eager)

        captured_optimizer.zero_grad(set_to_none=False)
        captured_loss, _ = runner(x)
        captured_optimizer.step()
        functional.reset_net(captured_model)

        torch.testing.assert_close(captured_loss, eager_loss)
        for actual, expected in zip(
            captured_model.parameters(), eager.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected)
