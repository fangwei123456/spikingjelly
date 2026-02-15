import pytest
import torch
from spikingjelly.activation_based import neuron


@pytest.mark.parametrize("tau", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("detach_reset", [True, False])
@pytest.mark.parametrize("v_threshold", [1.0, 0.5])
@pytest.mark.parametrize("v_reset", [0.0, -0.2])
def test_lif(tau, detach_reset, v_threshold, v_reset):
    lif = neuron.LIFNode(
        tau, detach_reset=detach_reset, v_threshold=v_threshold, v_reset=v_reset, step_mode="m", backend="torch"
    )
    lif_triton = neuron.LIFNode(
        tau, detach_reset=detach_reset, v_threshold=v_threshold, v_reset=v_reset, step_mode="m", backend="triton"
    )

    # test forward and backward equality
    x = torch.randn(32, 128).cuda()
    x1, x2 = x.clone(), x.clone()
    x1.requires_grad_()
    x2.requires_grad_()
    out1 = lif(x1)
    out2 = lif_triton(x2)
    print(out1.mean().item(), out2.mean().item())
    assert torch.allclose(out1, out2, atol=1e-6), f"Forward outputs are not close: {out1} vs {out2}"

    out1.sum().backward()
    out2.sum().backward()
    print(x1.grad.mean().item(), x2.grad.mean().item())
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)


@pytest.mark.parametrize("decay_input", [True, False])
@pytest.mark.parametrize("detach_reset", [True, False])
@pytest.mark.parametrize("v_threshold", [1.0, 0.5])
@pytest.mark.parametrize("v_reset", [0.0, -0.2])
def test_plif(decay_input, detach_reset, v_threshold, v_reset):
    lif = neuron.ParametricLIFNode(
        decay_input=decay_input, detach_reset=detach_reset, v_threshold=v_threshold, v_reset=v_reset, step_mode="m", backend="torch"
    )
    lif_triton = neuron.ParametricLIFNode(
        decay_input=decay_input,detach_reset=detach_reset, v_threshold=v_threshold, v_reset=v_reset, step_mode="m", backend="triton"
    )

    # test forward and backward equality
    x = torch.randn(32, 128).cuda()
    x1, x2 = x.clone(), x.clone()
    x1.requires_grad_()
    x2.requires_grad_()
    out1 = lif(x1)
    out2 = lif_triton(x2)
    print(out1.mean().item(), out2.mean().item())
    assert torch.allclose(out1, out2, atol=1e-6), f"Forward outputs are not close: {out1} vs {out2}"

    out1.sum().backward()
    out2.sum().backward()
    print(x1.grad.mean().item(), x2.grad.mean().item())
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
    print(lif.w.grad.mean().item(), lif_triton.w.grad.mean().item())
    assert torch.allclose(lif.w.grad, lif_triton.w.grad, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])