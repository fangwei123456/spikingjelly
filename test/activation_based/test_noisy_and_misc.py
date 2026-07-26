import torch

from spikingjelly.activation_based import quantize
from spikingjelly.activation_based.functional.misc import (
    delay,
    first_spike_index,
    redundant_one_hot,
)
from spikingjelly.activation_based.layer.container import StepModeContainer
from spikingjelly.activation_based.layer.dropout import Dropout
from spikingjelly.activation_based.neuron.noisy import (
    NoisyCUBALIFNode,
    NoisyILCCUBALIFNode,
    NoisyNonSpikingIFNode,
)
from spikingjelly.activation_based.neuron.lif_variants import (
    CUBALIFNode,
    GatedLIFNode,
    KLIFNode,
)


def test_loaded_zero_noise_uses_the_same_cubalif_dynamics_as_evaluation():
    training_node = NoisyCUBALIFNode(num_node=3, is_training=True, T=4)
    evaluation_node = NoisyCUBALIFNode(num_node=3, is_training=False, T=4)
    training_node.load_colored_noise(torch.zeros(2, 4, 6))
    x = torch.randn(4, 2, 3)

    torch.testing.assert_close(training_node(x.clone()), evaluation_node(x.clone()))


def test_zero_recurrent_connection_reduces_ilc_node_to_cubalif_node():
    ilc = NoisyILCCUBALIFNode(
        act_dim=2,
        dec_pop_dim=3,
        is_training=False,
        T=4,
    )
    plain = NoisyCUBALIFNode(num_node=6, is_training=False, T=4, v_threshold=1.0)
    with torch.no_grad():
        ilc.conn.weight.zero_()
        ilc.conn.bias.zero_()
    x = torch.randn(4, 2, 6)

    torch.testing.assert_close(ilc(x.clone()), plain(x.clone()))


def test_nonspiking_max_abs_decode_selects_the_largest_magnitude_membrane():
    node = NoisyNonSpikingIFNode(
        num_node=1, is_training=False, T=3, decode="max-abs-mem"
    )
    x = torch.tensor([[[-2.0]], [[3.0]], [[-5.0]]])

    torch.testing.assert_close(node(x), torch.tensor([[-4.0]]))


def test_neuron_parameters_keep_float32_when_default_dtype_changes():
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        nodes = [GatedLIFNode(T=3), NoisyNonSpikingIFNode(num_node=2, T=3)]
    finally:
        torch.set_default_dtype(default_dtype)

    assert all(
        parameter.dtype == torch.float32
        for node in nodes
        for parameter in node.parameters()
    )


def test_vectorized_misc_helpers_preserve_values_and_gradients():
    labels = torch.tensor([0, 2])
    torch.testing.assert_close(
        redundant_one_hot(labels, num_classes=3, n=2),
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ),
    )

    spikes = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert torch.equal(
        first_spike_index(spikes),
        torch.tensor([[False, True, False, False], [False, False, False, False]]),
    )

    x = torch.arange(6.0).view(3, 2).requires_grad_()
    shifted = delay(x, 1)
    torch.testing.assert_close(
        shifted, torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    )
    shifted.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    )
    torch.testing.assert_close(delay(x.detach(), 0), x.detach())
    torch.testing.assert_close(delay(x.detach(), 4), torch.zeros_like(x))


def test_dropout_reuses_one_mask_across_time_until_reset():
    torch.manual_seed(1)
    dropout = Dropout(p=0.5, step_mode="m")
    x = torch.ones(4, 3, 5)

    output = dropout(x)
    for t in range(1, x.shape[0]):
        torch.testing.assert_close(output[t], output[0])
    dropout.reset()
    assert dropout.mask is None


def test_step_mode_container_constructs_and_flattens_stateless_sequences():
    container = StepModeContainer(False, "m", torch.nn.Linear(3, 2))
    x = torch.randn(4, 5, 3)

    torch.testing.assert_close(
        container(x),
        container[0](x.flatten(0, 1)).view(4, 5, 2),
    )


def test_quantizers_keep_their_straight_through_gradients():
    x = torch.tensor([-0.1, 0.26, 0.74, 1.1], requires_grad=True)
    step_output = quantize.step_quantize(x, 0.25)
    bit_output = quantize.k_bit_quantize(x.clamp(0, 1), 2)

    torch.testing.assert_close(step_output, torch.tensor([0.0, 0.25, 0.75, 1.0]))
    torch.testing.assert_close(
        bit_output,
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
    )
    (step_output.sum() + bit_output.sum()).backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0, 2.0, 1.0]))


def test_cubalif_inherited_multi_step_matches_repeated_single_steps():
    multi_step_node = CUBALIFNode()
    single_step_node = CUBALIFNode()
    x_seq = torch.randn(5, 2, 3)

    multi_step_output = multi_step_node.multi_step_forward(x_seq)
    single_step_output = torch.stack(
        [single_step_node.single_step_forward(x) for x in x_seq]
    )

    torch.testing.assert_close(multi_step_output, single_step_output)
    torch.testing.assert_close(multi_step_node.v, single_step_node.v)
    torch.testing.assert_close(multi_step_node.c, single_step_node.c)


def test_klif_scaled_reset_matches_documented_equations():
    spike = torch.tensor([0.0, 1.0])

    soft_reset = KLIFNode(scale_reset=True, v_reset=None)
    with torch.no_grad():
        soft_reset.k.fill_(2.0)
    soft_reset.v = torch.tensor([0.5, 1.5])
    soft_reset.neuronal_reset(spike)
    torch.testing.assert_close(soft_reset.v, torch.tensor([0.25, 0.25]))

    hard_reset = KLIFNode(scale_reset=True, v_reset=0.1)
    with torch.no_grad():
        hard_reset.k.fill_(2.0)
    hard_reset.v = torch.tensor([0.5, 1.5])
    hard_reset.neuronal_reset(spike)
    torch.testing.assert_close(hard_reset.v, torch.tensor([0.25, 0.1]))
