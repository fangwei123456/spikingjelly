## Bugs History with Releases

Some fatal bugs and when the bug is fixed are shown in this table. Note that the bug fixed after a release indicates that this bug may exist in the release.

| Bugs/Releases                                                | Date       |
| ------------------------------------------------------------ | ---------- |
| **Release: 0.0.0.0.4**                                       | 2021-03-25 |
| **Release: 0.0.0.0.6**                                       | 2021-07-03 |
| Bug: Tempotron, https://github.com/fangwei123456/spikingjelly/issues/88. This bug makes the maximum value of  psp kernel not be v_threshold. But considering that weights are learnable, this bug may not have much influence. | 2021-07-17 |
| Bug: SpikingRNNBase, https://github.com/fangwei123456/spikingjelly/issues/101. This bug makes SpikingVanillaRNN, and SpikingGRU use SpikingLSTMCell in their layers. | 2021-08-26 |
| Bug: Cupy backend for spiking neurons, https://github.com/fangwei123456/spikingjelly/issues/106. This bug makes spiking neurons with cupy backend output wrong spikes and voltages. This bug has no influence on release 0.0.0.0.4, which does not use cupy. | 2021-09-16 |
| **Release: 0.0.0.0.8**                                       | 2021-11-21 |
| Bug: MultiStepParametricLIFNode, https://github.com/fangwei123456/spikingjelly/issues/151. This bug makes the gradient of the learnable parameter in MultiStepParametricLIFNode incomplete when backend is cupy. | 2021-12-10 |
| **Release: 0.0.0.0.10**                                      |            |
| Bug: When using CuPy with `version >= 10`, CuPy will change `torch.cuda.current_device()` to 0, https://github.com/cupy/cupy/issues/6569. This bug will break training when using Distributed Data Parallel (DDP). | 2022-03-22 |

