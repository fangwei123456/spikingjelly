# Apache-2.0 migration source review

Reviewed on 2026-10-01 against SpikingJelly
[`4cfe72d533020a661859786600a6aa863f61d161`](https://github.com/fangwei123456/spikingjelly/tree/4cfe72d533020a661859786600a6aa863f61d161).
This is a source and license-record review, not a certification of every
contributor's ownership. Maintainer/institutional approval of the migration is
an accepted project premise. No contributor or upstream author was contacted.

**Status: draft; do not merge or publish as a completed Apache-2.0 migration.**
The root Apache text is the intended project license; it does not grant rights
in the unresolved material below. The package's
`LicenseRef-SpikingJelly-Migration-Pending` identifier is explicitly a review
marker, not a substitute permission grant. See [the guide](README.md).

## Outstanding items

### QKFormer-derived attention: permission not found

Local scope: `spikingjelly/activation_based/layer/attention.py`, `QKAttention`
and its token/channel variants. Introduced by
[`4fe7e029`](https://github.com/fangwei123456/spikingjelly/commit/4fe7e029b9f2ab5830deaddb9043e1cb3ddd547b).
The documentation explicitly describes adaptation of QKFormer source.
The fused q/k implementation retains the q/k Conv-BN-LIF pipeline, q reduction,
threshold-0.5 LIF, multiplicative gating and projection pipeline.

At historical upstream
[`43f0adf64e7a19690dbcd422f2001adf91221f39`](https://github.com/zhouchenlin2096/QKFormer/tree/43f0adf64e7a19690dbcd422f2001adf91221f39),
the full Git tree, README and
[`imagenet/qkformer.py`](https://github.com/zhouchenlin2096/QKFormer/blob/43f0adf64e7a19690dbcd422f2001adf91221f39/imagenet/qkformer.py)
contain no permission grant found by this review. The absence of a public grant
does not rule out a separate agreement. Spikformer's MIT license cannot be
assumed to cover QKFormer-specific changes.

Required resolution: locate an existing permission covering this adaptation,
or obtain permission from the relevant rightsholder. Do not merely remove the
source citation, assign an MIT notice to the upstream author, or call ordinary
refactoring an independent rewrite. Any replacement is a separate change with
correctness validation.

Unsent request draft:

> We are preparing an Apache-2.0 migration of SpikingJelly. Its QKAttention
> implementation documents adaptation from QKFormer at commit
> 43f0adf64e7a19690dbcd422f2001adf91221f39. Is there an existing license or
> permission covering those contributions? If not, would the relevant
> rightsholder authorize their use and redistribution in this Apache-2.0
> migration, identifying any attribution or other conditions?

### s2net-referenced speech processing: origin needs clarification

Local scope: `spikingjelly/datasets/speechcommands.py` (sample weighting) and
`spikingjelly/activation_based/examples/speech_commands.py` (`Pad`, `Rescale`,
and spectral postprocessing).
The locally cited revisions
[`b073f755`](https://github.com/romainzimmer/s2net/blob/b073f755e70966ef133bbcd4a8f0343354f5edcd/LICENSE)
and [`82c38bf8`](https://github.com/romainzimmer/s2net/blob/82c38bf80b55d16d12d0243440e34e52d237a2df/LICENSE)
carry GPLv3. The inspected code has short matching processing patterns, but no
long identical block was found. Standard-deviation formulas and common
normalization algorithms alone do not establish copying of protected code.
The surrounding torchaudio adaptation is separately covered by BSD-2-Clause.

Required resolution: ask the original implementers to identify whether these
specific parts were independently implemented or adapted from s2net, and any
additional permission. Do not label the entire files GPL merely because they
cite s2net, or assume the citation can be deleted to clear the issue.

Unsent request draft:

> 在整理许可证时，我们看到 speechcommands 的数据集权重、Pad 和频谱后处理参考了
> s2net，其固定版本使用 GPLv3。请确认这些部分是按论文/算法独立实现，还是从源码
> 改编；若有改编，请指出范围及是否已有额外授权。我们不会仅因公式相同认定代码复制。

### Historical contributions: approval scope not established by Git authorship

The inspected contribution guide and tracked GitHub configuration contain no
CLA or general relicensing agreement. No such historical records were supplied
for this review. Maintainer/institutional approval is accepted for the material
it covers; contributor identity, employer ownership and authority cannot be
inferred from a Git username or number of commits.

A concrete surviving substantive contribution requiring scope confirmation is
[`RAFNode` and `raf_step`, PR #750](https://github.com/fangwei123456/spikingjelly/pull/750)
by `tritsystem`, in `neuron/resonate.py` and `functional/neuron.py` under
`spikingjelly/activation_based/`. The reviewed PR body/comments do not supply a
separate relicensing permission. This is a coverage example, not a claim that
this contributor refuses, owns all relevant rights, or is the only unresolved
contributor. Ordinary PR acceptance and bot approval are not additional grants.

Required resolution: maintainers identify which retained contributions are
covered by existing institutional or individual authorization, and record the
basis for any remaining substantive items. Trivial edits, removed contributions
and already licensed upstream material do not justify blanket signature requests.
Do not claim a complete title audit from this scoped source review.

Unsent request draft for an uncovered substantive contribution:

> We are migrating SpikingJelly's current development line to Apache-2.0. Please
> confirm whether you have authority to license your retained contributions in
> [specific PR/files], and whether an existing agreement already covers this
> migration. If not, do you agree to provide those contributions under
> Apache-2.0? Copyright remains with its existing holder(s).

## Identified permissive source material

Local paths are relative to `spikingjelly/`. These are fixed **license-evidence
snapshots**, not claims that every original author copied precisely that SHA.
The full notices are preserved in [third_party](third_party/) (trailing
whitespace normalized in torchvision/torchaudio; MA-SNN line endings normalized), and
[NOTICE](NOTICE) maps them to local material. Existing modified implementations
were inspected without changing their behavior.

| Local scope | Upstream evidence | License and treatment |
| --- | --- | --- |
| ResNet/VGG model variants; classification training and `tv_ref_classify` utilities | [torchvision v0.12.0](https://github.com/pytorch/vision/tree/9b5a3fecc72434dbd65148723efe54b28b9728c9), with the same BSD terms checked before the 2021 model import | BSD-3-Clause; retain Soumith Chintala notice |
| `examples/common/tv_ref_classify/sampler.py`, RASampler | [DeiT](https://github.com/facebookresearch/deit/tree/a2ffd162edc49b30f6ed8044cd10444bf2ae1299) | Apache-2.0; retain Facebook notice and mark modifications |
| `activation_based/lava_exchange.py`, BatchNorm2d | [Lava-DL](https://github.com/lava-nc/lava-dl/tree/a6927a2597b7b4dd483414b66855744051384796), before local addition `72d92547` | BSD-3-Clause; retain both repository and norm.py Intel notices |
| `activation_based/layer/attention.py`, SpikingSelfAttention | [Spikformer](https://github.com/ZK-Zhou/spikformer/tree/621268981827af47aa666c1b5f774de4281d90ad) | MIT; retain Zhaokun Zhou notice |
| Same module, multidimensional attention primitives | [MA-SNN](https://github.com/MA-SNN/MA-SNN/tree/8f87a7476920279e4f45e014aed868b62f8ef001), license present before local `208d2488` | MIT; retain xyz837 notice. Original local layout matches MA-SNN, not the distinctive changes in the separately cited SNN_Attention_VGG example |
| `activation_based/triton_kernel/triton_utils.py`, guard/AMP helpers | [flash-snn](https://github.com/AllenYolk/flash-snn/tree/e31d0c332536425bf6a3d5e4c5ff0dba44678552), [FLA](https://github.com/fla-org/flash-linear-attention/tree/f7d95fa04fcb2df07390b1bf8b7d6b96ffb7a16a) | MIT; retain Yifan Huang and historical Songlin Yang notices; original guard AST matches flash-snn |
| `datasets/cifar10_dvs.py` event helpers; `datasets/utils.py` ATIS decoder | [events-tfds metadata](https://github.com/jackd/events-tfds/blob/0e61bedf71c0c4be1d6c9a00a9f55850c7f29986/setup.py) | Explicit MIT declaration and Dominic Jack author metadata. No complete upstream LICENSE found; accompanying standard terms are identified as reconstructed from the declaration, with no invented copyright year |
| `datasets/utils.py` ATIS decoder through events-tfds | [event-Python](https://github.com/gorchard/event-Python/tree/78dd3b0a7fc508d551cecdbf93b959dc2d265765) | MIT; retain Garrick Orchard notice from the upstream decoder chain |
| `datasets/speechcommands.py`; example MelScaleDelta/create_fb_matrix | [torchaudio dataset snapshot](https://github.com/pytorch/audio/tree/95d9f2d272b2814010db7fc803a7d4dc6cf0c3b4), [transforms snapshot](https://github.com/pytorch/audio/tree/9d50acf3970d9f0d2cb903c176f7d9523ba51b6a) | BSD-2-Clause; retain Facebook/Soumith Chintala notice; separate from unresolved s2net scope |
| `examples/common/multiprocessing_env.py` | [OpenAI Baselines](https://github.com/openai/baselines/tree/ea25b9e8b234e6ee1bca43083f8f3cf974143998) | MIT; retain OpenAI notice |
| `examples/dsqn/ptan/` | [PTAN](https://github.com/Shmuma/ptan/tree/5258ad3322b917e3021e34b4f5895c85cda8b958), older matching agent snapshot `ddf9ae54` | MIT; retain Maxim Lapan notice |
| `examples/cifar10_r11_enabling_spikebased_backpropagation.py` | [Enabling Spike-based Backpropagation](https://github.com/chan8972/Enabling_Spikebased_Backpropagation/tree/46e18a7b0741d08f155435e2c62acac4bebe5b2f) | MIT; retain Chankyu Lee notice for implementation/initialization reference |

Independently installed dependencies are not automatically combined into this
package's SPDX expression. Data, pretrained weights and external download links
retain their own terms. This table does not assert that every upstream commit,
image, example, or later model addition has undergone an exhaustive ownership
audit.

## Release handoff

After the outstanding items are resolved, record the evidence here, replace the
pending package marker with an accurate SPDX expression, remove the pending
notice from the license-file list and package, and update both language versions
of the guide/README/docs/changelog to describe the completed migration. If no
additional license obligations are found, the known bundled notices imply
`Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND MIT`, not Apache-2.0 alone.
Do not mechanically apply that expression while the items above remain open.
Record the actual merge commit and first subsequent release tag in the release
notes; do not rewrite earlier tags or distributions.
