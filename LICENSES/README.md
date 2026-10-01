# SpikingJelly license guide / 许可证指南

## Migration status / 迁移状态

**This branch is a migration draft, not an Apache-2.0 release.** Maintainers
have approved the direction of the migration. The concrete outstanding source
and authorization items are recorded in [PROVENANCE.md](PROVENANCE.md).
Replacing the root license does not grant additional rights in unresolved
contributions or third-party material. Draft distribution metadata uses
`LicenseRef-SpikingJelly-Migration-Pending`; these artifacts are for migration
validation and must not be published as a completed relicensing.

**本分支是迁移草稿，不是已完成换证的 Apache-2.0 发行版。** 维护方已同意迁移方向，
具体来源与授权待办见 [PROVENANCE.md](PROVENANCE.md)。替换根许可证不会为尚未确认的
贡献或第三方内容额外授予权利。草稿发行包以
`LicenseRef-SpikingJelly-Migration-Pending` 标记元数据，仅用于迁移验证，
不得作为已完成换证的版本发布。

## Intended terms / 迁移后的条款

For project material covered by the migration authorization, the target license
is [Apache License, Version 2.0](../LICENSE). Its English text is authoritative.
This guide explains the intended terms; it is not an additional license or a
translation of the legal text. Copyright remains with the respective holders.

Apache-2.0 permits research, commercial use, modification, and redistribution
subject to its terms, including use in proprietary applications. It does not
require a commercial-use registration, publication of an application's source
merely because it uses the framework, or submission of changes upstream.
Redistributors must include the license, mark modified files, and retain
applicable copyright, attribution, and NOTICE material. Its patent grant is
limited to the claims described in Section 3; it does not license unrelated
patents or grant trademark rights.

对于获得本次迁移授权的项目内容，目标许可证是 [Apache License 2.0](../LICENSE)，
以英文原文为准。本指南用于解释拟采用的条款，不是额外许可或法律文本译文。
版权仍由各自权利人持有。

Apache-2.0 允许遵守条款的科研、商业使用、修改和再分发，也允许用于闭源应用。
它不要求商业使用备案，不会仅因应用使用该框架就要求公开应用源码，
也不要求向上游提交修改。再发布者须附带许可证、标记修改，并保留适用的版权、
归属及 NOTICE 内容。第 3 条的专利授权具有明确范围，不覆盖无关专利，也不授予商标权。

## Third-party material / 第三方内容

[NOTICE](../NOTICE) maps identified adapted code to the original notices in
[third_party](third_party/). Those notices remain applicable to their respective
material; the root Apache license does not replace them. The source review
separately identifies material whose licensing has not yet been resolved.

Dependencies installed separately, downloaded datasets, and model weights have
their own licenses. Including a loader, example, or reference does not relicense
them. Requests to cite SpikingJelly's paper are academic guidance, not additional
conditions on the Apache-2.0 grant.

[NOTICE](../NOTICE) 列出已识别的改编代码及 [third_party](third_party/) 中的原始声明。
这些声明继续适用于对应内容，根目录 Apache 许可证不替代它们。
来源核对记录另外列明尚未解决许可问题的内容。

单独安装的依赖、下载的数据集和模型权重具有各自的许可证，提供加载器、示例或引用
不改变它们的许可。对 SpikingJelly 论文的引用请求属于学术使用建议，
不构成 Apache-2.0 授权的额外条件。

## Historical boundary / 历史边界

The pre-migration baseline is
[`4cfe72d533020a661859786600a6aa863f61d161`](https://github.com/fangwei123456/spikingjelly/tree/4cfe72d533020a661859786600a6aa863f61d161).
Its [Chinese license](https://github.com/fangwei123456/spikingjelly/blob/4cfe72d533020a661859786600a6aa863f61d161/LICENSE),
[guide](https://github.com/fangwei123456/spikingjelly/blob/4cfe72d533020a661859786600a6aa863f61d161/LICENSES/README.md),
and [translations](https://github.com/fangwei123456/spikingjelly/tree/4cfe72d533020a661859786600a6aa863f61d161/LICENSES/translations)
remain available at that immutable revision. Previous tags and published
archives retain their own licensing records, including the Open-Intelligence
Open Source License 1.0 where applicable.

Once the outstanding items are resolved, the merge of this migration will mark
the change for the development line. The first subsequent release must identify
that merge commit and its own release tag in its release notes. This draft does
not designate an existing tag as Apache-2.0, change the package version, waive
past obligations, or retroactively relicense old distributions. New contribution
terms are described in [CONTRIBUTING.md](../CONTRIBUTING.md); they do not provide
retroactive permission for earlier contributions.

迁移前基线为上述固定提交，其中的中文许可证、指南和译文继续保留。
既有标签和已发布包保留各自的许可记录，包括适用的启智开源许可证 1.0。
待具体待办解决后，本迁移的合入提交将成为开发主线的切换点；后续首个发行版
应在发布说明中记录该提交及自己的发行标签。本草稿不把既有标签改标为 Apache-2.0，
不改变包版本，不豁免过去的义务，也不追溯更改旧包许可。
[贡献指南](../CONTRIBUTING.md) 中的新贡献条款不构成历史贡献的追溯授权。
