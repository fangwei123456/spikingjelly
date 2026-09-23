# 项目 Agent 约定

## 入口与环境

- 开发前读 [CONTRIBUTING.md](CONTRIBUTING.md)，检查 `git status`，保留无关改动。
- 安装步骤和可选依赖以 `CONTRIBUTING.md`、`pyproject.toml` 为准；用 `uv` 管理 Python 与依赖，优先复用已有环境。
- 远程测试或 worktree 环境复用：若本地存在 `ENV.md`，先读其中的机器约定；否则核实实际环境，不猜测配置。
- 存在 `.codegraph/` 时，定位代码先用 CodeGraph；未索引的文件直接读取，不自行创建索引。

## 实现与验证

- 正确性和公开契约优先，其次是实测关键路径性能，再其次是低复杂度。优先复用现有代码；不夹带无关重构、投机式抽象、兼容分支或无依据防御。
- 导入顺序：标准库 → 第三方 → 本地模块；本地优先相对导入。行宽 88，双引号；类用 `PascalCase`，函数/变量用 `snake_case`，常量大写，私有成员加 `_`。`src` 下不放 notebook。
- 公共函数参数、返回值完整标注，常用类型取自 `typing`；可空参数用 `Optional[...]`。
- 可选依赖（CuPy/Triton/Lava 等）用 `try-except`，加载失败记 `logging.info`，使用时按需抛清晰的 `ImportError`；仅为保持现有模式可宽捕获 `BaseException`。
- 内部 `StepModule` 默认是 step-mode 原子叶节点，子模块通常不再实现它；网络/block/attention 用普通 `nn.Module`。拥有独立时序状态和调度职责的模块、显式调度容器可例外，须说明依据；不限制外部用户模块。
- 用 `pytest` 运行最近的相关测试，不直接执行测试文件。数值、状态、输出变化验证正确性；热路径变化需同环境、同 workload 的前后测量。报告已验证和未验证项，不将 smoke test 当作等价性证明。
- 格式化及生产日志检查遵循 `CONTRIBUTING.md`；只检查与本次变更有关的范围。

## 公共 API 文档

**公开 API 必须提供语义一致的中英双语 docstring，并遵循下面的格式模板。**
适用于公开类、函数、方法、模块常量和工厂；修改参数、返回、状态或副作用时同步 docstring 与相关页面。

- 格式参考 `spikingjelly/activation_based/functional/loss.py`：语言导航及引用标签 → 中文 → 等价 English。新增 docstring 用 `r"""..."""`；已有文档仅在触及时或转义出错时迁移。
- 使用 Sphinx/RST：参数逐项写 `:param:`/`:type:`，有返回值才写 `:return:`/`:rtype:`，异常写 `:raises:`；类型与签名一致。
- 说明默认值、单位、范围、`None` 分支、张量 shape/dtype/device/backend、状态和副作用；返回行为依赖 training/eval、detach_reset 或后端时说明差异。
- 类接口说明必须放 `__init__`，不再重复写 class docstring，避免 Sphinx 重复拼接。仅有两个例外：被同文件其他类继承（子类可继承文档，避免重复渲染及重复标签），或没有公开构造函数时，保留 class docstring。
- 数学、代码、图片用 `.. math::`、`.. code-block:: python`、`.. image::`；交叉引用优先 `:class:`/`:func:`/`:mod:`。示例须可复制运行，覆盖相关边界。
- 验收：双语、字段和约束齐全，签名一致；文档可构建且无新增严重告警。

### Docstring 格式模板（必须遵循）

下例是填写骨架，不是新增 API。替换所有 `填写` / `Describe` 占位内容，按实际签名增删字段；
将 `module-api_name` 替换为该 API 的唯一标签前缀，并同步替换两处导航和两个标签，避免 Sphinx 标签冲突。
中文、English 小节均需写功能概述、关键行为和状态语义（如 `reset()` 的影响），以“输入条件 → 行为 → 输出”描述可验证事实。
无返回值（包括 `__init__`）时删除两种语言的 `:return:` / `:rtype:`；`:raises:` 仅列真实异常及触发条件。
必要时在两种语言中补充公式、注意事项、参考文献，并添加可复制运行的示例。

```python
def api_name(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    r"""
    **API Language** - :ref:`中文 <module-api_name-cn>` | :ref:`English <module-api_name-en>`

    ----

    .. _module-api_name-cn:

    * **中文**

    填写：功能概述、关键行为、状态变化及副作用。

    :param x: 填写：用途、形状及各维含义、dtype、device 和后端约束。
    :type x: torch.Tensor
    :param scale: 填写：用途、单位、取值范围；默认 ``None`` 时的具体行为。
    :type scale: Optional[float]
    :return: 填写：输出结构、形状、dtype/device，以及不同模式下的行为。
    :rtype: torch.Tensor
    :raises ValueError: 填写：实际触发该异常的条件。

    ----

    .. _module-api_name-en:

    * **English**

    Describe the purpose, key behavior, state changes, and side effects.

    :param x: Describe its purpose, shape and dimensions, dtype, device,
        and backend constraints.
    :type x: torch.Tensor
    :param scale: Describe its purpose, units, valid range, and the behavior
        when it is ``None`` (the default).
    :type scale: Optional[float]
    :return: Describe the output structure, shape, dtype/device, and behavior
        under different modes.
    :rtype: torch.Tensor
    :raises ValueError: Describe the actual condition that raises this error.
    """
    ...
```

## Change Log

- V2 起，用户可见功能、API、依赖/安装、语义、迁移或兼容性变化写入 `CHANGELOG.md`；纯内部重构通常免写，公开模块结构、文档入口或推荐用法变化除外。
- 用英文面向用户描述，不逐 commit 罗列或记录临时过程；一条 bullet 一个具体变化，重要功能单列，不用“其他改进”等空泛概括。`Features` 按功能域分组，每个新功能块标明 Python module；实验能力明确范围，不夸大为稳定支持。
- 只手改 `CHANGELOG.md`，不手改生成的 `docs/source/changelog.rst`。修改后运行：

```bash
uv run python tools/generate_changelog_rst.py
uv run python tools/generate_changelog_rst.py --check
```

修改公共 API 文档时验证构建；修改 Change Log 结构或生成脚本时也运行：

```bash
uv run sphinx-build -M html docs/source docs/build
```

HTML 入口为 `docs/build/html/index.html`；也可按 `CONTRIBUTING.md` 使用 `cd docs && make html`。
