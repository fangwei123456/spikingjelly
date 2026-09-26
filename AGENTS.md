# Project Agent Guidelines

## Getting Started and Environment

- Read [CONTRIBUTING.md](CONTRIBUTING.md) and check `git status` before development. Preserve unrelated changes.
- Follow `CONTRIBUTING.md` and `pyproject.toml` for installation steps and optional dependencies. Manage Python and dependencies with `uv`, and reuse an existing environment when possible.
- Before remote testing or reusing a worktree environment, read the machine-specific conventions in a local `ENV.md` if it exists. Otherwise, verify the actual environment instead of guessing its configuration.
- If `.codegraph/` exists, use CodeGraph first to locate code. Read files that are not indexed directly; do not create an index yourself.

## Implementation and Verification

- Prioritize correctness and public contracts, then measured performance on critical paths, then low complexity. Reuse existing code first; avoid unrelated refactoring, speculative abstractions, compatibility branches, and defensive code without a known failure mode.
- Import order: standard library, third-party packages, then local modules; prefer relative imports for local modules. Use an 88-character line width and double quotes. Use `PascalCase` for classes, `snake_case` for functions and variables, uppercase names for constants, and an `_` prefix for private members. Do not put notebooks under `src`.
- Fully annotate parameters and return values of public functions. Use common types from `typing`; use `Optional[...]` for nullable parameters.
- Load optional dependencies (CuPy, Triton, Lava, etc.) with `try-except`; log load failures at `logging.info` level and raise a clear `ImportError` when the dependency is needed. Catch `BaseException` broadly only to preserve an existing pattern.
- Internal `StepModule` implementations are atomic step-mode leaves by default; their children normally should not implement `StepModule`. Use plain `nn.Module` for networks, blocks, and attention modules. Modules that own independent temporal state and scheduling, and explicit scheduling containers, may be exceptions when the reason is documented. This does not restrict external user modules.
- Run the nearest relevant tests with `pytest`; do not execute test files directly. Verify correctness when changing numerical results, state, or output. For hot-path changes, compare before and after in the same environment with the same workload. Report what was and was not verified; do not treat a smoke test as proof of equivalence.
- Follow `CONTRIBUTING.md` for formatting and production-log checks. Check only the scope affected by the change.

## Public API Documentation

**Public APIs must have semantically equivalent Chinese and English docstrings and follow the template below.**
This applies to public classes, functions, methods, module constants, and factories. Update the docstring and relevant pages when changing parameters, returns, state, or side effects.

- Follow `spikingjelly/activation_based/functional/loss.py` for the format: language navigation and reference labels, then Chinese, then equivalent English. Use `r"""..."""` for new docstrings; migrate existing ones only when touching them or fixing escaping.
- Use Sphinx/RST. Document each parameter with `:param:` and `:type:`; add `:return:` and `:rtype:` only for functions with return values, and `:raises:` for actual exceptions. Keep types consistent with the signature.
- Describe defaults, units, ranges, `None` branches, tensor shape/dtype/device/backend constraints, state, and side effects. Explain differences in return behavior under training/eval, `detach_reset`, or different backends when applicable.
- Put class interface documentation in `__init__` rather than repeating it in a class docstring, to avoid duplicate Sphinx output. Keep a class docstring only when another class in the same file inherits from it (so subclasses can inherit the documentation without duplicate output or labels), or when there is no public constructor.
- Use `.. math::`, `.. code-block:: python`, and `.. image::` for math, code, and images. Prefer `:class:`, `:func:`, and `:mod:` for cross-references. Examples must be runnable when copied and cover relevant edge cases.
- Acceptance criteria: both languages, all fields and constraints, and a signature-consistent description; the documentation builds without new serious warnings.

### Required Docstring Template

The following is a fill-in skeleton, not a new API. Replace every `填写` / `Describe` placeholder, add or remove fields to match the actual signature, and replace `module-api_name` with a unique label prefix for the API in both navigation links and both labels to avoid Sphinx label collisions.
Both the Chinese and English sections must describe the purpose, key behavior, and state semantics (such as the effect of `reset()`) as verifiable facts in the order “input conditions → behavior → output.”
For functions without a return value, including `__init__`, remove `:return:` and `:rtype:` in both languages. List an exception in `:raises:` only when it can actually occur under the stated condition.
Add formulas, notes, references, and runnable examples in both languages when needed.

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

- Starting with V2, record user-visible changes to features, APIs, dependencies or installation, semantics, migration, or compatibility in `CHANGELOG.md`. Purely internal refactoring usually needs no entry, except when it changes the public module structure, documentation entry points, or recommended usage.
- Write user-facing English descriptions. Do not list changes commit by commit or describe temporary implementation steps. Give each bullet one specific change; give major features their own entries and avoid vague entries such as “other improvements.” Group `Features` by functional area and name the Python module for each new feature block. State the scope of experimental capabilities without presenting them as stable support.
- Edit only `CHANGELOG.md` by hand, not the generated `docs/source/changelog.rst`. After editing, run:

```bash
uv run python tools/generate_changelog_rst.py
uv run python tools/generate_changelog_rst.py --check
```

Verify the documentation build when changing public API documentation. Also run it when changing the Change Log structure or its generation script:

```bash
uv run sphinx-build -M html docs/source docs/build
```

The HTML entry point is `docs/build/html/index.html`. You may also use `cd docs && make html` as described in `CONTRIBUTING.md`.
