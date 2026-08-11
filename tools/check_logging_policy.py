"""Check the repository Loguru conventions with the Python AST."""

from __future__ import annotations

import argparse
import ast
import re
import string
from pathlib import Path


EXCLUDED_PARTS = {"example", "examples", "test", "benchmark", "docs"}
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
DIAGNOSTIC_PRINT_FUNCTIONS = {"check_manual_grad", "check_cuda_grad"}
PERCENT_PLACEHOLDER = re.compile(
    r"%(?!%)(?:\([^)]+\))?[#0 +\-]?(?:\d+|\*)?(?:\.\d+|\.\*)?[hlL]?[diouxXeEfFgGcrsa]"
)


def _is_production(path: Path) -> bool:
    return not (set(path.parts) & EXCLUDED_PARTS or path.name.endswith("_example.py"))


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.value if isinstance(node, ast.Attribute) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _is_logger_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_METHODS
        and _root_name(node.func.value) == "logger"
    )


def _is_allowed_diagnostic_print(
    path: Path, node: ast.Call, parents: dict[ast.AST, ast.AST]
) -> bool:
    if path.name != "surrogate.py":
        return False
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return (
                current.name in DIAGNOSTIC_PRINT_FUNCTIONS
                and path.parent.name == "activation_based"
            )
        current = parents.get(current)
    return False


def _is_eager_message(node: ast.AST) -> bool:
    has_string_literal = any(
        isinstance(part, ast.Constant) and isinstance(part.value, str)
        for part in ast.walk(node)
    )
    return (
        isinstance(node, ast.JoinedStr)
        or isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Mod))
        and has_string_literal
        or isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    )


def _placeholder_count(message: str) -> int:
    def count(template: str) -> int:
        total = 0
        for _, field, format_spec, _ in string.Formatter().parse(template):
            if field is not None:
                total += 1
                total += count(format_spec)
        return total

    return count(message)


def _check_logger_call(path: Path, node: ast.Call) -> list[str]:
    violations: list[str] = []
    if not node.args:
        return [f"{path}:{node.lineno}: logger call requires a message"]

    message = node.args[0]
    if _is_eager_message(message) or any(
        _is_eager_message(arg) for arg in node.args[1:]
    ):
        violations.append(f"{path}:{node.lineno}: eager formatting in logger call")

    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        if PERCENT_PLACEHOLDER.search(message.value):
            violations.append(
                f"{path}:{node.lineno}: percent placeholder in logger call"
            )
        try:
            placeholders = _placeholder_count(message.value)
        except ValueError as exc:
            violations.append(f"{path}:{node.lineno}: invalid Loguru format: {exc}")
        else:
            arguments = len(node.args) - 1
            if placeholders != arguments:
                violations.append(
                    f"{path}:{node.lineno}: Loguru placeholder count {placeholders} "
                    f"does not match argument count {arguments}"
                )

    return violations


def check(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if not _is_production(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            violations.append(f"{path}: syntax error: {exc}")
            continue

        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        imports_package_logger = path.name == "logger.py" or any(
            isinstance(node, ast.ImportFrom)
            and node.module == "spikingjelly.logger"
            and any(alias.name == "logger" for alias in node.names)
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logging" or alias.name.startswith("logging."):
                        violations.append(
                            f"{path}:{node.lineno}: stdlib logging import"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "logging" or (node.module or "").startswith(
                    "logging."
                ):
                    violations.append(f"{path}:{node.lineno}: stdlib logging import")
                if node.module == "loguru" and path.name != "logger.py":
                    violations.append(
                        f"{path}:{node.lineno}: logger must be imported from spikingjelly.logger"
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {"print", "pprint"}:
                    if not _is_allowed_diagnostic_print(path, node, parents):
                        violations.append(
                            f"{path}:{node.lineno}: direct {func.id} call"
                        )
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "builtins"
                    and func.attr == "print"
                ):
                    violations.append(
                        f"{path}:{node.lineno}: direct builtins.print call"
                    )
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"bind", "contextualize", "opt"}
                    and _root_name(func.value) == "logger"
                ):
                    violations.append(
                        f"{path}:{node.lineno}: logger.{func.attr} is not allowed"
                    )
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "sys"
                    and func.value.attr in {"stdout", "stderr"}
                    and func.attr == "write"
                ):
                    violations.append(f"{path}:{node.lineno}: direct sys stream write")
                elif _is_logger_call(node):
                    violations.extend(_check_logger_call(path, node))
                    if not imports_package_logger:
                        violations.append(
                            f"{path}:{node.lineno}: logger must be imported from spikingjelly.logger"
                        )
            elif isinstance(node, ast.Name) and node.id == "logging":
                violations.append(f"{path}:{node.lineno}: stdlib logging reference")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "builtins"
                        and target.attr == "print"
                    ):
                        violations.append(
                            f"{path}:{node.lineno}: builtins.print rebinding"
                        )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("spikingjelly"))
    args = parser.parse_args()
    violations = check(args.root)
    if violations:
        print("Logging policy violations:")
        print("\n".join(violations))
        return 1
    print("Logging policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
