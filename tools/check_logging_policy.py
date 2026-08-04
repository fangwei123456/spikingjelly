"""Check the repository logging conventions with the Python AST."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


EXCLUDED_PARTS = {"example", "examples", "test", "benchmark", "docs"}
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
DIAGNOSTIC_PRINT_FUNCTIONS = {"check_manual_grad", "check_cuda_grad"}


def _is_production(path: Path) -> bool:
    return not (set(path.parts) & EXCLUDED_PARTS or path.name.endswith("_example.py"))


def _is_logging_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"logger", "logging"}
        and node.func.attr in LOG_METHODS
    )


def _is_get_logger_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logging"
        and node.func.attr == "getLogger"
    )


def _is_package_logger_module(path: Path, root: Path) -> bool:
    return path.resolve() == (root / "logger.py").resolve()


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


def _is_eagerly_formatted_message(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod))


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
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
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
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "sys"
                    and func.value.attr in {"stdout", "stderr"}
                    and func.attr == "write"
                ):
                    violations.append(f"{path}:{node.lineno}: direct sys stream write")
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr in LOG_METHODS
                ):
                    violations.append(
                        f"{path}:{node.lineno}: root logging.{func.attr} call"
                    )
                elif (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr
                    in {
                        "basicConfig",
                        "dictConfig",
                        "fileConfig",
                        "StreamHandler",
                        "FileHandler",
                        "NullHandler",
                    }
                ):
                    if func.attr != "NullHandler" or not _is_package_logger_module(
                        path, root
                    ):
                        violations.append(
                            f"{path}:{node.lineno}: logging configuration call"
                        )
                elif _is_get_logger_call(node):
                    if not _is_package_logger_module(path, root):
                        violations.append(
                            f"{path}:{node.lineno}: logger must be imported from spikingjelly.logger"
                        )
                    if not (
                        len(node.args) == 1
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "spikingjelly"
                    ):
                        violations.append(
                            f"{path}:{node.lineno}: non-package logger name"
                        )
                elif isinstance(func, ast.Attribute) and _is_get_logger_call(
                    func.value
                ):
                    violations.append(
                        f"{path}:{node.lineno}: logger must be imported from spikingjelly.logger"
                    )
                if _is_logging_call(node):
                    if node.args and _is_eagerly_formatted_message(node.args[0]):
                        violations.append(
                            f"{path}:{node.lineno}: manual string formatting in logging call"
                        )
                    for value in node.args:
                        if isinstance(value, ast.JoinedStr):
                            violations.append(
                                f"{path}:{node.lineno}: f-string in logging call"
                            )
                        if (
                            isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Attribute)
                            and value.func.attr == "format"
                        ):
                            violations.append(
                                f"{path}:{node.lineno}: .format() in logging call"
                            )
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
