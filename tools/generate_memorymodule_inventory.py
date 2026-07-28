from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "spikingjelly" / "activation_based"
TEST_ROOT = REPO_ROOT / "test" / "activation_based"
MODULE_PREFIX = "spikingjelly.activation_based"


CLASSIFICATION_BY_NAME = {
    "ActivationAwareIFNode": "migrate",
    "AdaptBaseNode": "2c_function_only",
    "BatchNormThroughTime1d": "exclude_submodule_schedule",
    "BatchNormThroughTime2d": "exclude_submodule_schedule",
    "BatchNormThroughTime3d": "exclude_submodule_schedule",
    "BaseNode": "2c_function_only",
    "CUBALIFNode": "2c_function_only",
    "CubaLIFNode": "2c_function_only",
    "Delay": "migrate",
    "DropConnectLinear": "exclude_random_realization",
    "Dropout": "exclude_random_realization",
    "Dropout2d": "exclude_random_realization",
    "DSRIFNode": "exclude_local_autograd_state",
    "DSRLIFNode": "exclude_local_autograd_state",
    "EIFNode": "2c_function_only",
    "ElementWiseRecurrentContainer": "exclude_callable_composition",
    "FlexSN": "migrate_after_cache_prototype",
    "GatedLIFNode": "migrate",
    "HalfThresholdIFNode": "2c_function_only",
    "IFNode": "2c_function_only",
    "IzhikevichNode": "2c_function_only",
    "KLIFNode": "2c_function_only",
    "LatencyEncoder": "migrate",
    "LIAFNode": "2c_function_only",
    "LIFNode": "2c_function_only",
    "LinearRecurrentContainer": "exclude_callable_composition",
    "MaskedPSN": "migrate",
    "MPBNBaseNode": "2c_function_only",
    "MPBNLIFNode": "2c_function_only",
    "MSTDPLearner": "migrate_with_compat_helper",
    "MSTDPETLearner": "migrate_with_compat_helper",
    "NeuNorm": "migrate",
    "OTTTLIFNode": "2c_function_only",
    "ParametricLIFNode": "2c_function_only",
    "PeriodicEncoder": "exclude_time_indexing",
    "QIFNode": "2c_function_only",
    "save_v_LIFNode": "2c_function_only",
    "SimpleBaseNode": "2c_function_only",
    "SimpleIFNode": "2c_function_only",
    "SimpleLIFNode": "2c_function_only",
    "SlidingPSN": "migrate",
    "SLTTLIFNode": "2c_function_only",
    "SpikeZIPEmbedding": "exclude_first_step_schedule",
    "SpikeZIPConv2d": "migrate",
    "SpikeZIPLayerNorm": "exclude_state_adapter",
    "SpikeZIPLinear": "migrate",
    "SpikeZIPRobertaSelfAttention": "migrate",
    "SpikeZIPSoftmax": "exclude_state_adapter",
    "SpikeZIPViTSelfAttention": "exclude_composite_only",
    "STBIFNeuron": "migrate",
    "STDPLearner": "migrate_with_compat_helper",
    "StatefulEncoder": "exclude_time_indexing",
    "SynapseFilter": "migrate",
    "TDConv2d": "exclude_state_adapter",
    "TDGELU": "exclude_state_adapter",
    "TDLayerNorm": "exclude_state_adapter",
    "TDLinear": "exclude_state_adapter",
    "TDModule": "exclude_state_adapter",
    "TDMultiheadAttention": "exclude_state_adapter",
    "TDRMSNorm": "exclude_state_adapter",
    "TDScaledDotProductAttention": "exclude_state_adapter",
    "TDSiLU": "exclude_state_adapter",
    "TDSoftmax": "exclude_state_adapter",
    "SNNElementWiseProduct": "exclude_state_adapter",
    "SNNMatrixOperator": "exclude_state_adapter",
    "WeightedPhaseEncoder": "migrate",
    "_BatchNormThroughTimeBase": "exclude_submodule_schedule",
    "_STAConstant": "exclude_first_step_schedule",
    "_STASpikeEncoder": "migrate",
    "_TDTanh": "exclude_state_adapter",
}


@dataclass
class ClassInfo:
    module: str
    name: str
    file: Path
    line: int
    bases: list[str]
    resolved_bases: list[str]
    registered_memory: list[str] = field(default_factory=list)
    self_assignments: list[str] = field(default_factory=list)
    child_or_buffer_candidates: list[str] = field(default_factory=list)
    supported_backends: list[str] = field(default_factory=list)
    has_single_step: bool = False
    has_multi_step: bool = False

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


def module_from_path(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    return ".".join((MODULE_PREFIX, *relative.parts))


def resolve_relative_module(
    current_module: str, level: int, imported: str | None
) -> str:
    parts = current_module.split(".")
    package_parts = parts[:-1]
    if level > 1:
        package_parts = package_parts[: -(level - 1)]
    if imported:
        package_parts.extend(imported.split("."))
    return ".".join(package_parts)


def import_aliases(tree: ast.Module, current_module: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level and current_module.startswith(MODULE_PREFIX):
                module_name = resolve_relative_module(
                    current_module, node.level, node.module
                )
            else:
                module_name = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "*":
                    continue
                aliases[local] = f"{module_name}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name
    return aliases


def resolve_expr(expr: ast.expr, aliases: dict[str, str], current_module: str) -> str:
    if isinstance(expr, ast.Name):
        return aliases.get(expr.id, f"{current_module}.{expr.id}")
    if isinstance(expr, ast.Attribute):
        prefix = resolve_expr(expr.value, aliases, current_module)
        return f"{prefix}.{expr.attr}"
    if isinstance(expr, ast.Subscript):
        return resolve_expr(expr.value, aliases, current_module)
    return ast.unparse(expr)


def string_arg(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def collect_literal_strings(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
    return values


def self_attr_name(target: ast.AST) -> str | None:
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def is_child_or_buffer_assignment(value: ast.AST) -> bool:
    if isinstance(value, ast.Call):
        func = ast.unparse(value.func)
        return (
            func.startswith("nn.")
            or "BatchNorm" in func
            or func.endswith("Monitor")
            or func.startswith("base.")
        )
    return False


def scan_class(
    node: ast.ClassDef, module: str, file: Path, aliases: dict[str, str]
) -> ClassInfo:
    info = ClassInfo(
        module=module,
        name=node.name,
        file=file,
        line=node.lineno,
        bases=[ast.unparse(base) for base in node.bases],
        resolved_bases=[resolve_expr(base, aliases, module) for base in node.bases],
    )
    assigned: set[str] = set()
    child_or_buffer: set[str] = set()
    memories: set[str] = set()
    backends: set[str] = set()

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == "single_step_forward":
                info.has_single_step = True
            elif item.name == "multi_step_forward":
                info.has_multi_step = True
            elif item.name == "supported_backends":
                for literal in collect_literal_strings(item):
                    if literal in {"torch", "cupy", "triton", "inductor", "lava"}:
                        backends.add(literal)

        for child in ast.walk(item):
            if isinstance(child, ast.Call):
                func = ast.unparse(child.func)
                if func.endswith(".register_memory") and child.args:
                    name = string_arg(child.args[0])
                    if name is not None:
                        memories.add(name)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    attr = self_attr_name(target)
                    if attr is None:
                        continue
                    assigned.add(attr)
                    if is_child_or_buffer_assignment(child.value):
                        child_or_buffer.add(attr)
            elif isinstance(child, ast.AnnAssign):
                attr = self_attr_name(child.target)
                if attr is not None:
                    assigned.add(attr)
                    if is_child_or_buffer_assignment(child.value):
                        child_or_buffer.add(attr)

    info.registered_memory = sorted(memories)
    info.self_assignments = sorted(assigned)
    info.child_or_buffer_candidates = sorted(child_or_buffer)
    info.supported_backends = sorted(backends)
    return info


def load_classes() -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for file in sorted(SOURCE_ROOT.rglob("*.py")):
        if any(part in {"examples"} for part in file.parts):
            continue
        module = module_from_path(file)
        tree = ast.parse(file.read_text(), filename=str(file))
        aliases = import_aliases(tree, module)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                info = scan_class(node, module, file, aliases)
                classes[info.qualname] = info
    return classes


def memorymodule_qualnames(classes: dict[str, ClassInfo]) -> set[str]:
    roots = {f"{MODULE_PREFIX}.base.MemoryModule"}
    changed = True
    while changed:
        changed = False
        for qualname, info in classes.items():
            if qualname in roots:
                continue
            if any(base in roots for base in info.resolved_bases):
                roots.add(qualname)
                changed = True
    roots.remove(f"{MODULE_PREFIX}.base.MemoryModule")
    return roots


def test_references() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    if not TEST_ROOT.exists():
        return refs
    for file in sorted(TEST_ROOT.rglob("test_*.py")):
        tree = ast.parse(file.read_text(), filename=str(file))
        identifiers = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        for class_name in CLASSIFICATION_BY_NAME:
            if class_name in identifiers:
                refs.setdefault(class_name, []).append(str(file.relative_to(REPO_ROOT)))
    return refs


def summarize_names(names: list[str], limit: int = 8) -> str:
    if not names:
        return "-"
    head = names[:limit]
    suffix = "" if len(names) <= limit else f", +{len(names) - limit}"
    return ", ".join(f"`{name}`" for name in head) + suffix


def state_candidates(info: ClassInfo) -> list[str]:
    excluded = {
        "_backend",
        "_step_mode",
        "_memories",
        "_memories_rv",
        "_modules",
        "_parameters",
        "_buffers",
    }
    memory = set(info.registered_memory)
    return [
        name
        for name in info.self_assignments
        if name not in excluded and name not in memory and not name.startswith("__")
    ]


def cache_candidates(info: ClassInfo) -> list[str]:
    needles = ("cache", "kernel", "compiled", "handle", "registry")
    return [
        name
        for name in state_candidates(info)
        if name != "kernel_size" and any(n in name for n in needles)
    ]


def print_markdown(classes: dict[str, ClassInfo], selected: set[str]) -> None:
    refs = test_references()
    print("# MemoryModule inventory")
    print()
    print(
        "Generated from current source by `tools/generate_memorymodule_inventory.py`."
    )
    print()
    print(
        "| Class | File | Plan action | Memory | Plain state candidates | Caches | Tests |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for qualname in sorted(selected):
        info = classes[qualname]
        rel_file = info.file.relative_to(REPO_ROOT)
        classification = CLASSIFICATION_BY_NAME.get(info.name, "unclassified")
        tests = refs.get(info.name, [])
        print(
            "| "
            f"`{info.name}` | "
            f"`{rel_file}:{info.line}` | "
            f"`{classification}` | "
            f"{summarize_names(info.registered_memory)} | "
            f"{summarize_names(state_candidates(info))} | "
            f"{summarize_names(cache_candidates(info))} | "
            f"{summarize_names(tests, limit=3)} |"
        )

    missing = [
        classes[qualname].qualname
        for qualname in sorted(selected)
        if CLASSIFICATION_BY_NAME.get(classes[qualname].name) is None
    ]
    if missing:
        print()
        print("## Unclassified")
        print()
        for qualname in missing:
            print(f"- `{qualname}`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when any production MemoryModule class is unclassified",
    )
    args = parser.parse_args()

    classes = load_classes()
    selected = memorymodule_qualnames(classes)
    print_markdown(classes, selected)

    missing = [
        classes[qualname].qualname
        for qualname in sorted(selected)
        if CLASSIFICATION_BY_NAME.get(classes[qualname].name) is None
    ]
    if args.check and missing:
        raise SystemExit("unclassified MemoryModule classes:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
