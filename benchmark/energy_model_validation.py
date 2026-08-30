from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spikingjelly.activation_based import neuron, op_counter


OUTPUT_DIR = ROOT / "docs/source/_static/tutorials/op_counter"
CSV_PATH = OUTPUT_DIR / "energy_model_validation.csv"
CROSS_VALIDATION_CSV_PATH = OUTPUT_DIR / "energy_model_cross_validation.csv"
CROSS_VALIDATION_FIGURE_PATH = OUTPUT_DIR / "energy_model_cross_validation.png"

SPIKESIM_COMMIT = "c2627bc091a47bdcb630ca6207eaf44a00bd1da4"
NEUROMC_COMMIT = "712c66f47cf76ae530a55f8bcad3858bd68788de"
SENTINEL = "ENERGY_MODEL_VALIDATION_JSON="
AUTHOR_ADAPTER_TIMEOUT_SECONDS = 600

MIN_COMPARABLE_CASES = {
    "SpikeSim dense": 200,
    "Lemaire": 100,
    "NeuroMC": 786,
}
NEUROMC_PHASES = ("fe", "be", "we")
NEUROMC_CASES_PER_PHASE = 262
NEUROMC_TAU_MIN = 0.90
NEUROMC_RAW_P90_MAX = 1.50
NEUROMC_SCALE_RANGE = (0.80, 1.25)
CROSS_ESTIMATORS = (
    "Simple",
    "Lemaire",
    "NeuroMC",
    "SpikeSim dense",
    "SpikeSim event",
)


@dataclass(frozen=True)
class Score:
    model: str
    case_id: str
    workload: str
    oracle_pj: float
    spikingjelly_pj: float
    oracle_source: str
    oracle_revision: str
    details: dict[str, Any]


_SPIKESIM_CHANNELS = (
    (3, 8),
    (3, 16),
    (8, 16),
    (8, 32),
    (16, 16),
    (16, 32),
    (16, 64),
    (32, 32),
    (32, 64),
    (32, 96),
    (64, 64),
    (64, 128),
)
SPIKESIM_CASES = tuple(
    (f"s{index:03d}", cin, cout, dim, kernel)
    for index, (cin, cout, dim, kernel) in enumerate(
        (
            (cin, cout, dim, kernel)
            for cin, cout in _SPIKESIM_CHANNELS
            for dim in (4, 6, 8, 10, 12, 16)
            for kernel in (1, 3, 5)
        ),
        1,
    )
)


def _lemaire_cases() -> tuple[tuple[Any, ...], ...]:
    cases = [
        ("linear", cin, cout, 1, 1, activity, lif)
        for cin, cout in ((8, 16), (16, 32), (32, 16), (64, 64), (128, 32), (256, 64))
        for activity in (0.0625, 0.125, 0.25, 0.5)
        for lif in (False, True)
    ]
    cases.extend(
        ("conv", cin, cout, spatial, kernel, activity, lif)
        for cin, cout in ((4, 8), (8, 16), (16, 16), (16, 32), (32, 64))
        for spatial in (5, 8, 12)
        for kernel in (1, 3)
        for activity in (0.0625, 0.125, 0.25, 0.5)
        for lif in (False, True)
    )
    return tuple((f"l{index:03d}", *case) for index, case in enumerate(cases, 1))


LEMAIRE_CASES = _lemaire_cases()

NEUROMC_SUITES = (
    (
        "resnet18_ws",
        "S-ResNet18_imagenet_B16_T4_split",
        "aicore_S-ResNet18_imagenet_B16_T4_split_ws",
    ),
    (
        "resnet50_ws",
        "S-ResNet50_imagenet_B16_T4_split",
        "aicore_S-ResNet50_imagenet_B16_T4_split_ws",
    ),
    (
        "vgg16_ws",
        "S-VGG16_imagenet_B16_T4_split",
        "aicore_S-VGG16_imagenet_B16_T4_split_ws",
    ),
)


def _revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _require_revision(repo: Path, expected: str, source: str) -> None:
    if not repo.is_dir():
        raise FileNotFoundError(f"{source} repository not found: {repo}")
    actual = _revision(repo)
    if actual != expected:
        raise ValueError(f"{source} must be checked out at {expected}, got {actual}.")


def _run_author_adapter(code: str, payload: Any, cwd: Path) -> Any:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env={**os.environ, "MPLBACKEND": "Agg"},
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=AUTHOR_ADAPTER_TIMEOUT_SECONDS,
    )
    line = next(
        (
            line
            for line in reversed(result.stdout.splitlines())
            if line.startswith(SENTINEL)
        ),
        None,
    )
    if line is None:
        raise RuntimeError(
            f"Author adapter returned no result. stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}"
        )
    return json.loads(line.removeprefix(SENTINEL))


def _spikesim_scores(repo: Path) -> list[Score]:
    source = repo / "SNN_train_infer_quantization_ela/ela_spikesim.py"
    payload = []
    for case_id, cin, cout, dim, kernel in SPIKESIM_CASES:
        payload.append(
            {
                "case_id": case_id,
                "in_channels": [cin],
                "out_channels": [cout],
                "out_dims": [dim],
                "kernel": kernel,
            }
        )
    adapter = f"""
import contextlib
import io
import json
import runpy
import sys

payload = json.load(sys.stdin)
with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path({str(source)!r})
    compute_energy = namespace["compute_energy"]
    results = []
    for case in payload:
        layerwise = compute_energy(
            case["in_channels"],
            case["out_dims"],
            case["out_channels"],
            case["out_dims"],
            64,
            case["kernel"],
            0,
            "rram",
            1,
        )
        results.append({{"case_id": case["case_id"], "energy": sum(layerwise)}})
print({SENTINEL!r} + json.dumps(results))
"""
    author = {
        item["case_id"]: float(item["energy"])
        for item in _run_author_adapter(adapter, payload, repo)
    }

    scores = []
    for case_id, cin, cout, dim, kernel in SPIKESIM_CASES:
        model = nn.Conv2d(cin, cout, kernel, bias=False).eval()
        x = torch.zeros(1, cin, dim + kernel - 1, dim + kernel - 1)
        report = op_counter.estimate_spikesim_energy(model, x)
        scores.append(
            Score(
                model="SpikeSim dense",
                case_id=case_id,
                workload="static topology vs executed calls",
                oracle_pj=author[case_id],
                spikingjelly_pj=report.energy_total_pj,
                oracle_source="author code: ela_spikesim.py compute_energy",
                oracle_revision=SPIKESIM_COMMIT,
                details={
                    "in_channels": cin,
                    "out_channels": cout,
                    "output_dim": dim,
                    "kernel": kernel,
                },
            )
        )
    return scores


def _binary_input(shape: tuple[int, ...], activity: float, seed: int) -> torch.Tensor:
    total = math.prod(shape)
    active = max(1, round(total * activity))
    generator = torch.Generator().manual_seed(seed)
    flat = torch.zeros(total)
    flat[torch.randperm(total, generator=generator)[:active]] = 1.0
    return flat.reshape(shape)


def _lemaire_memory_cost(capacity_bytes: int) -> float:
    points = (
        (0.0, 0.0),
        (8192.0, 10.0),
        (32768.0, 20.0),
        (1048576.0, 100.0),
    )
    capacity = min(max(float(capacity_bytes), 0.0), points[-1][0])
    for left, right in zip(points, points[1:], strict=True):
        if capacity <= right[0]:
            return left[1] + (right[1] - left[1]) * (
                (capacity - left[0]) / (right[0] - left[0])
            )
    raise AssertionError("unreachable")


def _lemaire_paper_energy(
    *,
    kind: str,
    in_features: int,
    out_features: int,
    spatial: int,
    kernel: int,
    theta_in: int,
    theta_out: int,
    lif: bool,
) -> float:
    time_steps = 1
    output_neurons = out_features * spatial * spatial
    if kind == "linear":
        # Eq. (2) includes an extra N_in factor, although theta_in is defined as
        # the input-spike count and Eqs. (8), (10), (15), and (17) use theta_in
        # times N_out. Count each observed input spike's N_out fanout once.
        acc_ops = theta_in * out_features + time_steps * out_features
        mac_ops = time_steps * out_features if lif else 0
        acc_addr = theta_in * out_features
        mac_addr = 0
        parameter_accesses = theta_in * out_features + time_steps * out_features
        potential_accesses = theta_in * out_features + time_steps * out_features
    else:
        fanout = out_features * kernel * kernel
        acc_ops = (
            theta_in * kernel * kernel * out_features
            + time_steps * output_neurons
            + theta_out
        )
        mac_ops = time_steps * output_neurons if lif else 0
        acc_addr = theta_in * fanout
        mac_addr = theta_in * 2
        parameter_accesses = theta_in * fanout + time_steps * output_neurons
        potential_accesses = theta_in * fanout + time_steps * output_neurons

    parameter_capacity = (
        in_features * out_features * kernel * kernel + out_features
    ) * 4
    potential_capacity = output_neurons * 4
    fifo_capacity = 1000 * 4
    memory_pj = (
        (theta_in + theta_out) * _lemaire_memory_cost(fifo_capacity)
        + parameter_accesses * _lemaire_memory_cost(parameter_capacity)
        + 2 * potential_accesses * _lemaire_memory_cost(potential_capacity)
    )
    compute_pj = (acc_ops + acc_addr) * 0.1 + (mac_ops + mac_addr) * 3.2
    return compute_pj + memory_pj


def _lemaire_model(
    kind: str,
    in_features: int,
    out_features: int,
    spatial: int,
    kernel: int,
    lif: bool,
) -> nn.Sequential:
    if kind == "linear":
        synapse: nn.Module = nn.Linear(in_features, out_features, bias=True)
    else:
        synapse = nn.Conv2d(
            in_features,
            out_features,
            kernel,
            padding=kernel // 2,
            bias=True,
        )
    node: nn.Module
    if lif:
        node = neuron.LIFNode(v_threshold=1.0, detach_reset=True)
    else:
        node = neuron.IFNode(v_threshold=1.0, detach_reset=True)
    model = nn.Sequential(synapse, node).eval()
    with torch.no_grad():
        synapse.weight.fill_(0.125)
        synapse.bias.zero_()
    return model


def _lemaire_scores() -> list[Score]:
    scores = []
    for index, case in enumerate(LEMAIRE_CASES):
        case_id, kind, cin, cout, spatial, kernel, activity, lif = case
        shape = (1, cin) if kind == "linear" else (1, cin, spatial, spatial)
        x = _binary_input(shape, activity, index)

        observation_model = _lemaire_model(kind, cin, cout, spatial, kernel, lif)
        with torch.no_grad():
            observed_out = observation_model(x)
        theta_in = int(x.count_nonzero().item())
        theta_out = int(observed_out.count_nonzero().item())
        oracle = _lemaire_paper_energy(
            kind=kind,
            in_features=cin,
            out_features=cout,
            spatial=spatial,
            kernel=kernel,
            theta_in=theta_in,
            theta_out=theta_out,
            lif=lif,
        )

        dynamic_model = _lemaire_model(kind, cin, cout, spatial, kernel, lif)
        report = op_counter.estimate_lemaire_energy(dynamic_model, x)
        scores.append(
            Score(
                model="Lemaire",
                case_id=case_id,
                workload=f"{kind} + {'LIF' if lif else 'IF'}",
                oracle_pj=oracle,
                spikingjelly_pj=report.total_pj,
                oracle_source="paper equations (1)-(20); no released code",
                oracle_revision="arXiv:2210.13107v1",
                details={
                    "kind": kind,
                    "in_features": cin,
                    "out_features": cout,
                    "spatial": spatial,
                    "kernel": kernel,
                    "activity": activity,
                    "theta_in": theta_in,
                    "theta_out": theta_out,
                    "neuron": "LIF" if lif else "IF",
                },
            )
        )
    return scores


def _neuromc_cases(repo: Path) -> list[dict[str, Any]]:
    examples = repo / "code/hardware_dse_simulator/codes/zigzag/inputs/examples"
    cases = []
    for suite, workload, mapping in NEUROMC_SUITES:
        workload_dir = examples / "workload" / workload
        mapping_dir = examples / "mapping" / mapping
        for phase in NEUROMC_PHASES:
            prefix = f"{phase}_l"
            workload_layers = {path.stem for path in workload_dir.glob(f"{prefix}*.py")}
            mapping_layers = {path.stem for path in mapping_dir.glob(f"{prefix}*.py")}
            for stem in sorted(
                workload_layers & mapping_layers,
                key=lambda value: int(value.removeprefix(prefix)),
            ):
                cases.append(
                    {
                        "suite": suite,
                        "workload": workload,
                        "mapping": mapping,
                        "phase": phase,
                        "layer": int(stem.removeprefix(prefix)),
                    }
                )
    return cases


def _neuromc_author_scores(repo: Path) -> list[dict[str, Any]]:
    codes = repo / "code/hardware_dse_simulator/codes"
    adapter = f"""
import importlib
import json
import multiprocessing
import sys
import types
import warnings

# These packages are imported by ZigZag's stage package but are not used by
# the fixed, non-MPI workload/mapping/cost-model chain below.
mpi4py = types.ModuleType("mpi4py")
mpi4py.MPI = types.SimpleNamespace()
onnx = types.ModuleType("onnx")
onnx.ModelProto = type("ModelProto", (), {{}})
onnx.AttributeProto = type(
    "AttributeProto",
    (),
    {{"AttributeType": types.SimpleNamespace(INT=1, INTS=2)}},
)
yaml = types.ModuleType("yaml")
sys.modules.update(
    {{
        "mpi4py": mpi4py,
        "onnx": onnx,
        "yaml": yaml,
        "multiprocessing_on_dill": multiprocessing,
    }}
)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

from zigzag.classes.stages import (
    AcceleratorParserStage,
    CostModelStage,
    MainStage,
    SpatialMappingConversionStage,
    TemporalOrderingConversionStage,
    WorkloadParserStage,
    WorkloadStage,
)

results = []
for case in json.load(sys.stdin):
    layer = case["layer"]
    phase = case["phase"]
    workload_name = (
        "zigzag.inputs.examples.workload."
        f"{{case['workload']}}.{{phase}}_l{{layer}}"
    )
    mapping_name = (
        "zigzag.inputs.examples.mapping."
        f"{{case['mapping']}}.{{phase}}_l{{layer}}"
    )
    accelerator_name = f"zigzag.inputs.examples.hardware.aicore.aicore_{{phase}}"
    for module_name in (accelerator_name, workload_name, mapping_name):
        importlib.reload(importlib.import_module(module_name))
    item = importlib.import_module(workload_name).workload[0]
    answers = MainStage(
        [
            AcceleratorParserStage,
            WorkloadParserStage,
            WorkloadStage,
            SpatialMappingConversionStage,
            TemporalOrderingConversionStage,
            CostModelStage,
        ],
        accelerator=accelerator_name,
        workload=workload_name,
        mapping=mapping_name,
        access_same_data_considered_as_no_access=True,
    ).run()
    results.append(
        {{
            "suite": case["suite"],
            "phase": phase,
            "layer": layer,
            "energy": answers[0][0].energy_total,
            "compute_energy": answers[0][0].MAC_energy
                + answers[0][0].energy_compute_extra[answers[0][0].energy_type],
            "memory_energy": answers[0][0].mem_energy,
            "energy_type": answers[0][0].energy_type,
            "dims": item["loop_dim_size"],
            "operator_type": item["operator_type"],
            "b_type": item["B_type"],
            "t_type": item["T_type"],
            "conv_type": item["conv_type"],
        }}
    )
print({SENTINEL!r} + json.dumps(results))
"""
    cases = _neuromc_cases(repo)
    results = []
    for suite, _, _ in NEUROMC_SUITES:
        payload = [case for case in cases if case["suite"] == suite]
        results.extend(_run_author_adapter(adapter, payload, codes))
    return results


def _neuromc_runtime_score(item: dict[str, Any]) -> float:
    phase = item["phase"]
    dims = item["dims"]
    cin, cout = int(dims["C"]), int(dims["K"])
    if phase == "we":
        fy, fx = int(dims["OY"]), int(dims["OX"])
        oy, ox = int(dims["FY"]), int(dims["FX"])
    else:
        fy, fx = int(dims["FY"]), int(dims["FX"])
        oy, ox = int(dims["OY"]), int(dims["OX"])
    conv = nn.Conv2d(
        cin,
        cout,
        (fy, fx),
        padding=(fy // 2, fx // 2),
        bias=False,
    )
    model = (
        nn.Sequential(conv, neuron.IFNode())
        if phase == "fe"
        else nn.Sequential(neuron.IFNode(), conv)
    )
    model.train(phase != "fe")
    x = torch.zeros(
        1,
        cin,
        oy,
        ox,
        requires_grad=phase != "fe",
    )
    if phase == "fe":
        report = op_counter.estimate_neuromc_runtime_energy(model, x)
        return report.energy_by_core_type["fp_soma"]
    report = op_counter.estimate_neuromc_runtime_energy(
        model,
        x,
        target=torch.empty(0),
        loss_fn=lambda output, target: output.sum(),
    )
    return report.energy_by_core_type["bp_grad" if phase == "be" else "wg"]


def _neuromc_scores(repo: Path) -> list[Score]:
    scores = []
    for item in _neuromc_author_scores(repo):
        dims = item["dims"]
        scores.append(
            Score(
                model="NeuroMC",
                case_id=f"{item['suite']}_{item['phase']}_l{item['layer']}",
                workload=f"official {item['suite']} {item['phase'].upper()} fragment",
                oracle_pj=float(item["energy"]),
                spikingjelly_pj=_neuromc_runtime_score(item),
                oracle_source="author ZigZag fixed workload/mapping/cost model",
                oracle_revision=NEUROMC_COMMIT,
                details={
                    "suite": item["suite"],
                    "phase": item["phase"],
                    "official_layer": item["layer"],
                    "operator_type": item["operator_type"],
                    "b_type": item["b_type"],
                    "t_type": item["t_type"],
                    "conv_type": item["conv_type"],
                    "oracle_compute_pj": item["compute_energy"],
                    "oracle_memory_pj": item["memory_energy"],
                    "oracle_energy_type": item["energy_type"],
                    **{key: int(value) for key, value in dims.items()},
                },
            )
        )
    return scores


def _capture_conv2d_stages(
    model: nn.Module, inputs: torch.Tensor
) -> list[tuple[nn.Conv2d, torch.Tensor]]:
    stages = []

    def capture(module, args):
        stages.append((module, args[0].detach().clone()))

    handles = [
        module.register_forward_pre_hook(capture)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return stages


def _cross_validation_networks():
    from spikingjelly.activation_based.model import sew_resnet, spiking_vgg

    builders = (
        ("VGG-11", spiking_vgg.spiking_vgg11),
        ("VGG-13", spiking_vgg.spiking_vgg13),
        ("VGG-16", spiking_vgg.spiking_vgg16),
        ("VGG-19", spiking_vgg.spiking_vgg19),
        ("SEW-ResNet-18", sew_resnet.sew_resnet18),
        ("SEW-ResNet-34", sew_resnet.sew_resnet34),
        ("SEW-ResNet-50", sew_resnet.sew_resnet50),
    )
    for index, (name, builder) in enumerate(builders):
        for image_size in (32, 40, 48, 56):
            torch.manual_seed(20260830 + index * 2 + image_size)
            kwargs = {
                "num_classes": 10,
                "spiking_neuron": neuron.IFNode,
                "step_mode": "s",
            }
            if name.startswith("SEW"):
                kwargs.update(
                    {
                        "cnf": "ADD",
                        "detach_reset": True,
                        "norm_layer": lambda _: nn.Identity(),
                    }
                )
            model = builder(**kwargs).eval()
            inputs = torch.randn(1, 3, image_size, image_size)
            yield f"{name}-{image_size}", name, image_size, model, inputs


def _estimate_cross_stage(conv: nn.Conv2d, inputs: torch.Tensor) -> dict[str, float]:
    def stage():
        model = nn.Sequential(copy.deepcopy(conv), neuron.IFNode()).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    return {
        "Simple": op_counter.estimate_simple_energy(stage(), inputs).energy_total_pj,
        "Lemaire": op_counter.estimate_lemaire_energy(stage(), inputs).total_pj,
        "NeuroMC": op_counter.estimate_neuromc_runtime_energy(
            stage(), inputs
        ).energy_total_pj,
        "SpikeSim dense": op_counter.estimate_spikesim_energy(
            stage(), inputs
        ).energy_total_pj,
        "SpikeSim event": op_counter.estimate_spikesim_energy(
            stage(),
            inputs,
            config=op_counter.SpikeSimEnergyConfig(activity_mode="event"),
        ).energy_total_pj,
    }


def _energy_model_cross_validation() -> list[dict[str, Any]]:
    rows = []
    for case_id, family, image_size, model, inputs in _cross_validation_networks():
        totals = {name: 0.0 for name in CROSS_ESTIMATORS}
        stages = _capture_conv2d_stages(model, inputs)
        for conv, stage_inputs in stages:
            for name, value in _estimate_cross_stage(conv, stage_inputs).items():
                totals[name] += value
        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "image_size": image_size,
                "conv2d_stages": len(stages),
                **totals,
            }
        )
    return rows


def _cross_validation_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in CROSS_ESTIMATORS
    }
    size = len(CROSS_ESTIMATORS)
    metrics = {
        "Kendall tau-b": np.eye(size),
        "Spearman rho": np.eye(size),
        "Log-Pearson r": np.eye(size),
    }
    for i, left in enumerate(CROSS_ESTIMATORS):
        for j, right in enumerate(CROSS_ESTIMATORS):
            if i >= j:
                continue
            pairs = {
                "Kendall tau-b": kendalltau(values[left], values[right]).statistic,
                "Spearman rho": spearmanr(values[left], values[right]).statistic,
                "Log-Pearson r": pearsonr(
                    np.log(values[left]), np.log(values[right])
                ).statistic,
            }
            for name, value in pairs.items():
                metrics[name][i, j] = metrics[name][j, i] = float(value)
    return metrics


def _write_cross_validation_csv(rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "family",
        "image_size",
        "conv2d_stages",
        *CROSS_ESTIMATORS,
    )
    with CROSS_VALIDATION_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **{name: f"{row[name]:.12g}" for name in CROSS_ESTIMATORS},
                }
            )


def _plot_cross_validation(metrics: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    labels = ("Simple", "Lemaire", "NeuroMC", "SS dense", "SS event")
    image = None
    for index, (axis, (name, matrix)) in enumerate(
        zip(axes, metrics.items(), strict=True)
    ):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(name)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        if index:
            axis.tick_params(axis="y", left=False, labelleft=False)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[row, column]) > 0.65 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=axes, shrink=0.78, label="Correlation")
    fig.savefig(CROSS_VALIDATION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_tau(
    oracle: np.ndarray, dynamic: np.ndarray, samples: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260826)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(oracle), len(oracle))
        value = kendalltau(oracle[indices], dynamic[indices]).statistic
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return math.nan, math.nan
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def _metrics(scores: list[Score]) -> dict[str, float]:
    if not scores:
        raise ValueError("metrics require at least one score")
    oracle = np.asarray([score.oracle_pj for score in scores], dtype=float)
    dynamic = np.asarray([score.spikingjelly_pj for score in scores], dtype=float)
    if not np.all(np.isfinite(oracle) & (oracle > 0)) or not np.all(
        np.isfinite(dynamic) & (dynamic > 0)
    ):
        raise ValueError("energy scores must be finite and positive")
    log_ratio = np.log(dynamic / oracle)
    calibrated_log_error = log_ratio - np.median(log_ratio)
    raw_factors = np.exp(np.abs(log_ratio))
    scale_adjusted_factors = np.exp(np.abs(calibrated_log_error))
    tau = float(kendalltau(oracle, dynamic).statistic)
    ci_low, ci_high = _bootstrap_tau(oracle, dynamic)
    return {
        "kendall_tau_b": tau,
        "kendall_ci_low": ci_low,
        "kendall_ci_high": ci_high,
        "spearman_rho": float(spearmanr(oracle, dynamic).statistic),
        "log_pearson_r": float(pearsonr(np.log(oracle), np.log(dynamic)).statistic),
        "raw_p90_factor": float(np.percentile(raw_factors, 90)),
        "scale_adjusted_p90_factor": float(np.percentile(scale_adjusted_factors, 90)),
        "scale_ratio": float(np.exp(np.median(log_ratio))),
    }


def _neuromc_phase_metrics(scores: list[Score]) -> dict[str, dict[str, float]]:
    results = {}
    for phase in NEUROMC_PHASES:
        phase_scores = [
            score
            for score in scores
            if score.model == "NeuroMC" and score.details["phase"] == phase
        ]
        if len(phase_scores) != NEUROMC_CASES_PER_PHASE:
            raise RuntimeError(
                f"NeuroMC {phase.upper()} requires {NEUROMC_CASES_PER_PHASE} "
                f"cases, got {len(phase_scores)}."
            )
        values = _metrics(phase_scores)
        if values["kendall_tau_b"] < NEUROMC_TAU_MIN:
            raise RuntimeError(
                f"NeuroMC {phase.upper()} tau-b={values['kendall_tau_b']:.3f} "
                f"is below {NEUROMC_TAU_MIN:.2f}."
            )
        if values["raw_p90_factor"] > NEUROMC_RAW_P90_MAX:
            raise RuntimeError(
                f"NeuroMC {phase.upper()} raw-P90={values['raw_p90_factor']:.3f}x "
                f"exceeds {NEUROMC_RAW_P90_MAX:.2f}x."
            )
        if (
            not NEUROMC_SCALE_RANGE[0]
            <= values["scale_ratio"]
            <= NEUROMC_SCALE_RANGE[1]
        ):
            raise RuntimeError(
                f"NeuroMC {phase.upper()} median scale={values['scale_ratio']:.3f}x "
                f"is outside {NEUROMC_SCALE_RANGE}."
            )
        results[phase] = values
    return results


def _write_csv(
    scores: list[Score], metrics: dict[str, dict[str, float]], revision: str
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "case_id",
        "workload",
        "oracle_pj",
        "spikingjelly_pj",
        "oracle_source",
        "oracle_revision",
        "details_json",
        "ratio_spikingjelly_over_oracle",
        "symmetric_error_factor",
        "kendall_tau_b",
        "kendall_ci_low",
        "kendall_ci_high",
        "spearman_rho",
        "log_pearson_r",
        "raw_p90_factor",
        "scale_adjusted_p90_factor",
        "scale_ratio_spikingjelly_over_oracle",
        "spikingjelly_revision",
        "python_version",
        "torch_version",
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for score in scores:
            summary = metrics[score.model]
            writer.writerow(
                {
                    "model": score.model,
                    "case_id": score.case_id,
                    "workload": score.workload,
                    "oracle_pj": f"{score.oracle_pj:.12g}",
                    "spikingjelly_pj": f"{score.spikingjelly_pj:.12g}",
                    "oracle_source": score.oracle_source,
                    "oracle_revision": score.oracle_revision,
                    "details_json": json.dumps(score.details, sort_keys=True),
                    "ratio_spikingjelly_over_oracle": (
                        f"{score.spikingjelly_pj / score.oracle_pj:.12g}"
                    ),
                    "symmetric_error_factor": (
                        f"{math.exp(abs(math.log(score.spikingjelly_pj / score.oracle_pj))):.12g}"
                    ),
                    **{
                        key: f"{value:.12g}"
                        for key, value in summary.items()
                        if key != "scale_ratio"
                    },
                    "scale_ratio_spikingjelly_over_oracle": (
                        f"{summary['scale_ratio']:.12g}"
                    ),
                    "spikingjelly_revision": revision,
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SpikingJelly runtime energy scores with pinned external models."
        )
    )
    parser.add_argument(
        "--spikesim-root",
        type=Path,
        default=os.environ.get("SPIKESIM_ROOT"),
        required=os.environ.get("SPIKESIM_ROOT") is None,
    )
    parser.add_argument(
        "--neuromc-root",
        type=Path,
        default=os.environ.get("NEUROMC_ROOT"),
        required=os.environ.get("NEUROMC_ROOT") is None,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spikesim_root = args.spikesim_root.resolve()
    neuromc_root = args.neuromc_root.resolve()
    _require_revision(spikesim_root, SPIKESIM_COMMIT, "SpikeSim")
    _require_revision(neuromc_root, NEUROMC_COMMIT, "NeuroMC")

    scores = [
        *_spikesim_scores(spikesim_root),
        *_lemaire_scores(),
        *_neuromc_scores(neuromc_root),
    ]
    metrics = {
        model: _metrics([score for score in scores if score.model == model])
        for model in ("SpikeSim dense", "Lemaire", "NeuroMC")
    }
    for model, minimum in MIN_COMPARABLE_CASES.items():
        actual = sum(score.model == model for score in scores)
        if actual < minimum:
            raise RuntimeError(
                f"{model} requires at least {minimum} comparable cases, got {actual}."
            )
    neuromc_phases = _neuromc_phase_metrics(scores)
    revision = _revision(ROOT)
    _write_csv(scores, metrics, revision)
    cross_rows = _energy_model_cross_validation()
    cross_metrics = _cross_validation_metrics(cross_rows)
    _write_cross_validation_csv(cross_rows)
    _plot_cross_validation(cross_metrics)

    for model, values in metrics.items():
        case_count = sum(score.model == model for score in scores)
        print(
            f"{model}: n={case_count}, "
            f"tau-b={values['kendall_tau_b']:.3f} "
            f"[{values['kendall_ci_low']:.3f}, {values['kendall_ci_high']:.3f}], "
            f"rho={values['spearman_rho']:.3f}, "
            f"log-r={values['log_pearson_r']:.3f}, "
            f"raw-P90={values['raw_p90_factor']:.3f}x, "
            f"scale-adjusted-P90={values['scale_adjusted_p90_factor']:.3f}x, "
            f"raw-scale={values['scale_ratio']:.3g}x"
        )
    for phase, values in neuromc_phases.items():
        print(
            f"NeuroMC {phase.upper()}: n={NEUROMC_CASES_PER_PHASE}, "
            f"tau-b={values['kendall_tau_b']:.3f}, "
            f"raw-P90={values['raw_p90_factor']:.3f}x, "
            f"raw-scale={values['scale_ratio']:.3f}x"
        )
    print(CSV_PATH)
    print(CROSS_VALIDATION_CSV_PATH)
    print(CROSS_VALIDATION_FIGURE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
