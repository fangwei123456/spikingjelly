from __future__ import annotations

import argparse
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

from spikingjelly.activation_based import neuron, op_counter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs/source/_static/tutorials/op_counter"
CSV_PATH = OUTPUT_DIR / "energy_model_validation.csv"
FIGURE_PATH = OUTPUT_DIR / "energy_model_validation.png"

SPIKESIM_COMMIT = "c2627bc091a47bdcb630ca6207eaf44a00bd1da4"
NEUROMC_COMMIT = "712c66f47cf76ae530a55f8bcad3858bd68788de"
SENTINEL = "ENERGY_MODEL_VALIDATION_JSON="


@dataclass(frozen=True)
class Score:
    model: str
    case_id: str
    workload: str
    oracle_pj: float
    spikingjelly_pj: float
    oracle_source: str
    oracle_revision: str
    included_in_correlation: bool
    details: dict[str, Any]


SPIKESIM_CASES = (
    ("s01", 3, 16, 24, 3, False, False, 1),
    ("s02", 16, 16, 20, 3, False, False, 1),
    ("s03", 16, 32, 16, 3, True, True, 1),
    ("s04", 32, 64, 12, 3, True, False, 1),
    ("s05", 64, 64, 10, 1, True, True, 1),
    ("s06", 64, 128, 8, 3, False, False, 2),
    ("s07", 128, 128, 6, 3, True, False, 1),
    ("s08", 32, 96, 10, 1, False, False, 3),
    ("s09", 96, 64, 7, 3, True, True, 1),
    ("s10", 8, 32, 18, 3, True, False, 1),
    ("s11", 64, 192, 5, 1, False, False, 1),
    ("s12", 192, 256, 4, 3, True, True, 1),
)

LEMAIRE_CASES = (
    ("l01", "linear", 8, 16, 1, 1, 0.125, False),
    ("l02", "linear", 16, 16, 1, 1, 0.25, True),
    ("l03", "linear", 32, 24, 1, 1, 0.375, False),
    ("l04", "linear", 64, 32, 1, 1, 0.5, True),
    ("l05", "linear", 128, 48, 1, 1, 0.25, False),
    ("l06", "linear", 256, 64, 1, 1, 0.125, True),
    ("l07", "conv", 4, 8, 12, 3, 0.125, False),
    ("l08", "conv", 8, 16, 10, 3, 0.25, True),
    ("l09", "conv", 16, 16, 8, 1, 0.375, False),
    ("l10", "conv", 16, 32, 8, 3, 0.5, True),
    ("l11", "conv", 32, 32, 6, 1, 0.25, False),
    ("l12", "conv", 32, 64, 5, 3, 0.125, True),
)

NEUROMC_LAYERS = (1, 3, 5, 17, 19, 21, 29, 33, 37, 41, 51, 55, 59)


class _ConvWorkload(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        optional: bool,
        execute_optional: bool,
        repeats: int,
    ):
        super().__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.optional = (
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
            if optional
            else None
        )
        self.execute_optional = execute_optional
        self.repeats = repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        for _ in range(1, self.repeats):
            out = out + self.main(x)
        if self.execute_optional:
            out = out + self.optional(x)
        return out


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
    for case_id, cin, cout, dim, kernel, optional, _, _ in SPIKESIM_CASES:
        layer_count = 2 if optional else 1
        payload.append(
            {
                "case_id": case_id,
                "in_channels": [cin] * layer_count,
                "out_channels": [cout] * layer_count,
                "out_dims": [dim] * layer_count,
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
    for (
        case_id,
        cin,
        cout,
        dim,
        kernel,
        optional,
        execute_optional,
        repeats,
    ) in SPIKESIM_CASES:
        model = _ConvWorkload(
            cin, cout, kernel, optional, execute_optional, repeats
        ).eval()
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
                included_in_correlation=(
                    repeats + int(execute_optional) == (2 if optional else 1)
                ),
                details={
                    "in_channels": cin,
                    "out_channels": cout,
                    "output_dim": dim,
                    "kernel": kernel,
                    "static_layers": 2 if optional else 1,
                    "executed_main_calls": repeats,
                    "executed_optional": execute_optional,
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
        # Published Eq. (2), including its N_in factor, is used verbatim.
        acc_ops = theta_in * in_features * out_features + time_steps * out_features
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
                included_in_correlation=True,
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
for layer in json.load(sys.stdin):
    workload_name = (
        "zigzag.inputs.examples.workload."
        f"S-ResNet18_imagenet_B16_T4_split.fe_l{{layer}}"
    )
    mapping_name = (
        "zigzag.inputs.examples.mapping."
        f"aicore_S-ResNet18_imagenet_B16_T4_split_ws.fe_l{{layer}}"
    )
    answers = MainStage(
        [
            AcceleratorParserStage,
            WorkloadParserStage,
            WorkloadStage,
            SpatialMappingConversionStage,
            TemporalOrderingConversionStage,
            CostModelStage,
        ],
        accelerator="zigzag.inputs.examples.hardware.aicore.aicore_fe",
        workload=workload_name,
        mapping=mapping_name,
        access_same_data_considered_as_no_access=True,
    ).run()
    item = importlib.import_module(workload_name).workload[0]
    results.append(
        {{
            "layer": layer,
            "energy": answers[0][0].energy_total,
            "dims": item["loop_dim_size"],
            "operator_type": item["operator_type"],
        }}
    )
print({SENTINEL!r} + json.dumps(results))
"""
    return _run_author_adapter(adapter, NEUROMC_LAYERS, codes)


def _neuromc_scores(repo: Path) -> list[Score]:
    scores = []
    for item in _neuromc_author_scores(repo):
        dims = item["dims"]
        cin, cout = int(dims["C"]), int(dims["K"])
        fy, fx = int(dims["FY"]), int(dims["FX"])
        oy, ox = int(dims["OY"]), int(dims["OX"])
        model = nn.Conv2d(cin, cout, (fy, fx), bias=False).eval()
        x = torch.zeros(1, cin, oy + fy - 1, ox + fx - 1)
        report = op_counter.estimate_neuromc_runtime_energy(model, x)
        scores.append(
            Score(
                model="NeuroMC",
                case_id=f"n{item['layer']:02d}",
                workload="official S-ResNet18 FE fragment",
                oracle_pj=float(item["energy"]),
                spikingjelly_pj=report.energy_total_pj,
                oracle_source="author ZigZag fixed workload/mapping/cost model",
                oracle_revision=NEUROMC_COMMIT,
                included_in_correlation=True,
                details={
                    "official_layer": item["layer"],
                    "operator_type": item["operator_type"],
                    **{key: int(value) for key, value in dims.items()},
                },
            )
        )
    return scores


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
    oracle = np.asarray([score.oracle_pj for score in scores], dtype=float)
    dynamic = np.asarray([score.spikingjelly_pj for score in scores], dtype=float)
    log_ratio = np.log(dynamic / oracle)
    calibrated_log_error = log_ratio - np.median(log_ratio)
    factors = np.exp(np.abs(calibrated_log_error))
    tau = float(kendalltau(oracle, dynamic).statistic)
    ci_low, ci_high = _bootstrap_tau(oracle, dynamic)
    return {
        "kendall_tau_b": tau,
        "kendall_ci_low": ci_low,
        "kendall_ci_high": ci_high,
        "spearman_rho": float(spearmanr(oracle, dynamic).statistic),
        "log_pearson_r": float(pearsonr(np.log(oracle), np.log(dynamic)).statistic),
        "p90_factor": float(np.percentile(factors, 90)),
        "scale_ratio": float(np.exp(np.median(log_ratio))),
    }


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
        "included_in_correlation",
        "details_json",
        "kendall_tau_b",
        "kendall_ci_low",
        "kendall_ci_high",
        "spearman_rho",
        "log_pearson_r",
        "p90_factor",
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
                    "included_in_correlation": score.included_in_correlation,
                    "details_json": json.dumps(score.details, sort_keys=True),
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


def _plot(scores: list[Score], metrics: dict[str, dict[str, float]]) -> None:
    models = list(metrics)
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, (scatter, bars) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model, color in zip(models, colors, strict=True):
        group = [score for score in scores if score.model == model]
        included = [score for score in group if score.included_in_correlation]
        excluded = [score for score in group if not score.included_in_correlation]
        oracle_scale = np.exp(np.mean(np.log([score.oracle_pj for score in included])))
        dynamic_scale = np.exp(
            np.mean(np.log([score.spikingjelly_pj for score in included]))
        )
        oracle = np.asarray([score.oracle_pj / oracle_scale for score in included])
        dynamic = np.asarray(
            [score.spikingjelly_pj / dynamic_scale for score in included]
        )
        scatter.scatter(oracle, dynamic, label=model, color=color, s=38, alpha=0.85)
        if excluded:
            scatter.scatter(
                [score.oracle_pj / oracle_scale for score in excluded],
                [score.spikingjelly_pj / dynamic_scale for score in excluded],
                color=color,
                marker="x",
                s=46,
                label=f"{model} dynamic stress",
            )
    limits = scatter.get_xlim()
    lower = min(limits[0], scatter.get_ylim()[0])
    upper = max(limits[1], scatter.get_ylim()[1])
    scatter.plot([lower, upper], [lower, upper], color="#555555", linestyle="--")
    scatter.set_xscale("log")
    scatter.set_yscale("log")
    scatter.set_xlabel("Author/paper score (geometric-mean normalized)")
    scatter.set_ylabel("SpikingJelly score (geometric-mean normalized)")
    scatter.legend(frameon=False)
    scatter.grid(alpha=0.2)

    x = np.arange(len(models))
    tau = [metrics[model]["kendall_tau_b"] for model in models]
    p90 = [metrics[model]["p90_factor"] for model in models]
    bars.bar(
        x - 0.18,
        tau,
        width=0.36,
        color="#56B4E9",
        label="Kendall tau-b",
    )
    bars.bar(
        x + 0.18,
        np.asarray(p90) - 1.0,
        width=0.36,
        color="#E69F00",
        label="P90 factor - 1",
    )
    bars.axhline(0.8, color="#555555", linestyle="--", linewidth=1)
    bars.set_xticks(x, models, rotation=15, ha="right")
    bars.set_ylim(0, max(1.05, max(p90)))
    bars.set_ylabel("Scale-free validation metric")
    bars.legend(frameon=False)
    bars.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


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
        model: _metrics(
            [
                score
                for score in scores
                if score.model == model and score.included_in_correlation
            ]
        )
        for model in ("SpikeSim dense", "Lemaire", "NeuroMC")
    }
    revision = _revision(ROOT)
    _write_csv(scores, metrics, revision)
    _plot(scores, metrics)

    for model, values in metrics.items():
        included_count = sum(
            score.model == model and score.included_in_correlation for score in scores
        )
        print(
            f"{model}: n={included_count}, "
            f"tau-b={values['kendall_tau_b']:.3f} "
            f"[{values['kendall_ci_low']:.3f}, {values['kendall_ci_high']:.3f}], "
            f"rho={values['spearman_rho']:.3f}, "
            f"log-r={values['log_pearson_r']:.3f}, "
            f"P90={values['p90_factor']:.3f}x, "
            f"raw-scale={values['scale_ratio']:.3g}x"
        )
    stress_ratios = [
        score.spikingjelly_pj / score.oracle_pj
        for score in scores
        if not score.included_in_correlation
    ]
    print(
        f"SpikeSim dynamic stress: n={len(stress_ratios)}, "
        f"runtime/oracle={min(stress_ratios):.3f}x..{max(stress_ratios):.3f}x"
    )
    print(CSV_PATH)
    print(FIGURE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
