import json

import torch

from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
    save_precision_reports,
)


def test_save_precision_reports_writes_expected_files(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    artifacts = prepare_model_for_precision(model, "cpu", PrecisionConfig(mode="fp32"))
    save_precision_reports(artifacts, str(tmp_path))

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "capability_report.json",
        "conversion_report.json",
        "precision_policy.json",
        "precision_runtime.json",
    ]

    runtime = json.loads((tmp_path / "precision_runtime.json").read_text())
    assert set(runtime.keys()) == {
        "requested_config",
        "effective_config",
        "fallback_reason",
    }

    conversion = json.loads((tmp_path / "conversion_report.json").read_text())
    assert "convertible_modules" in conversion
    assert "converted_modules" in conversion
    assert "high_precision_modules" in conversion
    assert "skipped_modules" in conversion
