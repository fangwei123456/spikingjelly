"""Shared quality contract for Qwen evaluation and aggregation."""

from __future__ import annotations

from typing import Mapping


TASKS = (
    "lambada_openai",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
)


def validate_quality(report: Mapping[str, object]) -> None:
    ppl = report["quality"]["wikitext"]
    if float(ppl["relative_degradation"]) > 0.15:
        raise ValueError("SNN WikiText PPL degradation exceeds 15%.")
    tasks = report["quality"].get("zero_shot")
    if tasks is None:
        return
    if float(tasks["mean_drop_percentage_points"]) > 3.0:
        raise ValueError("Mean zero-shot accuracy drop exceeds 3 percentage points.")
    if float(tasks["max_drop_percentage_points"]) > 7.0:
        raise ValueError("A zero-shot task accuracy drop exceeds 7 percentage points.")
