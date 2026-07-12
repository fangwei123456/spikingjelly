from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from spikingjelly.activation_based.distributed.pipeline.partition import (
    _partition_costs_contiguously,
    parse_pipeline_layout,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    SNNPipelineRuntime,
    _PipelineSequentialModule,
    _build_snn_pipeline_runtime,
    _measure_module_cost,
)


class _CIFAR10DVSVGGPipelineStage(nn.Module):
    def __init__(
        self,
        feature_modules: Sequence[nn.Module],
        classifier: Optional[nn.Module] = None,
        transpose_input: bool = False,
    ):
        super().__init__()
        self.transpose_input = transpose_input
        self.features = nn.Sequential(*list(feature_modules))
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        if self.transpose_input:
            if x.ndim != 5:
                raise ValueError(
                    f"expected 5D input with shape [N, T, C, H, W], but got {tuple(x.shape)}"
                )
            x = x.transpose(0, 1).contiguous()
        x = self.features(x)
        if self.classifier is not None:
            x = torch.flatten(x, 2)
            x = self.classifier(x)
        return x


def _build_cifar10dvs_vgg_pipeline_module(
    module: nn.Module,
    num_logical_stages: int,
    example_input: torch.Tensor,
    layout_counts: Optional[Sequence[int]] = None,
) -> _PipelineSequentialModule:
    if num_logical_stages < 2:
        raise ValueError("CIFAR10DVSVGG pipeline parallel requires at least 2 stages.")
    if not (hasattr(module, "features") and hasattr(module, "classifier")):
        raise TypeError(
            "Expected a CIFAR10DVSVGG-like module with features and classifier."
        )

    feature_modules = list(module.features.children())
    current = example_input.transpose(0, 1).contiguous()
    feature_costs: list[float] = []
    for feature_module in feature_modules:
        current, cost = _measure_module_cost(feature_module, current)
        feature_costs.append(cost)
    classifier_input = torch.flatten(current, 2)
    _, classifier_cost = _measure_module_cost(module.classifier, classifier_input)
    unit_costs = [*feature_costs, classifier_cost]
    stage_unit_counts = (
        list(layout_counts)
        if layout_counts is not None
        else _partition_costs_contiguously(unit_costs, num_logical_stages)
    )
    first_active_stage_idx = next(
        (idx for idx, count in enumerate(stage_unit_counts) if count > 0), None
    )
    stages: list[nn.Module] = []
    cursor = 0
    total_feature_modules = len(feature_modules)
    classifier_assigned = False
    stage_costs: list[float] = []
    for stage_idx, count in enumerate(stage_unit_counts):
        feature_end = min(cursor + count, total_feature_modules)
        stage_features = feature_modules[cursor:feature_end]
        classifier = None
        if not classifier_assigned and cursor + count > total_feature_modules:
            classifier = module.classifier
            classifier_assigned = True
        start_unit = cursor
        end_unit = cursor + count
        cursor = feature_end
        stages.append(
            _CIFAR10DVSVGGPipelineStage(
                feature_modules=stage_features,
                classifier=classifier,
                transpose_input=stage_idx == first_active_stage_idx,
            )
        )
        stage_costs.append(sum(float(cost) for cost in unit_costs[start_unit:end_unit]))
    pipeline_module = _PipelineSequentialModule(stages)
    pipeline_module.stage_costs = tuple(stage_costs)
    return pipeline_module


def configure_cifar10dvs_vgg_pipeline(
    module: nn.Module,
    example_input: torch.Tensor,
    device: Union[str, torch.device],
    n_microbatches: int,
    pp_schedule: str = "auto",
    pp_virtual_stages: int = 1,
    pp_layout: Optional[Union[str, Sequence[int]]] = None,
    pp_delay_wgrad: bool = False,
    stage_index: Optional[int] = None,
    group=None,
) -> SNNPipelineRuntime:
    physical_num_stages = dist.get_world_size(group) if dist.is_initialized() else 1
    logical_num_stages = physical_num_stages * pp_virtual_stages
    feature_count = len(module.features)
    total_units = feature_count + 1
    layout_counts = parse_pipeline_layout(pp_layout, logical_num_stages, total_units)
    pipeline_module = _build_cifar10dvs_vgg_pipeline_module(
        module=module,
        num_logical_stages=logical_num_stages,
        example_input=example_input,
        layout_counts=layout_counts,
    )
    return _build_snn_pipeline_runtime(
        pipeline_module=pipeline_module,
        example_input=example_input,
        device=torch.device(device),
        n_microbatches=n_microbatches,
        stage_index=stage_index,
        model_family="cifar10dvs_vgg",
        schedule_kind=pp_schedule,
        virtual_pipeline_size=pp_virtual_stages,
        delayed_wgrad=pp_delay_wgrad,
        pp_layout=layout_counts,
        group=group,
    )
