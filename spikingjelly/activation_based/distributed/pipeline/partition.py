from typing import List, Optional, Sequence, Tuple, Union


def _even_partition_sizes(num_items: int, num_parts: int) -> List[int]:
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, but got {num_parts}.")
    base = num_items // num_parts
    rem = num_items % num_parts
    return [base + (1 if idx < rem else 0) for idx in range(num_parts)]


def _partition_costs_contiguously(costs: Sequence[float], num_parts: int) -> List[int]:
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, but got {num_parts}.")
    num_items = len(costs)
    if num_items == 0:
        return [0 for _ in range(num_parts)]
    if num_items < num_parts:
        return _even_partition_sizes(num_items, num_parts)

    lo = max(float(cost) for cost in costs)
    hi = sum(float(cost) for cost in costs)

    def _fits(limit: float) -> bool:
        parts = 1
        acc = 0.0
        for cost in costs:
            cost = float(cost)
            if acc == 0.0 or acc + cost <= limit:
                acc += cost
            else:
                parts += 1
                acc = cost
        return parts <= num_parts

    for _ in range(48):
        mid = (lo + hi) * 0.5
        if _fits(mid):
            hi = mid
        else:
            lo = mid
    limit = hi

    sizes_reversed: List[int] = []
    acc = 0.0
    count = 0
    parts_remaining = num_parts
    for idx in range(num_items - 1, -1, -1):
        cost = float(costs[idx])
        remaining_items = idx + 1
        if count > 0 and (acc + cost > limit or remaining_items < parts_remaining):
            sizes_reversed.append(count)
            parts_remaining -= 1
            acc = 0.0
            count = 0
        acc += cost
        count += 1
    sizes_reversed.append(count)
    sizes = list(reversed(sizes_reversed))
    if len(sizes) < num_parts:
        sizes = [0] * (num_parts - len(sizes)) + sizes
    return sizes


def parse_pipeline_layout(
    layout: Optional[Union[str, Sequence[int]]],
    num_logical_stages: int,
    total_units: int,
) -> Optional[Tuple[int, ...]]:
    r"""
    **API Language** - :ref:`中文 <parse_pipeline_layout-cn>` | :ref:`English <parse_pipeline_layout-en>`

    ----

    .. _parse_pipeline_layout-cn:

    * **中文**

    解析流水线并行布局配置。

    ----

    .. _parse_pipeline_layout-en:

    * **English**

    Parse pipeline parallel layout configuration.
    """
    if layout is None:
        return None
    if isinstance(layout, str):
        raw_tokens = layout.replace(",", "|").split("|")
        counts = tuple(int(token.strip()) for token in raw_tokens if token.strip())
    else:
        counts = tuple(int(item) for item in layout)
    if len(counts) != num_logical_stages:
        raise ValueError(
            f"Pipeline layout must provide {num_logical_stages} stage counts, "
            f"but got {len(counts)} from {layout!r}."
        )
    if any(count < 0 for count in counts):
        raise ValueError(
            f"Pipeline layout counts must be non-negative, but got {counts}."
        )
    if sum(counts) != total_units:
        raise ValueError(
            f"Pipeline layout {counts} covers {sum(counts)} units, but the model requires "
            f"{total_units} units."
        )
    return counts


def resolve_pipeline_schedule_kind(
    schedule_kind: str,
    virtual_pipeline_size: int,
    delayed_wgrad: bool,
) -> str:
    r"""
    **API Language** - :ref:`中文 <resolve_pipeline_schedule_kind-cn>` | :ref:`English <resolve_pipeline_schedule_kind-en>`

    ----

    .. _resolve_pipeline_schedule_kind-cn:

    * **中文**

    解析流水线调度类型。

    ----

    .. _resolve_pipeline_schedule_kind-en:

    * **English**

    Resolve pipeline schedule kind.
    """
    normalized = schedule_kind.lower()
    if normalized not in ("auto", "gpipe", "1f1b", "interleaved", "zero_bubble"):
        raise ValueError(
            f"Unsupported pp schedule '{schedule_kind}'. "
            "Expected one of: auto, gpipe, 1f1b, interleaved, zero_bubble."
        )
    if normalized == "auto":
        if delayed_wgrad:
            normalized = "zero_bubble"
        elif virtual_pipeline_size > 1:
            normalized = "interleaved"
        else:
            normalized = "1f1b"
    if normalized == "gpipe" and delayed_wgrad:
        raise ValueError("pp_delay_wgrad is incompatible with pp_schedule='gpipe'.")
    if normalized in ("gpipe", "1f1b") and virtual_pipeline_size != 1:
        raise ValueError(
            f"pp_schedule='{normalized}' does not support pp_virtual_stages={virtual_pipeline_size}. "
            "Use pp_schedule='interleaved' or 'zero_bubble' when pp_virtual_stages > 1."
        )
    if normalized in ("interleaved", "zero_bubble") and virtual_pipeline_size < 2:
        raise ValueError(
            f"pp_schedule='{normalized}' requires pp_virtual_stages >= 2, "
            f"but got {virtual_pipeline_size}."
        )
    return normalized
