from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseCounter, is_binary_tensor
from .config import SpikeSimEnergyConfig

__all__ = ["SpikeSimCounter"]
aten = torch.ops.aten


def _shape_tuple(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _pair_tuple(x: Any) -> tuple[int, int]:
    if isinstance(x, int):
        return (int(x), int(x))
    if len(x) != 2:
        raise ValueError(f"expected a pair, but got {x}.")
    return (int(x[0]), int(x[1]))


def _exclude_python_dispatch_guard():
    return torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
    )


@dataclass
class _StageStats:
    dense_pe_cycle_count: int = 0
    active_patch_tile_count: int = 0
    active_row_count: int = 0
    active_row_count_by_tile: list[int] | None = None
    active_output_tile_site_count: int = 0
    dense_patch_tile_count: int = 0
    dense_row_count: int = 0
    dense_row_count_by_tile: list[int] | None = None
    dense_output_tile_site_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _StageMetadata:
    scope: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    out_channel_tiles: int
    input_tile_channels: list[int]
    total_calls: int = 0
    event_driven_calls: int = 0
    dense_fallback_calls: int = 0
    shape_mismatch_detected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpikeSimCounter(BaseCounter):
    def __init__(
        self,
        *,
        config: SpikeSimEnergyConfig,
        strict: bool,
        verbose: bool,
    ):
        super().__init__()
        self.config = config
        self.strict = strict
        self.verbose = verbose
        self.stage_stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self.stage_metadata: dict[str, _StageMetadata] = {}
        self.warnings: list[str] = []
        self._warning_keys: set[str] = set()
        self._ones_kernel_cache: dict[
            tuple[int, int, int, torch.device], torch.Tensor
        ] = {}
        self.rules = {
            aten.convolution.default: self._count_convolution,
        }

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules=None,
        parent_names=None,
    ) -> int:
        return int(
            self.rules[func](
                args,
                kwargs,
                out,
                active_modules=active_modules,
                parent_names=parent_names,
            )
        )

    def has_rule(self, func) -> bool:
        return func in self.rules

    def _warn_or_raise(self, key: str, message: str) -> None:
        if key in self._warning_keys:
            return
        self._warning_keys.add(key)
        if self.strict:
            raise NotImplementedError(message)
        self.warnings.append(message)

    def _leaf_scope(self, parent_names: set[str] | None) -> str:
        names = [name for name in (parent_names or set()) if name != "Global"]
        if not names:
            return "Global"
        return max(names, key=lambda name: (name.count("."), len(name)))

    def _input_tile_channels(self, in_channels: int) -> list[int]:
        return [
            min(self.config.xbar_size, in_channels - start)
            for start in range(0, in_channels, self.config.xbar_size)
        ]

    def _dense_event_counts(
        self,
        *,
        w: torch.Tensor,
        out: torch.Tensor,
        out_channel_tiles: int,
    ) -> tuple[int, int, list[int], int]:
        num_sites = int(out.shape[0] * out.shape[2] * out.shape[3])
        input_tile_channels = self._input_tile_channels(int(w.shape[1]))
        dense_a = num_sites * len(input_tile_channels)
        dense_row_count_by_tile = [
            num_sites * tile_channels * w.shape[2] * w.shape[3]
            for tile_channels in input_tile_channels
        ]
        dense_r = sum(dense_row_count_by_tile)
        dense_z = out_channel_tiles * num_sites
        return dense_a, dense_r, dense_row_count_by_tile, dense_z

    def _dense_pe_cycles(
        self,
        *,
        w: torch.Tensor,
        out: torch.Tensor,
    ) -> int:
        p_i = math.ceil(int(w.shape[1]) / self.config.xbar_size)
        q_i = math.ceil(int(w.shape[0]) / self.config.xbar_size)
        num_sites = int(out.shape[0] * out.shape[2] * out.shape[3])
        return int(p_i * q_i * num_sites)

    def _spike_event_counts(
        self,
        *,
        x: torch.Tensor,
        w: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        out: torch.Tensor,
        out_channel_tiles: int,
    ) -> tuple[int, int, list[int], int]:
        xbar_size = self.config.xbar_size
        c_in = x.shape[1]
        num_tiles = math.ceil(c_in / xbar_size)
        k_h, k_w = w.shape[2], w.shape[3]
        padded_channels = num_tiles * xbar_size
        if padded_channels == c_in:
            x_padded = x
        else:
            x_padded = F.pad(x, (0, 0, 0, 0, 0, padded_channels - c_in))

        tile_sums = (
            x_padded.reshape(x.shape[0], num_tiles, xbar_size, x.shape[2], x.shape[3])
            .sum(dim=2)
            .to(dtype=torch.float32)
        )
        cache_key = (num_tiles, k_h, k_w, tile_sums.device)
        if cache_key not in self._ones_kernel_cache:
            self._ones_kernel_cache[cache_key] = tile_sums.new_ones(
                (num_tiles, 1, k_h, k_w)
            )
        ones_kernel = self._ones_kernel_cache[cache_key]
        with torch.no_grad():
            with _exclude_python_dispatch_guard():
                occupancy = F.conv2d(
                    tile_sums,
                    ones_kernel,
                    None,
                    stride,
                    padding,
                    dilation,
                    num_tiles,
                )

        active_patch = occupancy.gt(0)
        active_patch_tile_count = int(active_patch.sum().item())
        active_row_count_by_tile = [
            int(v) for v in occupancy.sum(dim=(0, 2, 3), dtype=torch.float64).tolist()
        ]
        active_row_count = int(sum(active_row_count_by_tile))
        active_site_mask = active_patch.any(dim=1)
        active_output_tile_site_count = out_channel_tiles * int(
            active_site_mask.sum().item()
        )
        return (
            active_patch_tile_count,
            active_row_count,
            active_row_count_by_tile,
            active_output_tile_site_count,
        )

    def _update_stage(
        self,
        *,
        scope: str,
        x: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        spike_like_input: bool,
    ) -> None:
        out_channel_tiles = math.ceil(w.shape[0] / self.config.xbar_size)
        metadata = self.stage_metadata.get(scope)
        if metadata is None:
            metadata = _StageMetadata(
                scope=scope,
                in_channels=int(w.shape[1]),
                out_channels=int(w.shape[0]),
                kernel_size=(int(w.shape[2]), int(w.shape[3])),
                stride=stride,
                padding=padding,
                dilation=dilation,
                input_shape=_shape_tuple(x),
                output_shape=_shape_tuple(out),
                out_channel_tiles=out_channel_tiles,
                input_tile_channels=self._input_tile_channels(int(w.shape[1])),
            )
            self.stage_metadata[scope] = metadata
        else:
            same_shape = (
                metadata.input_shape == _shape_tuple(x)
                and metadata.output_shape == _shape_tuple(out)
                and metadata.kernel_size == (int(w.shape[2]), int(w.shape[3]))
                and metadata.stride == stride
                and metadata.padding == padding
                and metadata.dilation == dilation
            )
            if not same_shape:
                metadata.shape_mismatch_detected = True
                self._warn_or_raise(
                    f"shape-mismatch:{scope}",
                    f"SpikeSim stage '{scope}' was invoked with inconsistent shapes; "
                    "runtime energy is accumulated across calls anyway.",
                )

        metadata.total_calls += 1
        dense_a, dense_r, dense_r_by_tile, dense_z = self._dense_event_counts(
            w=w,
            out=out,
            out_channel_tiles=out_channel_tiles,
        )
        if self.config.activity_mode == "event" and spike_like_input:
            metadata.event_driven_calls += 1
            active_a, active_r, active_r_by_tile, active_z = self._spike_event_counts(
                x=x,
                w=w,
                stride=stride,
                padding=padding,
                dilation=dilation,
                out=out,
                out_channel_tiles=out_channel_tiles,
            )
        else:
            metadata.dense_fallback_calls += 1
            active_a, active_r, active_r_by_tile, active_z = (
                dense_a,
                dense_r,
                dense_r_by_tile,
                dense_z,
            )

        stats = self.stage_stats[scope]
        stats.dense_pe_cycle_count += self._dense_pe_cycles(w=w, out=out)
        stats.active_patch_tile_count += active_a
        stats.active_row_count += active_r
        if stats.active_row_count_by_tile is None:
            stats.active_row_count_by_tile = [0] * len(active_r_by_tile)
        elif len(stats.active_row_count_by_tile) != len(active_r_by_tile):
            stats.active_row_count_by_tile = None
        for i, value in enumerate(active_r_by_tile):
            if stats.active_row_count_by_tile is None:
                break
            stats.active_row_count_by_tile[i] += value
        stats.active_output_tile_site_count += active_z
        stats.dense_patch_tile_count += dense_a
        stats.dense_row_count += dense_r
        if stats.dense_row_count_by_tile is None:
            stats.dense_row_count_by_tile = [0] * len(dense_r_by_tile)
        elif len(stats.dense_row_count_by_tile) != len(dense_r_by_tile):
            stats.dense_row_count_by_tile = None
        for i, value in enumerate(dense_r_by_tile):
            if stats.dense_row_count_by_tile is None:
                break
            stats.dense_row_count_by_tile[i] += value
        stats.dense_output_tile_site_count += dense_z

    def _handle_convolution(
        self,
        scope: str,
        args: tuple[Any, ...],
        out,
        active_modules: set[nn.Module] | None,
    ) -> int:
        x, w = args[0], args[1]
        stride, padding, dilation = args[3], args[4], args[5]
        transposed = bool(args[6])
        groups = int(args[8])

        if transposed:
            self._warn_or_raise(
                f"transposed-conv:{scope}",
                "SpikeSim energy only covers original SpikeSim Conv2d inference "
                f"stages; transposed convolutions are outside scope: {scope}.",
            )
            return 0
        if not self._is_forward_inference_conv(active_modules):
            self._warn_or_raise(
                f"outside-scope:{scope}",
                "SpikeSim energy only covers Conv2d forward inference stages from "
                f"nn.Conv2d modules: {scope}.",
            )
            return 0
        if groups != 1:
            self._warn_or_raise(
                f"grouped-conv:{scope}",
                "SpikeSim event energy does not support grouped/depthwise "
                f"convolutions: {scope}.",
            )
            return 0
        if x.dim() != 4 or w.dim() != 4 or out.dim() != 4:
            self._warn_or_raise(
                f"conv-rank:{scope}",
                "SpikeSim event energy only supports Conv2d-like calls, but got "
                f"x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)}, "
                f"out.shape={tuple(out.shape)}.",
            )
            return 0

        stride = _pair_tuple(stride)
        padding = _pair_tuple(padding)
        dilation = _pair_tuple(dilation)
        spike_like_input = is_binary_tensor(x)
        self._update_stage(
            scope=scope,
            x=x,
            w=w,
            out=out,
            stride=stride,
            padding=padding,
            dilation=dilation,
            spike_like_input=spike_like_input,
        )
        if self.verbose:
            mode = (
                "event"
                if self.config.activity_mode == "event" and spike_like_input
                else "dense"
            )
            print(
                f"SpikeSimCounter: {scope} - aten.convolution.default "
                f"[{mode}] x={tuple(x.shape)} w={tuple(w.shape)} out={tuple(out.shape)}"
            )
        return self._dense_pe_cycles(w=w, out=out)

    def _count_convolution(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        out,
        *,
        active_modules=None,
        parent_names=None,
    ) -> int:
        return self._handle_convolution(
            self._leaf_scope(parent_names), args, out, active_modules
        )

    def _is_forward_inference_conv(self, active_modules: set[nn.Module] | None) -> bool:
        modules = active_modules or set()
        conv2d_modules = [module for module in modules if isinstance(module, nn.Conv2d)]
        if len(conv2d_modules) != 1:
            return False
        conv = conv2d_modules[0]
        return (not conv.training) and (not torch.is_grad_enabled())

    def get_stage_stats(self) -> dict[str, dict[str, Any]]:
        return {stage: stats.as_dict() for stage, stats in self.stage_stats.items()}

    def get_stage_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            stage: metadata.as_dict() for stage, metadata in self.stage_metadata.items()
        }
