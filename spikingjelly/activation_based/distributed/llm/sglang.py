from __future__ import annotations

import json
import os
from typing import Any

import torch

from .config import SGLangGenerationConfig


def create_sglang_engine(config: SGLangGenerationConfig) -> Any:
    r"""Create an experimental SGLang offline Engine for repeated inference calls.

    **API Language** - 中文 | English

    **中文：** 此接口为实验性接口，其行为在稳定前可能调整。校验 artifact 和并行
    拓扑后创建 offline Engine。调用方拥有 Engine 生命周期，完成后必须调用
    ``shutdown()``。

    **English:** This API is experimental and may change before stabilization.
    It validates the artifact and topology, then creates an offline Engine. The
    caller owns its lifecycle and must call ``shutdown()``.

    :param config: SGLang generation configuration.
    :type config: SGLangGenerationConfig
    :return: SGLang offline Engine.
    :rtype: sglang.Engine
    :raises ImportError: ``spikingjelly[sglang]`` is unavailable.
    :raises ValueError: The artifact or DCP topology is invalid.
    """

    if not config.artifact.is_dir():
        raise ValueError(f"SGLang artifact directory does not exist: {config.artifact}")
    artifact_config = json.loads(
        (config.artifact / "config.json").read_text(encoding="utf-8")
    )
    if (
        artifact_config.get("spikingjelly_artifact_schema") is not None
        and config.decode_context_parallel_size > 1
    ):
        kv_heads = int(artifact_config["num_key_value_heads"])
        if kv_heads <= 0 or config.tensor_parallel_size % (
            kv_heads * config.decode_context_parallel_size
        ):
            raise ValueError(
                "DCP requires TP-replicated KV heads; increase TP or reduce "
                "decode_context_parallel_size."
            )
    previous_package = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    if config.external_model_package is None:
        os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    else:
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = config.external_model_package
    try:
        try:
            import sglang
        except ImportError as error:
            raise ImportError(
                "SGLang inference requires a separate environment with "
                "spikingjelly[sglang]."
            ) from error
        return sglang.Engine(
            model_path=str(config.artifact),
            tokenizer_path=(
                str(config.tokenizer) if config.tokenizer is not None else None
            ),
            skip_tokenizer_init=config.tokenizer is None,
            tp_size=config.tensor_parallel_size,
            pp_size=config.pipeline_parallel_size,
            dp_size=config.data_parallel_size,
            attn_cp_size=config.prefill_context_parallel_size,
            enable_prefill_cp=config.prefill_context_parallel_size > 1,
            dcp_size=config.decode_context_parallel_size,
            mem_fraction_static=config.memory_fraction,
            random_seed=config.seed,
            attention_backend="triton",
            disable_cuda_graph=True,
        )
    finally:
        if previous_package is None:
            os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
        else:
            os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = previous_package


def generate_sglang(
    config: SGLangGenerationConfig, input_ids: torch.Tensor
) -> list[dict[str, Any]]:
    r"""Generate token IDs with the experimental SGLang offline integration.

    **API Language** - 中文 | English

    **中文：** 此接口为实验性接口，其行为在稳定前可能调整。它在独立 SGLang 环境中
    加载 artifact，直接向 offline ``Engine`` 提交 tokenized prompts。本函数不启动
    HTTP、router 或其他 serving 控制面。

    **English:** This API is experimental and may change before stabilization.
    It loads the artifact in a separate SGLang environment and submits tokenized
    prompts directly to the offline ``Engine``. This function starts no HTTP
    server, router, or other serving control plane.

    :param config: SGLang generation configuration.
    :type config: SGLangGenerationConfig
    :param input_ids: Non-empty integer prompts shaped ``[B, S]``.
    :type input_ids: torch.Tensor
    :return: SGLang results in input order; each item contains ``output_ids``,
        ``text``, and ``meta_info``.
    :rtype: list[dict[str, Any]]
    :raises ImportError: ``spikingjelly[sglang]`` is unavailable.
    :raises ValueError: The artifact or prompt tensor is invalid.
    """
    if (
        input_ids.ndim != 2
        or input_ids.numel() == 0
        or input_ids.dtype
        not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("input_ids must be a non-empty integer [B, S] tensor.")
    engine = create_sglang_engine(config)
    try:
        return engine.generate(
            input_ids=input_ids.long().tolist(),
            sampling_params={
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            },
        )
    finally:
        engine.shutdown()


__all__ = ["create_sglang_engine", "generate_sglang"]
