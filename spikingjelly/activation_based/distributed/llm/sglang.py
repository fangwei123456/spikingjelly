from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import SGLangEngineConfig

_ARTIFACT_SCHEMA_VERSION = 2


def _validate_artifact(config: SGLangEngineConfig) -> None:
    if not config.artifact.is_dir():
        raise ValueError(f"SGLang artifact directory does not exist: {config.artifact}")
    if config.tokenizer is not None and not config.tokenizer.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {config.tokenizer}")
    try:
        model = json.loads(
            (config.artifact / "config.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (config.artifact / "spikingjelly_sglang.json").read_text(encoding="utf-8")
        )
        index = json.loads(
            (config.artifact / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("SGLang artifact metadata is missing or invalid.") from error
    if (
        not isinstance(model, dict)
        or not isinstance(manifest, dict)
        or not isinstance(index, dict)
    ):
        raise ValueError("Unsupported SpikingJelly SGLang artifact.")
    architectures = model.get("architectures")
    if (
        manifest.get("schema_version") != _ARTIFACT_SCHEMA_VERSION
        or not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise ValueError("Unsupported SpikingJelly SGLang artifact.")
    if manifest.get("dtype") != "bfloat16":
        raise ValueError("SGLang artifacts currently require bfloat16 weights.")
    if not isinstance(manifest.get("recipe_name"), str) or not manifest["recipe_name"]:
        raise ValueError("SGLang artifact must declare a non-empty recipe_name.")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("SGLang artifact tensor index is empty or invalid.")
    if any(
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not (config.artifact / filename).is_file()
        for filename in weight_map.values()
    ):
        raise ValueError("SGLang artifact tensor shard is missing or invalid.")


@contextlib.contextmanager
def open_sglang_engine(config: SGLangEngineConfig) -> Iterator[Any]:
    r"""Open a managed SGLang offline Engine for SpikingJelly artifacts.

    **API Language** - 中文 | English

    **中文：** 校验 SpikingJelly artifact，显式加载 ``external_model_package``，
    并返回原生 SGLang ``Engine``。退出上下文时总会调用 ``shutdown()`` 并恢复
    进程环境。由于 SGLang 使用 spawned workers，应只在受
    ``if __name__ == "__main__"`` 保护的应用入口调用本函数。

    **English:** Validate a SpikingJelly artifact, explicitly load
    ``external_model_package``, and yield the native SGLang ``Engine``. Leaving
    the context always calls ``shutdown()`` and restores the process environment.
    Because SGLang uses spawned workers, call this function only from an
    application entry point protected by ``if __name__ == "__main__"``.

    :param config: SGLang Engine configuration.
    :type config: SGLangEngineConfig
    :return: Managed native SGLang Engine.
    :rtype: contextlib.AbstractContextManager[sglang.Engine]
    :raises ImportError: ``spikingjelly[sglang]`` is unavailable.
    :raises ValueError: The artifact or topology is invalid.
    """
    if not isinstance(config, SGLangEngineConfig):
        raise TypeError("open_sglang_engine requires SGLangEngineConfig.")
    _validate_artifact(config)
    try:
        if importlib.util.find_spec(config.external_model_package) is None:
            raise ModuleNotFoundError(config.external_model_package)
    except (ImportError, AttributeError) as error:
        raise ImportError(
            f"SGLang external model package is unavailable: "
            f"{config.external_model_package}"
        ) from error
    previous_package = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = config.external_model_package
    engine = None
    try:
        try:
            import sglang
        except ImportError as error:
            raise ImportError(
                "SGLang inference requires a separate Python 3.12 environment "
                "with spikingjelly[sglang]."
            ) from error
        engine = sglang.Engine(
            model_path=str(config.artifact),
            tokenizer_path=(
                str(config.tokenizer) if config.tokenizer is not None else None
            ),
            skip_tokenizer_init=config.tokenizer is None,
            tp_size=config.tensor_parallel_size,
            pp_size=config.pipeline_parallel_size,
            dp_size=config.data_parallel_size,
            mem_fraction_static=config.memory_fraction,
            random_seed=config.seed,
            attention_backend="triton",
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            disable_decode_cuda_graph=True,
        )
        yield engine
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            if previous_package is None:
                os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            else:
                os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = previous_package


__all__ = ["open_sglang_engine"]
