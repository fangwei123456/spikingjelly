"""Phase 5.0 GPT-2 dense baseline and conversion contract.

This package only establishes the dense GPT-2 baseline and a structural
conversion contract. It does not perform FAS / QCFS / IF conversion, does not
implement KV cache, generation, Qwen / RoPE / GQA, multi-GPU, FP8, or a full
PPL benchmark. All of those are explicit ``explicitly_unsupported_now`` items.
"""

from .conversion_contract import (
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_MODEL_NAME,
    EXPECTED_REVISION,
    GPT2ModelPaths,
    REQUIRED_MODEL_FILES,
    REQUIRED_TOKENIZER_FILES,
    build_contract_report,
    scan_structure,
)
from .dense_baseline import (
    BASELINE_SCHEMA_VERSION,
    DEFAULT_MAX_SAMPLES,
    FIXED_PROMPTS,
    MAX_LENGTH,
    build_baseline_report,
    build_environment,
    compute_baseline,
    fixed_prompts,
    hash_files,
    validate_model_root,
)

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_MODEL_NAME",
    "EXPECTED_REVISION",
    "FIXED_PROMPTS",
    "GPT2ModelPaths",
    "MAX_LENGTH",
    "REQUIRED_MODEL_FILES",
    "REQUIRED_TOKENIZER_FILES",
    "build_baseline_report",
    "build_contract_report",
    "build_environment",
    "compute_baseline",
    "fixed_prompts",
    "hash_files",
    "scan_structure",
    "validate_model_root",
]
