from __future__ import annotations

from typing import Any

import torch

from .base import is_binary_tensor

__all__ = [
    "active_element_count",
    "dense_bytes",
    "dense_bytes_tree",
    "is_sparse_access_tensor",
    "sparse_bytes",
    "sparse_bytes_tree",
]


def active_element_count(x: torch.Tensor) -> int:
    r"""
    **API Language** - :ref:`中文 <active_element_count-cn>` | :ref:`English <active_element_count-en>`

    ----

    .. _active_element_count-cn:

    * **中文**

    :param x: 输入张量
    :type x: torch.Tensor

    :return: ``x`` 中非零元素的个数
    :rtype: int

    ----

    .. _active_element_count-en:

    * **English**

    :param x: input tensor
    :type x: torch.Tensor

    :return: number of non-zero elements in ``x``
    :rtype: int
    """
    return int(x.count_nonzero().item())


def dense_bytes(x: torch.Tensor) -> int:
    r"""
    **API Language** - :ref:`中文 <dense_bytes-cn>` | :ref:`English <dense_bytes-en>`

    ----

    .. _dense_bytes-cn:

    * **中文**

    :param x: 输入张量
    :type x: torch.Tensor

    :return: ``x`` 以密集格式存储所需的字节数，即 ``numel * element_size``
    :rtype: int

    ----

    .. _dense_bytes-en:

    * **English**

    :param x: input tensor
    :type x: torch.Tensor

    :return: number of bytes for dense storage of ``x``, i.e. ``numel * element_size``
    :rtype: int
    """
    return int(x.numel()) * int(x.element_size())


def sparse_bytes(x: torch.Tensor) -> int:
    r"""
    **API Language** - :ref:`中文 <sparse_bytes-cn>` | :ref:`English <sparse_bytes-en>`

    ----

    .. _sparse_bytes-cn:

    * **中文**

    :param x: 输入张量
    :type x: torch.Tensor

    :return: ``x`` 以稀疏格式存储所需的字节数，即 ``active_elements * element_size``
    :rtype: int

    ----

    .. _sparse_bytes-en:

    * **English**

    :param x: input tensor
    :type x: torch.Tensor

    :return: number of bytes for sparse storage of ``x``, i.e. ``active_elements * element_size``
    :rtype: int
    """
    return active_element_count(x) * int(x.element_size())


def is_sparse_access_tensor(
    x: torch.Tensor, *, zero_ratio_threshold: float = 0.5
) -> bool:
    r"""
    **API Language** - :ref:`中文 <is_sparse_access_tensor-cn>` | :ref:`English <is_sparse_access_tensor-en>`

    ----

    .. _is_sparse_access_tensor-cn:

    * **中文**

    :param x: 输入张量
    :type x: torch.Tensor

    :param zero_ratio_threshold: 稀疏判断的零值比例阈值。若零值的比例超过此阈值，则视为稀疏访问张量。
        默认为 ``0.5``
    :type zero_ratio_threshold: float

    :return: 当 ``x`` 为零值比例高于 ``zero_ratio_threshold`` 的二元脉冲张量或稠密张量时返回 ``True``
    :rtype: bool

    ----

    .. _is_sparse_access_tensor-en:

    * **English**

    :param x: input tensor
    :type x: torch.Tensor

    :param zero_ratio_threshold: zero-ratio threshold for sparsity detection.
        A tensor is considered sparse if the ratio of zero elements exceeds this threshold.
        Default to ``0.5``.
    :type zero_ratio_threshold: float

    :return: ``True`` if ``x`` is a binary spike tensor or a dense tensor with
        zero-ratio exceeding ``zero_ratio_threshold``
    :rtype: bool
    """
    if x.numel() == 0:
        return False
    if x.is_meta:
        return False
    if x.dtype == torch.bool or is_binary_tensor(x):
        return True
    zero_ratio = 1.0 - float(x.count_nonzero().item()) / float(x.numel())
    return zero_ratio >= zero_ratio_threshold


def dense_bytes_tree(tree: Any) -> int:
    r"""
    **API Language** - :ref:`中文 <dense_bytes_tree-cn>` | :ref:`English <dense_bytes_tree-en>`

    ----

    .. _dense_bytes_tree-cn:

    * **中文**

    :param tree: 可能包含张量的嵌套容器（tuple / list / dict）
    :type tree: Any

    :return: 容器中所有张量的密集字节数之和
    :rtype: int

    ----

    .. _dense_bytes_tree-en:

    * **English**

    :param tree: a nested container (tuple / list / dict) that may contain tensors
    :type tree: Any

    :return: total dense bytes of all tensors in the container
    :rtype: int
    """
    if torch.is_tensor(tree):
        return dense_bytes(tree)
    if isinstance(tree, (tuple, list)):
        return sum(dense_bytes_tree(item) for item in tree)
    if isinstance(tree, dict):
        return sum(dense_bytes_tree(item) for item in tree.values())
    return 0


def sparse_bytes_tree(tree: Any, *, zero_ratio_threshold: float = 0.5) -> int:
    r"""
    **API Language** - :ref:`中文 <sparse_bytes_tree-cn>` | :ref:`English <sparse_bytes_tree-en>`

    ----

    .. _sparse_bytes_tree-cn:

    * **中文**

    :param tree: 可能包含张量的嵌套容器（tuple / list / dict）
    :type tree: Any

    :param zero_ratio_threshold: 稀疏判断的零值比例阈值。参见 :func:`is_sparse_access_tensor`。
        默认为 ``0.5``
    :type zero_ratio_threshold: float

    :return: 容器中所有张量的稀疏字节数之和。对于稀疏张量，仅计算非零元素对应的字节数
    :rtype: int

    ----

    .. _sparse_bytes_tree-en:

    * **English**

    :param tree: a nested container (tuple / list / dict) that may contain tensors
    :type tree: Any

    :param zero_ratio_threshold: zero-ratio threshold for sparsity detection.
        See :func:`is_sparse_access_tensor`. Default to ``0.5``.
    :type zero_ratio_threshold: float

    :return: total sparse bytes of all tensors in the container. For sparse
        tensors, only non-zero elements are counted
    :rtype: int
    """
    if torch.is_tensor(tree):
        if is_sparse_access_tensor(tree, zero_ratio_threshold=zero_ratio_threshold):
            return sparse_bytes(tree)
        return dense_bytes(tree)
    if isinstance(tree, (tuple, list)):
        return sum(
            sparse_bytes_tree(item, zero_ratio_threshold=zero_ratio_threshold)
            for item in tree
        )
    if isinstance(tree, dict):
        return sum(
            sparse_bytes_tree(item, zero_ratio_threshold=zero_ratio_threshold)
            for item in tree.values()
        )
    return 0
