from __future__ import annotations

from collections import OrderedDict, defaultdict

import torch

from .config import MemoryHierarchyConfig
from .utils import _tensor_bits

class MemoryResidencySimulator:
    def __init__(self, config: MemoryHierarchyConfig):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.__init__-cn>` | :ref:`English <MemoryResidencySimulator.__init__-en>`

        ----

        .. _MemoryResidencySimulator.__init__-cn:

        * **中文**

        NeuroMC runtime 访存驻留模拟器。
        该模拟器维护 ``reg/sram/dram`` 三级层级，在每次读写事件发生时更新缓存驻留状态，并记录：

        - 各层级读/写/总 bit 数；
        - 分算子的层级读/写/总 bit 数；
        - 跨层搬移边（如 ``dram->sram``）的 bit 数。

        当前版本采用 LRU 淘汰策略，且层级顺序固定为 ``("reg", "sram", "dram")``。

        :param config: 驻留模拟配置
        :type config: MemoryHierarchyConfig

        ----

        .. _MemoryResidencySimulator.__init__-en:

        * **English**

        Runtime memory residency simulator for NeuroMC-style energy profiling.
        It tracks a 3-level hierarchy (``reg/sram/dram``), updates residency
        state on each read/write event, and records:

        - read/write/total bits per level;
        - read/write/total bits per op and level;
        - inter-level transfer bits by edges (e.g., ``dram->sram``).

        The current version uses LRU eviction and a fixed level order
        ``("reg", "sram", "dram")``.

        :param config: residency simulation config
        :type config: MemoryHierarchyConfig
        """
        if config.memory_model != "residency":
            raise ValueError(
                f"MemoryResidencySimulator requires memory_model='residency', got {config.memory_model}."
            )
        if config.eviction_policy != "LRU":
            raise ValueError("Only LRU eviction_policy is supported.")
        if config.level_order != ("reg", "sram", "dram"):
            raise ValueError("level_order must be ('reg', 'sram', 'dram').")

        capacity = config.capacity_bits or {}
        self.capacity_bits = {
            "reg": float(capacity.get("reg", float("inf"))),
            "sram": float(capacity.get("sram", float("inf"))),
            "dram": float(capacity.get("dram", float("inf"))),
        }
        self.reg_cache: OrderedDict[str, int] = OrderedDict()
        self.sram_cache: OrderedDict[str, int] = OrderedDict()
        self.usage_bits = {"reg": 0, "sram": 0}
        self.level_rw_bits: dict[str, dict[str, int]] = defaultdict(
            lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
        )
        self.op_level_rw_bits: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
            )
        )
        self.move_bits_by_edge: dict[str, int] = defaultdict(int)
        self.move_bits_by_op: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def _record_level_rw(self, level: str, rw: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._record_level_rw-cn>` | :ref:`English <MemoryResidencySimulator._record_level_rw-en>`

        ----

        .. _MemoryResidencySimulator._record_level_rw-cn:

        * **中文**

        在内部统计表中累计一次层级读/写 bit 事件。

        :param level: 访存层级名，如 ``reg/sram/dram``
        :type level: str

        :param rw: 读写字段名，通常为 ``read_bits`` 或 ``write_bits``
        :type rw: str

        :param bits: 本次事件 bit 数
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator._record_level_rw-en:

        * **English**

        Accumulate one read/write-bit event into internal per-level and
        per-op-per-level counters.

        :param level: memory level name, e.g. ``reg/sram/dram``
        :type level: str

        :param rw: read/write key, usually ``read_bits`` or ``write_bits``
        :type rw: str

        :param bits: event size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str
        """
        if bits <= 0:
            return
        self.level_rw_bits[level][rw] += bits
        self.level_rw_bits[level]["total_bits"] += bits
        self.op_level_rw_bits[op_name][level][rw] += bits
        self.op_level_rw_bits[op_name][level]["total_bits"] += bits

    def _record_move(self, src: str, dst: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._record_move-cn>` | :ref:`English <MemoryResidencySimulator._record_move-en>`

        ----

        .. _MemoryResidencySimulator._record_move-cn:

        * **中文**

        记录一次跨层搬移事件（如 ``sram->reg``），并同步折算为源层读与目标层写。

        :param src: 源层级
        :type src: str

        :param dst: 目标层级
        :type dst: str

        :param bits: 搬移 bit 数
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator._record_move-en:

        * **English**

        Record one inter-level movement event (e.g., ``sram->reg``), and
        reflect it as a read on source level and a write on destination level.

        :param src: source level
        :type src: str

        :param dst: destination level
        :type dst: str

        :param bits: transfer size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str
        """
        if bits <= 0:
            return
        edge = f"{src}->{dst}"
        self.move_bits_by_edge[edge] += bits
        self.move_bits_by_op[op_name][edge] += bits
        self._record_level_rw(src, "read_bits", bits, op_name)
        self._record_level_rw(dst, "write_bits", bits, op_name)

    def _touch(self, cache: OrderedDict[str, int], key: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._touch-cn>` | :ref:`English <MemoryResidencySimulator._touch-en>`

        ----

        .. _MemoryResidencySimulator._touch-cn:

        * **中文**

        将 LRU 容器中 ``key`` 标记为最新使用（移动到末尾）。

        :param cache: 目标 LRU 容器
        :type cache: OrderedDict[str, int]

        :param key: 张量键
        :type key: str

        ----

        .. _MemoryResidencySimulator._touch-en:

        * **English**

        Mark ``key`` as most recently used in a LRU container.

        :param cache: target LRU container
        :type cache: OrderedDict[str, int]

        :param key: tensor key
        :type key: str
        """
        cache.move_to_end(key, last=True)

    def _touch_inclusive_levels(self, key: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._touch_inclusive_levels-cn>` | :ref:`English <MemoryResidencySimulator._touch_inclusive_levels-en>`

        ----

        .. _MemoryResidencySimulator._touch_inclusive_levels-cn:

        * **中文**

        对包含该 ``key`` 的层级做联合触摸：
        若 ``reg`` 命中，同时刷新 ``sram`` 的 LRU 顺序，避免热点数据在下层被过早淘汰。

        :param key: 张量键
        :type key: str

        ----

        .. _MemoryResidencySimulator._touch_inclusive_levels-en:

        * **English**

        Touch all inclusive levels containing ``key``.
        For a ``reg`` hit, this also refreshes the ``sram`` LRU order to avoid
        premature lower-level eviction of hot tensors.

        :param key: tensor key
        :type key: str
        """
        if key in self.reg_cache:
            self._touch(self.reg_cache, key)
        if key in self.sram_cache:
            self._touch(self.sram_cache, key)

    def _insert_sram(self, key: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._insert_sram-cn>` | :ref:`English <MemoryResidencySimulator._insert_sram-en>`

        ----

        .. _MemoryResidencySimulator._insert_sram-cn:

        * **中文**

        尝试将张量插入 SRAM 层，必要时按 LRU 淘汰旧条目，并记录 ``sram->dram`` 搬移。

        :param key: 张量键
        :type key: str

        :param bits: 张量大小（bit）
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        :return: 是否成功插入 SRAM
        :rtype: bool

        ----

        .. _MemoryResidencySimulator._insert_sram-en:

        * **English**

        Try to insert a tensor into SRAM. If needed, evict old entries using
        LRU and record ``sram->dram`` movement.

        :param key: tensor key
        :type key: str

        :param bits: tensor size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str

        :return: whether insertion into SRAM succeeds
        :rtype: bool
        """
        if bits > self.capacity_bits["sram"]:
            return False
        if key in self.sram_cache:
            old = self.sram_cache[key]
            if bits > old:
                self.usage_bits["sram"] += bits - old
                self.sram_cache[key] = bits
            self._touch(self.sram_cache, key)
            return True

        while self.sram_cache and (
            self.usage_bits["sram"] + bits > self.capacity_bits["sram"]
        ):
            evict_key, evict_bits = self.sram_cache.popitem(last=False)
            self.usage_bits["sram"] -= evict_bits
            self._record_move("sram", "dram", evict_bits, op_name)
            reg_bits = self.reg_cache.pop(evict_key, None)
            if reg_bits is not None:
                self.usage_bits["reg"] -= reg_bits

        if self.usage_bits["sram"] + bits > self.capacity_bits["sram"]:
            return False

        self.sram_cache[key] = bits
        self.usage_bits["sram"] += bits
        return True

    def _insert_reg(self, key: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._insert_reg-cn>` | :ref:`English <MemoryResidencySimulator._insert_reg-en>`

        ----

        .. _MemoryResidencySimulator._insert_reg-cn:

        * **中文**

        尝试将张量插入寄存器层。容量不足时按 LRU 淘汰并下推到 SRAM，
        记录 ``reg->sram`` 搬移。

        :param key: 张量键
        :type key: str

        :param bits: 张量大小（bit）
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        :return: 是否成功插入 reg
        :rtype: bool

        ----

        .. _MemoryResidencySimulator._insert_reg-en:

        * **English**

        Try to insert a tensor into register level. On capacity pressure, evict
        by LRU, push evicted entries to SRAM, and record ``reg->sram`` movement.

        :param key: tensor key
        :type key: str

        :param bits: tensor size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str

        :return: whether insertion into reg succeeds
        :rtype: bool
        """
        if bits > self.capacity_bits["reg"]:
            return False
        if key in self.reg_cache:
            old = self.reg_cache[key]
            if bits > old:
                self.usage_bits["reg"] += bits - old
                self.reg_cache[key] = bits
            self._touch(self.reg_cache, key)
            return True

        while self.reg_cache and (
            self.usage_bits["reg"] + bits > self.capacity_bits["reg"]
        ):
            evict_key, evict_bits = self.reg_cache.popitem(last=False)
            self.usage_bits["reg"] -= evict_bits
            self._record_move("reg", "sram", evict_bits, op_name)
            self._insert_sram(evict_key, evict_bits, op_name)

        if self.usage_bits["reg"] + bits > self.capacity_bits["reg"]:
            return False

        self.reg_cache[key] = bits
        self.usage_bits["reg"] += bits
        return True

    def _tensor_key(self, tensor: torch.Tensor) -> str:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._tensor_key-cn>` | :ref:`English <MemoryResidencySimulator._tensor_key-en>`

        ----

        .. _MemoryResidencySimulator._tensor_key-cn:

        * **中文**

        为张量构造驻留模拟使用的键，包含设备、dtype、data_ptr、
        storage_offset 和 shape。

        :param tensor: 输入张量
        :type tensor: torch.Tensor

        :return: 张量键
        :rtype: str

        ----

        .. _MemoryResidencySimulator._tensor_key-en:

        * **English**

        Build a simulator key for tensor identity using device, dtype, data_ptr,
        storage_offset, and shape.

        :param tensor: input tensor
        :type tensor: torch.Tensor

        :return: tensor key
        :rtype: str
        """
        data_ptr = int(tensor.data_ptr())
        return (
            f"{tensor.device}:{tensor.dtype}:{data_ptr}:"
            f"{int(tensor.storage_offset())}:{tuple(tensor.shape)}"
        )

    def on_tensor_read(self, tensor: torch.Tensor, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.on_tensor_read-cn>` | :ref:`English <MemoryResidencySimulator.on_tensor_read-en>`

        ----

        .. _MemoryResidencySimulator.on_tensor_read-cn:

        * **中文**

        处理一次张量读事件：

        - 若命中 reg：记 reg 读；
        - 若命中 sram：记 sram 读，并尝试提升到 reg；
        - 若 miss：记 dram 读，并尝试 ``dram->sram->reg`` 或 ``dram->reg`` 分配路径。

        :param tensor: 被读取张量
        :type tensor: torch.Tensor

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator.on_tensor_read-en:

        * **English**

        Handle one tensor read event:

        - reg hit: record reg read;
        - sram hit: record sram read and try to promote to reg;
        - miss: record dram read, then try allocation path
          ``dram->sram->reg`` or ``dram->reg``.

        :param tensor: tensor being read
        :type tensor: torch.Tensor

        :param op_name: operator name
        :type op_name: str
        """
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        self._record_level_rw("dram", "read_bits", bits, op_name)
        sram_ok = self._insert_sram(key, bits, op_name)
        if sram_ok:
            self._record_move("dram", "sram", bits, op_name)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_move("dram", "reg", bits, op_name)
            self._record_level_rw("reg", "read_bits", bits, op_name)

    def on_tensor_write(self, tensor: torch.Tensor, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.on_tensor_write-cn>` | :ref:`English <MemoryResidencySimulator.on_tensor_write-en>`

        ----

        .. _MemoryResidencySimulator.on_tensor_write-cn:

        * **中文**

        处理一次张量写事件：

        - 若命中 reg：记 reg 写；
        - 若命中 sram：记 sram 写，并尝试提升到 reg；
        - 若 miss：优先分配到 reg，再尝试 sram，否则落到 dram 写。

        :param tensor: 被写入张量
        :type tensor: torch.Tensor

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator.on_tensor_write-en:

        * **English**

        Handle one tensor write event:

        - reg hit: record reg write;
        - sram hit: record sram write and try to promote to reg;
        - miss: allocate to reg first, then sram, otherwise record dram write.

        :param tensor: tensor being written
        :type tensor: torch.Tensor

        :param op_name: operator name
        :type op_name: str
        """
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "write_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_sram(key, bits, op_name):
            self._record_level_rw("sram", "write_bits", bits, op_name)
            return

        self._record_level_rw("dram", "write_bits", bits, op_name)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_level_rw_bits-cn>` | :ref:`English <MemoryResidencySimulator.get_level_rw_bits-en>`

        ----

        .. _MemoryResidencySimulator.get_level_rw_bits-cn:

        * **中文**

        获取各层级读/写/总 bit 统计快照。

        :return: ``dict[level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, int]]

        ----

        .. _MemoryResidencySimulator.get_level_rw_bits-en:

        * **English**

        Get a snapshot of per-level read/write/total bit statistics.

        :return: ``dict[level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, int]]
        """
        return {k: dict(v) for k, v in self.level_rw_bits.items()}

    def get_op_level_rw_bits(self) -> dict[str, dict[str, dict[str, int]]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_op_level_rw_bits-cn>` | :ref:`English <MemoryResidencySimulator.get_op_level_rw_bits-en>`

        ----

        .. _MemoryResidencySimulator.get_op_level_rw_bits-cn:

        * **中文**

        获取分算子、分层级的读/写/总 bit 统计快照。

        :return: ``dict[op_name][level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, dict[str, int]]]

        ----

        .. _MemoryResidencySimulator.get_op_level_rw_bits-en:

        * **English**

        Get a snapshot of read/write/total bit statistics grouped by op and level.

        :return: ``dict[op_name][level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, dict[str, int]]]
        """
        return {
            op: {lv: dict(info) for lv, info in d.items()}
            for op, d in self.op_level_rw_bits.items()
        }

    def get_move_bits_by_edge(self) -> dict[str, int]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_move_bits_by_edge-cn>` | :ref:`English <MemoryResidencySimulator.get_move_bits_by_edge-en>`

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_edge-cn:

        * **中文**

        获取跨层搬移边统计（例如 ``dram->sram``）。

        :return: ``dict[edge] = moved_bits``
        :rtype: dict[str, int]

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_edge-en:

        * **English**

        Get inter-level transfer statistics by edge (e.g., ``dram->sram``).

        :return: ``dict[edge] = moved_bits``
        :rtype: dict[str, int]
        """
        return dict(self.move_bits_by_edge)

    def get_move_bits_by_op(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_move_bits_by_op-cn>` | :ref:`English <MemoryResidencySimulator.get_move_bits_by_op-en>`

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_op-cn:

        * **中文**

        获取分算子的跨层搬移统计。

        :return: ``dict[op_name][edge] = moved_bits``
        :rtype: dict[str, dict[str, int]]

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_op-en:

        * **English**

        Get inter-level transfer statistics grouped by op name.

        :return: ``dict[op_name][edge] = moved_bits``
        :rtype: dict[str, dict[str, int]]
        """
        return {op: dict(d) for op, d in self.move_bits_by_op.items()}


