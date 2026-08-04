r"""SpikingJelly package-level logger.

**API Language** - :ref:`中文 <spikingjelly-logger-cn>` | :ref:`English <spikingjelly-logger-en>`

----

.. _spikingjelly-logger-cn:

* **中文**

提供固定名称为 ``"spikingjelly"`` 的包级 :class:`logging.Logger`。框架代码应从
``spikingjelly.logger`` 导入共享变量 ``logger``。详细配置和用法见
``docs/source/APIs/spikingjelly.logger.rst``。

:var logger: 包级 Logger，名称固定为 ``"spikingjelly"``。
:vartype logger: logging.Logger

----

.. _spikingjelly-logger-en:

* **English**

This module exposes the package-level :class:`logging.Logger` with the fixed name
``"spikingjelly"``. Framework code should import the shared variable ``logger`` from
``spikingjelly.logger``.

The module installs one :class:`logging.NullHandler` and does not configure the root
logger, level, formatter, or output destination. See
``docs/source/APIs/spikingjelly.logger.rst`` for configuration examples.

:var logger: Package-level logger with the fixed name ``"spikingjelly"``.
:vartype logger: logging.Logger
"""

import logging


logger: logging.Logger = logging.getLogger("spikingjelly")
logger.addHandler(logging.NullHandler())

__all__ = ["logger"]
