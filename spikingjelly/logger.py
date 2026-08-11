r"""SpikingJelly package-level logger.

**API Language** - :ref:`中文 <spikingjelly-logger-cn>` | :ref:`English <spikingjelly-logger-en>`

----

.. _spikingjelly-logger-cn:

* **中文**

提供默认禁用的包级 Loguru logger。框架代码应从 ``spikingjelly.logger`` 导入共享
变量 ``logger``。详细配置和用法见
``docs/source/APIs/spikingjelly.logger.rst``。

:var logger: Loguru 全局 logger。
:vartype logger: loguru.Logger

----

.. _spikingjelly-logger-en:

* **English**

This module exposes Loguru's global logger, disabled for the ``"spikingjelly"``
namespace by default. Framework code should import the shared variable ``logger``
from ``spikingjelly.logger``.

The module does not add or remove sinks. See ``docs/source/APIs/spikingjelly.logger.rst``
for configuration examples.

:var logger: Loguru's global logger.
:vartype logger: loguru.Logger
"""

from loguru import logger


logger.disable("spikingjelly")

__all__ = ["logger"]
