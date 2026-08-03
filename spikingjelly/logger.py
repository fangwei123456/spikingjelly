r"""SpikingJelly package-level logger.

**API Language** - :ref:`中文 <spikingjelly-logger-cn>` | :ref:`English <spikingjelly-logger-en>`

----

.. _spikingjelly-logger-cn:

* **中文**

提供 SpikingJelly 统一使用的包级 :class:`logging.Logger`。框架代码应从
``spikingjelly.logger`` 导入 ``logger``，不要在每次记录日志时重复调用
``logging.getLogger(...)``。

该模块只注册一个 :class:`logging.NullHandler`，不会配置 root logger、日志级别、
格式或输出位置。应用程序负责决定是否显示日志，以及是否添加控制台、文件或其他
Handler。Logger 默认允许向 root logger 传播，因此应用可以在入口统一配置
``logging.basicConfig`` 或 ``logging.config.dictConfig``。

日志应集中在初始化、backend 选择、fallback、转换完成和缓存生命周期等低频边界；
不要把默认日志调用放进神经元 ``forward``、time-step 循环、batch 循环或算子执行
热路径。

:var logger: 包级 Logger，名称固定为 ``"spikingjelly"``。
:vartype logger: logging.Logger

----

.. _spikingjelly-logger-en:

* **English**

This module exposes the package-level :class:`logging.Logger` used by SpikingJelly.
Framework code should import ``logger`` from ``spikingjelly.logger`` instead of
calling ``logging.getLogger(...)`` at every logging site.

The module installs only one :class:`logging.NullHandler`; it does not configure the
root logger, default level, formatter, or output destination. The application owns
those decisions and may add console, file, or other handlers. Propagation remains
enabled by default so an application can configure logging once at its entry point
with ``logging.basicConfig`` or ``logging.config.dictConfig``.

Default logging belongs at low-frequency lifecycle boundaries such as initialization,
backend selection, fallback, conversion completion, and cache events. Do not add
default logging calls to neuron ``forward`` methods, time-step loops, batch loops, or
operator execution hot paths.

:var logger: Package-level logger with the fixed name ``"spikingjelly"``.
:vartype logger: logging.Logger
"""

import logging


logger: logging.Logger = logging.getLogger("spikingjelly")
logger.addHandler(logging.NullHandler())

__all__ = ["logger"]
