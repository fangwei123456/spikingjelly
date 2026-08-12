spikingjelly.logger module
==========================

.. automodule:: spikingjelly.logger
   :members:
   :undoc-members:
   :show-inheritance:

使用方法
--------

**中文**

SpikingJelly 直接导出 Loguru 的全局 logger，但默认禁用 ``"spikingjelly"`` 命名空间。
导入包不会输出日志，也不会添加、删除或配置 sink。应用应在导入功能子模块前配置所需
级别并启用该命名空间：

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

Loguru 默认 stderr sink 的级别是 ``DEBUG``，因此不建议直接调用 ``logger.enable()``。
SpikingJelly 不缓存日志；启用前由模块导入产生的诊断无法事后恢复。

``logger.remove()`` 会影响进程中的所有 Loguru sink，因此只能由应用调用。框架内部只使用
``logger.debug/info/warning/error/exception/critical`` 和 ``{}`` 参数化格式，不配置
sink，也不附加结构化上下文。使用标准库 ``logging`` 的应用可以在自己的入口按
`Loguru 标准 logging 兼容说明 <https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging>`_
完成桥接；SpikingJelly 不内置桥接层。
测试若要捕获日志，应使用 ``logger.add(sink)`` 添加临时 sink，并在测试后移除。

Common usage
------------

**English**

SpikingJelly exports Loguru's global logger directly, but the ``"spikingjelly"``
namespace is disabled by default. Importing the package emits nothing and does not
add, remove, or configure sinks. Applications should configure the desired level
and enable the namespace before importing feature modules:

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

Loguru's default stderr sink accepts ``DEBUG`` records, so calling
``logger.enable()`` alone is not recommended. SpikingJelly does not buffer records;
import-time diagnostics emitted before enabling the namespace cannot be recovered.

``logger.remove()`` affects every Loguru sink in the process and therefore belongs
only in application code. Framework code only calls
``logger.debug/info/warning/error/exception/critical`` with ``{}``-style arguments;
it does not configure sinks or attach structured context. Applications built around
stdlib ``logging`` can add interoperability at their own entry point by following
Loguru's `standard logging compatibility guide <https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging>`_;
SpikingJelly does not provide a bridge.
Tests that need to capture records should add a temporary sink with
``logger.add(sink)`` and remove it afterwards.
