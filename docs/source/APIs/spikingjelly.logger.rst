spikingjelly.logger module
==========================

.. automodule:: spikingjelly.logger
   :members:
   :undoc-members:
   :show-inheritance:

使用方法
--------

**中文**

SpikingJelly 直接导出 Loguru 的全局 logger：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("Model prepared: backend={} device={}", "torch", "cpu")

SpikingJelly 默认禁用 ``"spikingjelly"`` 命名空间，导入包不会输出日志，也不会添加、
删除或配置 sink。应用可以启用默认 sink：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

需要自定义输出时，由应用入口直接配置 Loguru：

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

``logger.remove()`` 会影响进程中的所有 Loguru sink，因此只能由应用调用。框架内部只使用
``logger.debug/info/warning/error/exception/critical`` 和 ``{}`` 参数化格式，不配置
sink，也不附加结构化上下文。使用标准库 ``logging`` 的应用可以在自己的入口按
`Loguru 标准 logging 兼容说明 <https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging>`_
完成桥接；SpikingJelly 不内置桥接层。

Common usage
------------

**English**

SpikingJelly exports Loguru's global logger directly:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("Model prepared: backend={} device={}", "torch", "cpu")

The ``"spikingjelly"`` namespace is disabled by default, so importing the package
emits nothing and does not add, remove, or configure sinks. Applications can enable
the default sink:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

Applications that need custom output configure Loguru at their entry point:

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.enable("spikingjelly")

``logger.remove()`` affects every Loguru sink in the process and therefore belongs
only in application code. Framework code only calls
``logger.debug/info/warning/error/exception/critical`` with ``{}``-style arguments;
it does not configure sinks or attach structured context. Applications built around
stdlib ``logging`` can add interoperability at their own entry point by following
Loguru's `standard logging compatibility guide <https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging>`_;
SpikingJelly does not provide a bridge.
