spikingjelly.logger module
===========================

.. automodule:: spikingjelly.logger
   :members:
   :undoc-members:
   :show-inheritance:

常见用法
--------

**中文**

SpikingJelly 提供一个名称固定为 ``"spikingjelly"`` 的共享 Logger。Logger 名称
区分大小写；``logging.getLogger("SpikingJelly")`` 与
``logging.getLogger("spikingjelly")`` 是不同的 Logger。框架代码和应用代码都应
从模块中导入共享变量：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("model prepared backend=%s", "torch")

``spikingjelly.logger`` 只安装一个 ``NullHandler``。它不输出内容，也不配置 root
logger、级别、格式或输出位置，因此 import SpikingJelly 不会改变应用的 logging
配置。它不会阻止记录传播到应用配置的 root handler；没有应用侧 Handler 时，默认
不会显示 SpikingJelly 日志。应用可以在入口配置 root logger：

.. code-block:: python

    import logging

    from spikingjelly.logger import logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("device selected device=%s", "cpu")

Logger 和 Handler 都有最低级别门槛。Logger 决定记录是否继续处理，Handler 决定
已经创建的记录是否写入自己的输出端。多个 Handler 可以为同一个 Logger 提供不同
输出：

.. code-block:: python

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    error_file = logging.FileHandler("spikingjelly-error.log")
    error_file.setLevel(logging.ERROR)

    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(error_file)

此时 INFO 会进入控制台，只有 ERROR 及以上会写入文件。若应用只使用专用 Handler，
应设置 ``logger.propagate = False``，避免同时被 root handler 输出。重复执行装配
代码时应复用 Handler，避免同一条记录重复输出。

性能方面，热路径不应启用 INFO 日志；若确实需要在神经元 ``forward``、time-step、
batch、dispatch 或 CUDA/Triton kernel 路径保留 DEBUG 诊断，必须使用参数化格式，
并且只有在需要避免昂贵计算或 I/O 时才使用 ``logger.isEnabledFor``：

.. code-block:: python

    import logging

    from spikingjelly.logger import logger

    # 推荐：INFO 被过滤时不会构造格式化后的消息
    logger.info("precision prepared mode=%s device=%s", "fp32", "cpu")

    # 只有 expensive_report() 很昂贵时，额外判断才有意义
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("report=%s", expensive_report())

应用侧 Handler 可能执行格式化、终端 I/O、文件 I/O 或网络 I/O；这些开销远高于
``NullHandler``，不应在每个 batch 或每个时间步启用 INFO 日志。

Common usage
------------

**English**

SpikingJelly provides one shared Logger with the fixed name ``"spikingjelly"``.
Logger names are case-sensitive; ``logging.getLogger("SpikingJelly")`` and
``logging.getLogger("spikingjelly")`` are different loggers. Framework and
application code should import the shared variable from the module:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("model prepared backend=%s", "torch")

``spikingjelly.logger`` installs only one ``NullHandler``. It emits no output and
does not configure the root logger, level, formatter, or destination, so importing
SpikingJelly does not change the application's logging configuration. It does not
block records from propagating to an application-configured root handler; with no
application Handler, SpikingJelly output is invisible by default. Configure the root
logger at the application entry point:

.. code-block:: python

    import logging

    from spikingjelly.logger import logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("device selected device=%s", "cpu")

Logger and Handler levels are independent minimum thresholds. The Logger decides
whether a record continues through logging; each Handler decides whether that record
is emitted by its output. Multiple Handlers can provide different destinations for
the same Logger:

.. code-block:: python

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    error_file = logging.FileHandler("spikingjelly-error.log")
    error_file.setLevel(logging.ERROR)

    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(error_file)

INFO is emitted by the console, while only ERROR and above are written to the file.
When an application uses only dedicated Handlers, set ``logger.propagate = False``
to avoid duplicate output through the root Handler. Reuse Handlers when setup code can
run more than once.

For performance, do not enable INFO logging in hot paths. If DEBUG diagnostics are
needed in neuron ``forward`` methods, time-step loops, batch loops, dispatch paths,
or CUDA/Triton kernel paths, use parameterized messages; use
``logger.isEnabledFor`` only when it avoids an expensive calculation or I/O
operation:

.. code-block:: python

    import logging

    from spikingjelly.logger import logger

    # Preferred: filtered INFO does not build the formatted message
    logger.info("precision prepared mode=%s device=%s", "fp32", "cpu")

    # The extra check is useful only when expensive_report() is expensive
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("report=%s", expensive_report())

Application Handlers may perform formatting, terminal I/O, file I/O, or network I/O;
these costs are much higher than a ``NullHandler``. Do not enable INFO logging in
per-batch or per-time-step paths.
