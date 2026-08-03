spikingjelly.logger module
===========================

The module exposes one package-level logger named ``spikingjelly``. Import the
shared variable at module scope:

.. code-block:: python

    from spikingjelly.logger import logger

    backend = "torch"
    logger.info("model prepared backend=%s", backend)

Do not configure the root logger or add application-owned output handlers from
library code. ``spikingjelly.logger`` installs a ``NullHandler`` so importing
SpikingJelly does not produce output or change the host application's logging
configuration.

.. automodule:: spikingjelly.logger
   :members:
   :undoc-members:
   :show-inheritance:

常见用法
--------

默认情况下，框架只产生 LogRecord；是否显示由应用程序决定。在程序入口配置
root logger 时，包级日志会通过默认的 ``propagate=True`` 传播到 root handler：

.. code-block:: python

    import logging

    import torch

    from spikingjelly.logger import logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("device selected device=%s", torch.device("cpu"))

如果希望只为 SpikingJelly 添加专用输出，可以在应用侧添加 Handler，并关闭向
root 的传播以避免重复输出：

.. code-block:: python

    import logging

    from spikingjelly import logger

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.propagate = False

Logger 和 Handler 的 level 都是最低级别门槛。Logger 的 level 决定是否创建并
处理记录，Handler 的 level 决定该输出端是否处理已经创建的记录：

.. code-block:: python

    logger.setLevel(logging.INFO)
    console.setLevel(logging.WARNING)

    logger.info("this record is filtered by console")
    logger.warning("this record is emitted by console")

要同时输出到不同目标，可以为同一个 logger 添加多个 Handler，并分别设置门槛：

.. code-block:: python

    import logging

    from spikingjelly import logger

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    error_file = logging.FileHandler("spikingjelly-error.log")
    error_file.setLevel(logging.ERROR)

    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(error_file)

这里 INFO 会进入控制台，只有 ERROR 及以上会写入文件。应用如果反复执行装配
代码，应保存并复用 Handler，或先确认尚未添加，避免同一条日志重复输出。

性能注意事项
--------------

日志不应放入神经元 ``forward``、time-step、batch、``__torch_dispatch__``、
``__torch_function__`` 或 CUDA/Triton kernel 路径。低频生命周期日志使用 logging
的参数化格式，避免 f-string 和 ``.format()``：

.. code-block:: python

    # 推荐：被 level 过滤时不格式化消息
    mode = "fp32"
    device = "cpu"
    logger.info("precision prepared mode=%s device=%s", mode, device)

    # 不推荐：即使 INFO 被关闭，也会先构造字符串
    logger.info(f"precision prepared mode={mode} device={device}")

应用侧 Handler 可能执行格式化、终端 I/O、文件 I/O 或网络 I/O；这些开销远高于
``NullHandler``，因此不应在每个 batch 或每个时间步启用 INFO 日志。

Common usage
------------

The logger uses the exact name ``"spikingjelly"``. Logger names are case-sensitive:
``logging.getLogger("SpikingJelly")`` and ``logging.getLogger("spikingjelly")``
are different objects. Repeated calls with the same exact name return the cached
logger, but importing the shared variable makes the package convention explicit:

.. code-block:: python

    from spikingjelly.logger import logger

    backend = "torch"
    logger.warning("fallback selected backend=%s", backend)

``NullHandler`` is a no-op output handler. It prevents the library from configuring
application output during import; it does not suppress records that propagate to a
root handler configured by the application. A clean process with no application
handler produces no visible SpikingJelly output by default.
