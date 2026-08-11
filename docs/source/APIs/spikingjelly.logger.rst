spikingjelly.logger module
===========================

.. automodule:: spikingjelly.logger
   :members:
   :undoc-members:
   :show-inheritance:

常见用法
--------

**中文**

SpikingJelly 使用 Loguru 全局 logger。框架代码和应用代码都从同一模块导入：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("Model prepared: backend={} device={}", "torch", "cpu")

SpikingJelly 默认执行 ``logger.disable("spikingjelly")``，因此导入包不会输出日志。
模块不会添加或删除 sink，也不会修改标准库 root logger。应用启用默认彩色控制台只需：

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

``logging.basicConfig()``、``dictConfig()``、标准库 Handler/Filter 和 pytest
``caplog`` 不会控制或捕获 SpikingJelly 日志。Loguru 是进程级 singleton；
``logger.remove()`` 会删除进程内所有 Loguru sink，只能由应用入口调用，
SpikingJelly 内部不会调用。

应用自定义控制台
~~~~~~~~~~~~~~~~

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()  # 仅应用入口可以接管全局 sink
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.enable("spikingjelly")

生产环境应使用 ``diagnose=False``，避免异常日志输出局部变量、数据路径、token
或密钥。``logger.catch()`` 只应由应用入口使用，并设置 ``reraise=True``，不要装饰
框架内部函数以免改变异常传播。

JSONL 文件、轮转与保留
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikingjelly.logger import logger

    file_sink = logger.add(
        "logs/spikingjelly-{time:YYYY-MM-DD}.jsonl",
        level="INFO",
        serialize=True,
        rotation="100 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.enable("spikingjelly")

    # 应用退出前等待 enqueue=True 的记录写完
    logger.complete()
    logger.remove(file_sink)

结构化字段位于 JSON record 的 ``extra`` 中。框架生命周期摘要使用稳定的
``extra.event``，应用可用 ``bind()`` 或 ``contextualize()`` 追加运行上下文：

.. code-block:: python

    with logger.contextualize(run_id="experiment-42", epoch=3):
        logger.info("Epoch started")

DDP：rank-0 控制台与全 rank 文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import sys

    import torch.distributed as dist

    from spikingjelly.logger import logger

    rank = dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger.remove()  # 仅应用入口
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda record: record["extra"].get("rank") == 0,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | rank={extra[rank]} | <level>{message}</level>"
        ),
        colorize=True,
        diagnose=False,
    )
    logger.add(
        f"logs/train-rank{rank}.jsonl",
        serialize=True,
        enqueue=True,
        diagnose=False,
    )
    logger.enable("spikingjelly")

    with logger.contextualize(rank=rank, world_size=world_size):
        # 在此作用域运行训练/推理
        logger.info("Distributed worker ready")

    logger.complete()

性能与格式
~~~~~~~~~~

Loguru 使用 ``{}`` 参数化格式，不使用 ``%s``、f-string 或调用前 ``.format()``：

.. code-block:: python

    logger.info("Precision prepared: mode={} device={}", "fp32", "cpu")
    logger.opt(lazy=True).debug("Report={}", lambda: expensive_report())

``opt(lazy=True)`` 只用于避免昂贵计算或 I/O。不要在神经元 ``forward``、time-step、
batch、dispatch 或 CUDA/Triton kernel 热路径启用高频 INFO/DEBUG。终端、文件和网络
I/O 的开销远高于被禁用日志调用。

从标准库 logging 迁移
~~~~~~~~~~~~~~~~~~~~~~

- ``logger.info("x=%s", x)`` 改为 ``logger.info("x={}", x)``；
- ``extra={"event": name}`` 改为 ``logger.bind(event=name)``；
- 作用域上下文使用 ``logger.contextualize()``；
- pytest 测试添加临时 Loguru sink，并在结束时只移除自己得到的 sink ID；
- 只采集标准库 logging 的 Lightning、MLflow、Ray、torchrun 集成不会自动收到这些
  记录，应按对应工具的 stdout 或 Loguru sink 集成方式配置。

Common usage
------------

**English**

SpikingJelly uses Loguru's global logger. Framework and application code import the
same object:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.info("Model prepared: backend={} device={}", "torch", "cpu")

SpikingJelly calls ``logger.disable("spikingjelly")`` by default, so importing the
package emits nothing. The module neither adds nor removes sinks and does not modify
the stdlib root logger. Enable Loguru's default colored console with one line:

.. code-block:: python

    from spikingjelly.logger import logger

    logger.enable("spikingjelly")

``logging.basicConfig()``, ``dictConfig()``, stdlib handlers and filters, and pytest
``caplog`` do not control or capture SpikingJelly records. Loguru is a process-wide
singleton. ``logger.remove()`` removes every Loguru sink in the process, so only the
application entry point may call it; SpikingJelly never does.

Application-owned console
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import sys

    from spikingjelly.logger import logger

    logger.remove()  # Only the application entry point owns global sinks.
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.enable("spikingjelly")

Use ``diagnose=False`` in production to avoid exposing local variables, data paths,
tokens, or secrets. Use ``logger.catch(reraise=True)`` only at application entry
points; decorating framework internals could change exception propagation.

JSONL, rotation, and retention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikingjelly.logger import logger

    file_sink = logger.add(
        "logs/spikingjelly-{time:YYYY-MM-DD}.jsonl",
        level="INFO",
        serialize=True,
        rotation="100 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.enable("spikingjelly")

    logger.complete()  # Flush enqueue=True before application shutdown.
    logger.remove(file_sink)

Structured fields are stored in the JSON record's ``extra`` object. Framework
lifecycle summaries expose a stable ``extra.event``. Applications can add run
context with ``bind()`` or ``contextualize()``:

.. code-block:: python

    with logger.contextualize(run_id="experiment-42", epoch=3):
        logger.info("Epoch started")

DDP: rank-0 console and per-rank files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import sys

    import torch.distributed as dist

    from spikingjelly.logger import logger

    rank = dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger.remove()  # Application entry point only.
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda record: record["extra"].get("rank") == 0,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | rank={extra[rank]} | <level>{message}</level>"
        ),
        colorize=True,
        diagnose=False,
    )
    logger.add(
        f"logs/train-rank{rank}.jsonl",
        serialize=True,
        enqueue=True,
        diagnose=False,
    )
    logger.enable("spikingjelly")

    with logger.contextualize(rank=rank, world_size=world_size):
        logger.info("Distributed worker ready")
        # Run training or inference in this scope.

    logger.complete()

Performance and formatting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Loguru uses ``{}`` parameterized formatting. Do not use ``%s``, f-strings, or
``.format()`` before the call:

.. code-block:: python

    logger.info("Precision prepared: mode={} device={}", "fp32", "cpu")
    logger.opt(lazy=True).debug("Report={}", lambda: expensive_report())

Use ``opt(lazy=True)`` only to avoid expensive computation or I/O. Do not enable
high-frequency INFO or DEBUG records in neuron ``forward`` methods, time-step loops,
batch loops, dispatch paths, or CUDA/Triton kernel paths. Terminal, file, and network
I/O dominate the cost of disabled logging calls.

Migrating from stdlib logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Change ``logger.info("x=%s", x)`` to ``logger.info("x={}", x)``.
- Change ``extra={"event": name}`` to ``logger.bind(event=name)``.
- Use ``logger.contextualize()`` for scoped context.
- In pytest, add a temporary Loguru sink and remove only its returned sink ID.
- Lightning, MLflow, Ray, torchrun, and similar integrations that only collect stdlib
  logging do not receive these records automatically. Configure their stdout or
  Loguru sink integration explicitly.
