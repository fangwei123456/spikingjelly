中文教程 | Chinese Tutorials
===============================================

先按 :doc:`./install` 准备环境，再读 :doc:`./basic_concept` 和 :doc:`./neuron`，
了解步进模式、神经元状态和重置。:doc:`./lif_fc_mnist` 是第一个完整的训练示例。
在 GPU 上加速或编译网络见 :doc:`./triton_backend`，旧版本代码迁移见
:doc:`./migrate_from_legacy`。

.. toctree::
    :maxdepth: 1
    :caption: 基础

    /tutorials/cn/install
    /tutorials/cn/basic_concept
    /tutorials/cn/container
    /tutorials/cn/neuron
    /tutorials/cn/surrogate
    /tutorials/cn/monitor

.. toctree::
    :maxdepth: 1
    :caption: 训练第一个 SNN

    /tutorials/cn/lif_fc_mnist
    /tutorials/cn/conv_fashion_mnist
    /tutorials/cn/recurrent_connection_and_stateful_synapse
    /tutorials/cn/neuromorphic_datasets
    /tutorials/cn/classify_dvsg

.. toctree::
    :maxdepth: 1
    :caption: 学习规则、模型与转换

    /tutorials/cn/stdp
    /tutorials/cn/timing_based
    /tutorials/cn/ann2snn
    /tutorials/cn/ann2snn_transformer
    /tutorials/cn/spikformer

.. toctree::
    :maxdepth: 1
    :caption: 性能与大规模训练

    /tutorials/cn/triton_backend
    /tutorials/cn/flexsn
    /tutorials/cn/memopt
    /tutorials/cn/precision
    /tutorials/cn/op_counter
    /tutorials/cn/distributed_training
    /tutorials/cn/train_large_scale_snn

.. toctree::
    :maxdepth: 1
    :caption: 硬件部署与模型交换

    /tutorials/cn/inference_on_lynxi
    /tutorials/cn/lava_exchange
    /tutorials/cn/nir_exchange

.. toctree::
    :maxdepth: 1
    :caption: 强化学习应用

    /tutorials/cn/dsqn
    /tutorials/cn/ilc_san

.. toctree::
    :maxdepth: 1
    :caption: 版本迁移

    /tutorials/cn/migrate_from_legacy
