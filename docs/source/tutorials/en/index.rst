英文教程 | English Tutorials
========================================

Prepare your environment with :doc:`./install`, then read :doc:`./basic_concept`
and :doc:`./neuron` to learn about step modes, neuron state and resetting.
:doc:`./lif_fc_mnist` is the first complete training example. See
:doc:`./triton_backend` to accelerate or compile a network on a GPU, and
:doc:`./migrate_from_legacy` to migrate code from older versions.

The following tutorials are currently available only in Chinese: :doc:`DSQN <../cn/dsqn>`, :doc:`ILC-SAN <../cn/ilc_san>`, and :doc:`Lynxi inference <../cn/inference_on_lynxi>`.

.. toctree::
    :maxdepth: 1
    :caption: Basics

    /tutorials/en/install
    /tutorials/en/basic_concept
    /tutorials/en/container
    /tutorials/en/neuron
    /tutorials/en/surrogate
    /tutorials/en/monitor

.. toctree::
    :maxdepth: 1
    :caption: Training your first SNN

    /tutorials/en/lif_fc_mnist
    /tutorials/en/conv_fashion_mnist
    /tutorials/en/recurrent_connection_and_stateful_synapse
    /tutorials/en/neuromorphic_datasets
    /tutorials/en/classify_dvsg

.. toctree::
    :maxdepth: 1
    :caption: Learning rules, models and conversion

    /tutorials/en/stdp
    /tutorials/en/timing_based
    /tutorials/en/ann2snn
    /tutorials/en/ann2snn_transformer
    /tutorials/en/spikformer

.. toctree::
    :maxdepth: 1
    :caption: Performance and scale

    /tutorials/en/triton_backend
    /tutorials/en/flexsn
    /tutorials/en/memopt
    /tutorials/en/precision
    /tutorials/en/op_counter
    /tutorials/en/distributed_training
    /tutorials/en/train_large_scale_snn

.. toctree::
    :maxdepth: 1
    :caption: Hardware and model exchange

    /tutorials/en/lava_exchange
    /tutorials/en/nir_exchange

.. toctree::
    :maxdepth: 1
    :caption: Version migration

    /tutorials/en/migrate_from_legacy
