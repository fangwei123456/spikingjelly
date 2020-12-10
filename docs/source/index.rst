.. _index:

欢迎来到SpikingJelly的文档
###################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ ，使用脉冲神经\
网络(Spiking Neural Network, SNN)进行深度学习的框架。

* :ref:`Homepage in English <index_en>`

安装
----------------

注意，SpikingJelly是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装spikingjelly。

从 `PyPI <https://pypi.org/project/spikingjelly/>`_ 安装：

.. code-block:: bash

    pip install spikingjelly

或者对于开发者，下载源代码并安装：

通过 `GitHub <https://github.com/fangwei123456/spikingjelly>`_：

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

通过 `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_ ：

.. code-block:: bash

    git clone http://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    python setup.py install



.. toctree::
    :maxdepth: 1
    :caption: 上手教程

    tutorial.clock_driven
    tutorial.event_driven
    /clock_driven/0_neuron
    /clock_driven/2_encoding
    /clock_driven/3_fc_mnist
    /clock_driven/4_conv_fashion_mnist
    /clock_driven/5_ann2snn
    /clock_driven/6_dqn_cart_pole
    /clock_driven/7_a2c_cart_pole
    /clock_driven/8_ppo_cart_pole
    /clock_driven/9_spikingLSTM_text
    /clock_driven/10_forward_pattern


模块文档
-------------------------

* :ref:`APIs`

文档索引
-------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


项目信息
-------------------------
北京大学信息科学技术学院数字媒体所媒体学习组 `Multimedia Learning Group <https://pkuml.org/>`_ 和 `鹏城实验室 <http://www.pcl.ac.cn/>`_ 是SpikingJelly的主要开发者。

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

开发人员名单可见于 `贡献者 <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_ 。

.. _index_en:

Welcome to SpikingJelly's documentation
############################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ is an open-source deep learning framework for Spiking Neural Network (SNN) based on `PyTorch <https://pytorch.org/>`_.

* :ref:`中文首页 <index>`

Installation
----------------

Note that SpikingJelly is based on PyTorch. Please make sure that you have installed PyTorch before you install SpikingJelly.

Install from `PyPI <https://pypi.org/project/spikingjelly/>`_：

.. code-block:: bash

    pip install spikingjelly

Developers can download and install the latest version from source codes:

From `GitHub <https://github.com/fangwei123456/spikingjelly>`_:

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

From `OpenI <https://git.openi.org.cn/OpenI/spikingjelly>`_：

.. code-block:: bash

    git clone http://git.openi.org.cn/OpenI/spikingjelly.git
    cd spikingjelly
    python setup.py install


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorial.clock_driven_en
    /clock_driven_en/0_neuron
    /clock_driven_en/2_encoding
    /clock_driven_en/3_fc_mnist
    /clock_driven_en/4_conv_fashion_mnist
    /clock_driven_en/5_ann2snn
    /clock_driven_en/6_dqn_cart_pole
    /clock_driven_en/7_a2c_cart_pole
    /clock_driven_en/8_ppo_cart_pole
    /clock_driven_en/9_spikingLSTM_text



Modules Docs
-------------------------

* :ref:`APIs`

Indices and tables
-------------------------

* :ref:`Index <genindex>`
* :ref:`Module Index <modindex>`
* :ref:`Search Page <search>`


About
-------------------------
`Multimedia Learning Group, Institute of Digital Media (NELVT), Peking University <https://pkuml.org/>`_ and `Peng Cheng Laboratory <http://www.szpclab.com/>`_ are the main developers of SpikingJelly.

.. image:: ./_static/logo/pku.png
    :width: 20%

.. image:: ./_static/logo/pcl.png
    :width: 20%

The list of developers can be found at `contributors <https://github.com/fangwei123456/spikingjelly/graphs/contributors>`_.

.. _APIs:
.. toctree::
   :maxdepth: 4
   :caption: APIs

   spikingjelly.clock_driven
   spikingjelly.datasets
   spikingjelly.event_driven
   spikingjelly.visualizing
   spikingjelly.cext