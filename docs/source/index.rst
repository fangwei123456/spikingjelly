.. _index:

欢迎来到SpikingJelly的文档
###################################

`SpikingJelly <https://github.com/fangwei123456/spikingjelly>`_ 是一个基于 `PyTorch <https://pytorch.org/>`_ ，使用脉冲神经\
网络(Spiking Neuron Network, SNN)进行深度学习的框架。

* :ref:`Homepage in English <index_en>`

安装
----------------

注意，SpikingJelly是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装spikingjelly。

从 `PyPI <https://pypi.org/project/spikingjelly/>`_ 安装：

.. code-block:: bash

    pip install spikingjelly

或者对于开发者，从Github最新的源代码进行安装：

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

上手教程
-------------------------
.. toctree::
   :maxdepth: 2

   tutorials


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

开发人员名单可见于 https://github.com/fangwei123456/spikingjelly/graphs/contributors。

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

Developers can download the latest version from GitHub and install:

.. code-block:: bash

    git clone https://github.com/fangwei123456/spikingjelly.git
    cd spikingjelly
    python setup.py install

Tutorials
-------------------------
.. toctree::
   :maxdepth: 2

   tutorials_en

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

The list of developers can be found at https://github.com/fangwei123456/spikingjelly/graphs/contributors.

.. _APIs:

APIs
###################

.. toctree::
   :maxdepth: 4

   APIs <spikingjelly>